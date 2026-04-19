//=- PoseidonMultiFloat.cpp - Double-single arithmetic for Poseidon -------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PoseidonMultiFloat.h"
#include "../Utils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Type.h"

using namespace llvm;

static Type *f32Ty(IRBuilder<> &B) { return B.getFloatTy(); }
static Type *f64Ty(IRBuilder<> &B) { return B.getDoubleTy(); }

DSValue emitTwoSum(IRBuilder<> &B, Value *a, Value *b) {
  Value *s = B.CreateFAdd(a, b, "ts.s");
  Value *a_prime = B.CreateFSub(s, b, "ts.ap");
  Value *b_prime = B.CreateFSub(s, a_prime, "ts.bp");
  Value *da = B.CreateFSub(a, a_prime, "ts.da");
  Value *db = B.CreateFSub(b, b_prime, "ts.db");
  Value *e = B.CreateFAdd(da, db, "ts.e");
  return {s, e};
}

DSValue emitFastTwoSum(IRBuilder<> &B, Value *a, Value *b) {
  Value *s = B.CreateFAdd(a, b, "fts.s");
  Value *bp = B.CreateFSub(s, a, "fts.bp");
  Value *e = B.CreateFSub(b, bp, "fts.e");
  return {s, e};
}

DSValue emitTwoProdFMA(IRBuilder<> &B, Value *a, Value *b) {
  Value *p = B.CreateFMul(a, b, "tp.p");
  Value *neg_p = B.CreateFNeg(p, "tp.np");
  Module *M = B.GetInsertBlock()->getParent()->getParent();
  Function *fma_fn = getIntrinsicDeclaration(M, Intrinsic::fma, {f32Ty(B)});
  Value *e = B.CreateCall(fma_fn, {a, b, neg_p}, "tp.e");
  return {p, e};
}

DSValue emitDSAdd(IRBuilder<> &B, DSValue x, DSValue y) {
  DSValue ab = emitTwoSum(B, x.hi, y.hi);
  DSValue cd = emitTwoSum(B, x.lo, y.lo);
  DSValue ac = emitFastTwoSum(B, ab.hi, cd.hi);
  Value *bd = B.CreateFAdd(ab.lo, cd.lo, "dsa.bd");
  Value *b3 = B.CreateFAdd(bd, ac.lo, "dsa.b3");
  return emitFastTwoSum(B, ac.hi, b3);
}

DSValue emitDSSub(IRBuilder<> &B, DSValue x, DSValue y) {
  DSValue neg_y = emitDSNeg(B, y);
  return emitDSAdd(B, x, neg_y);
}

DSValue emitDSMul(IRBuilder<> &B, DSValue x, DSValue y) {
  DSValue pe = emitTwoProdFMA(B, x.hi, y.hi);
  Value *c1 = B.CreateFMul(x.hi, y.lo, "dsm.c1");
  Value *c2 = B.CreateFMul(x.lo, y.hi, "dsm.c2");
  Value *cross = B.CreateFAdd(c1, c2, "dsm.cr");
  Value *e2 = B.CreateFAdd(pe.lo, cross, "dsm.e2");
  return emitFastTwoSum(B, pe.hi, e2);
}

DSValue emitDSDiv(IRBuilder<> &B, DSValue x, DSValue y) {
  Value *z_hi = B.CreateFDiv(x.hi, y.hi, "dsd.zhi");
  DSValue pe = emitTwoProdFMA(B, z_hi, y.hi);
  Value *d1 = B.CreateFSub(x.hi, pe.hi, "dsd.d1");
  Value *d2 = B.CreateFSub(d1, pe.lo, "dsd.d2");
  Value *d3 = B.CreateFAdd(d2, x.lo, "dsd.d3");
  Value *q = B.CreateFMul(z_hi, y.lo, "dsd.q");
  Value *d4 = B.CreateFSub(d3, q, "dsd.d4");
  Value *z_lo = B.CreateFDiv(d4, y.hi, "dsd.zlo");
  return emitFastTwoSum(B, z_hi, z_lo);
}

DSValue emitDSSqrt(IRBuilder<> &B, DSValue x) {
  Module *M = B.GetInsertBlock()->getParent()->getParent();
  Function *sqrtf_fn = getIntrinsicDeclaration(M, Intrinsic::sqrt, {f32Ty(B)});
  Value *z_hi = B.CreateCall(sqrtf_fn, {x.hi}, "dssq.zhi");
  DSValue pe = emitTwoProdFMA(B, z_hi, z_hi);
  Value *d1 = B.CreateFSub(x.hi, pe.hi, "dssq.d1");
  Value *d2 = B.CreateFSub(d1, pe.lo, "dssq.d2");
  Value *d3 = B.CreateFAdd(d2, x.lo, "dssq.d3");
  Value *two_z = B.CreateFMul(ConstantFP::get(f32Ty(B), 2.0), z_hi, "dssq.2z");
  Value *z_lo = B.CreateFDiv(d3, two_z, "dssq.zlo");
  return emitFastTwoSum(B, z_hi, z_lo);
}

DSValue emitDSNeg(IRBuilder<> &B, DSValue x) {
  Value *neg_hi = B.CreateFNeg(x.hi, "dsn.hi");
  Value *neg_lo = B.CreateFNeg(x.lo, "dsn.lo");
  return {neg_hi, neg_lo};
}

DSValue emitF64ToDS(IRBuilder<> &B, Value *f64val) {
  Value *hi = B.CreateFPTrunc(f64val, f32Ty(B), "ds.hi");
  Value *hi_back = B.CreateFPExt(hi, f64Ty(B), "ds.hib");
  Value *lo_src = B.CreateFSub(f64val, hi_back, "ds.los");
  Value *lo = B.CreateFPTrunc(lo_src, f32Ty(B), "ds.lo");
  return {hi, lo};
}

Value *emitDSToF64(IRBuilder<> &B, DSValue ds) {
  Value *hi64 = B.CreateFPExt(ds.hi, f64Ty(B), "ds.hi64");
  Value *lo64 = B.CreateFPExt(ds.lo, f64Ty(B), "ds.lo64");
  return B.CreateFAdd(hi64, lo64, "ds.f64");
}

static DSValue getOrSplitOperand(IRBuilder<> &B, Value *op,
                                 DenseMap<Value *, DSValue> &dsMap) {
  if (dsMap.count(op))
    return dsMap[op];

  if (auto *CFP = dyn_cast<ConstantFP>(op)) {
    double val = CFP->getValueAPF().convertToDouble();
    float hi = (float)val;
    float lo = (float)(val - (double)hi);
    DSValue ds;
    ds.hi = ConstantFP::get(f32Ty(B), hi);
    ds.lo = ConstantFP::get(f32Ty(B), lo);
    return ds;
  }

  return emitF64ToDS(B, op);
}

static DSValue emitDSForInstruction(IRBuilder<> &B, Instruction *I,
                                    DenseMap<Value *, DSValue> &dsMap) {
  unsigned opcode = I->getOpcode();

  if (auto *BO = dyn_cast<BinaryOperator>(I)) {
    DSValue lhs = getOrSplitOperand(B, BO->getOperand(0), dsMap);
    DSValue rhs = getOrSplitOperand(B, BO->getOperand(1), dsMap);

    switch (opcode) {
    case Instruction::FAdd:
      return emitDSAdd(B, lhs, rhs);
    case Instruction::FSub:
      return emitDSSub(B, lhs, rhs);
    case Instruction::FMul:
      return emitDSMul(B, lhs, rhs);
    case Instruction::FDiv:
      return emitDSDiv(B, lhs, rhs);
    default:
      break;
    }
  }

  if (auto *UO = dyn_cast<UnaryOperator>(I)) {
    if (opcode == Instruction::FNeg) {
      DSValue x = getOrSplitOperand(B, UO->getOperand(0), dsMap);
      return emitDSNeg(B, x);
    }
  }

  if (auto *CI = dyn_cast<CallInst>(I)) {
    Function *callee = CI->getCalledFunction();
    if (!callee)
      return {nullptr, nullptr};

    StringRef mathName;
    if (callee->isIntrinsic()) {
      Intrinsic::ID id = callee->getIntrinsicID();
      if (id == Intrinsic::sqrt)
        mathName = "sqrt";
      else if (id == Intrinsic::fmuladd)
        mathName = "fmuladd";
      else if (id == Intrinsic::fma)
        mathName = "fma";
    } else {
      if (callee->hasFnAttribute("enzyme_math")) {
        mathName = callee->getFnAttribute("enzyme_math").getValueAsString();
      } else {
        StringRef name = callee->getName();
        if (name.starts_with("__nv_"))
          name = name.drop_front(5);
        if (!name.empty() && (name.back() == 'f' || name.back() == 'l'))
          name = name.drop_back(1);
        mathName = name;
      }
    }

    if (mathName == "sqrt") {
      DSValue x = getOrSplitOperand(B, CI->getArgOperand(0), dsMap);
      return emitDSSqrt(B, x);
    }
    if (mathName == "fmuladd" || mathName == "fma") {
      DSValue a = getOrSplitOperand(B, CI->getArgOperand(0), dsMap);
      DSValue b = getOrSplitOperand(B, CI->getArgOperand(1), dsMap);
      DSValue c = getOrSplitOperand(B, CI->getArgOperand(2), dsMap);
      DSValue ab = emitDSMul(B, a, b);
      return emitDSAdd(B, ab, c);
    }
  }

  return {nullptr, nullptr};
}

void applyMultiFloat(ArrayRef<Instruction *> instsToChange,
                     const SmallPtrSetImpl<Instruction *> &allChanged,
                     DenseMap<Value *, Value *> *restoredValues) {
  DenseMap<Value *, DSValue> dsMap;

  for (Instruction *I : instsToChange) {
    IRBuilder<> B(I);
    B.clearFastMathFlags();

    DSValue ds = emitDSForInstruction(B, I, dsMap);
    if (!ds.hi) {
      errs() << "MultiFloat: unsupported instruction, skipping: " << *I << "\n";
      continue;
    }

    dsMap[I] = ds;
  }

  for (auto &[val, ds] : dsMap) {
    Instruction *I = cast<Instruction>(val);
    SmallVector<Use *, 4> externalUses;
    for (Use &U : I->uses()) {
      auto *userI = dyn_cast<Instruction>(U.getUser());
      if (!userI || !dsMap.count(userI))
        externalUses.push_back(&U);
    }

    if (!externalUses.empty()) {
      IRBuilder<> RestoreB(I->getParent(), ++BasicBlock::iterator(I));
      Value *restored = emitDSToF64(RestoreB, ds);
      for (Use *U : externalUses)
        U->set(restored);
      if (restoredValues)
        (*restoredValues)[I] = restored;
    }
  }

  for (auto it = instsToChange.rbegin(); it != instsToChange.rend(); ++it) {
    Instruction *I = *it;
    if (!dsMap.count(I))
      continue;
    if (!I->use_empty()) {
      if (restoredValues && restoredValues->count(I))
        I->replaceAllUsesWith((*restoredValues)[I]);
      else {
        errs() << "MultiFloat: replacing with undef (remaining uses): " << *I
               << "\n";
        for (auto &U : I->uses())
          errs() << "  used by: " << *U.getUser() << "\n";
        I->replaceAllUsesWith(UndefValue::get(I->getType()));
      }
    }
    I->eraseFromParent();
  }
}
