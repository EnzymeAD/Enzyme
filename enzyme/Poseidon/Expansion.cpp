//=- Expansion.cpp - Double-single arithmetic for Poseidon ----------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Expansion.h"
#include "CostModel.h"
#include "Flags.h"
#include "Optimize.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

namespace poseidon {

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
  Function *fma_fn =
      Intrinsic::getOrInsertDeclaration(M, Intrinsic::fma, {f32Ty(B)});
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

// Fused double-single FMA (16 F32 ops against 30 for emitDSMul + emitDSAdd).
// The product's renormalizing fastTwoSum is dead work when the pair is consumed
// by an addition, the cross terms become hardware FMAs into the residual, and
// the addition's second twoSum over the low limbs is dropped because after the
// exact twoSum of the high parts every remaining term is O(eps) relative to
// max(|x*y|, |z|). x.lo*y.lo is dropped as in emitDSMul. Low-limb accumulation
// is ordered smallest-magnitude first.
DSValue emitDSFMA(IRBuilder<> &B, DSValue x, DSValue y, DSValue z) {
  Module *M = B.GetInsertBlock()->getParent()->getParent();
  Function *fma_fn =
      Intrinsic::getOrInsertDeclaration(M, Intrinsic::fma, {f32Ty(B)});

  // p + e == x.hi*y.hi exactly.
  Value *p = B.CreateFMul(x.hi, y.hi, "dsf.p");
  Value *neg_p = B.CreateFNeg(p, "dsf.np");
  Value *e = B.CreateCall(fma_fn, {x.hi, y.hi, neg_p}, "dsf.e");
  e = B.CreateCall(fma_fn, {x.hi, y.lo, e}, "dsf.e1");
  e = B.CreateCall(fma_fn, {x.lo, y.hi, e}, "dsf.e2");

  // s.hi + s.lo == p + z.hi exactly (captures catastrophic cancellation).
  DSValue s = emitTwoSum(B, p, z.hi);

  Value *t = B.CreateFAdd(e, z.lo, "dsf.t0");
  t = B.CreateFAdd(t, s.lo, "dsf.t1");
  return emitFastTwoSum(B, s.hi, t);
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
  Function *sqrtf_fn;
  if (Triple(M->getTargetTriple()).isNVPTX())
    sqrtf_fn =
        Intrinsic::getOrInsertDeclaration(M, Intrinsic::nvvm_sqrt_approx_f, {});
  else
    sqrtf_fn =
        Intrinsic::getOrInsertDeclaration(M, Intrinsic::sqrt, {f32Ty(B)});
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
  assert(f64val->getType()->isDoubleTy() && "emitF64ToDS: input must be f64");
  Value *hi = B.CreateFPTrunc(f64val, f32Ty(B), "ds.hi");
  Value *hi_back = B.CreateFPExt(hi, f64Ty(B), "ds.hib");
  Value *lo_src = B.CreateFSub(f64val, hi_back, "ds.los");
  Value *lo = B.CreateFPTrunc(lo_src, f32Ty(B), "ds.lo");
  return {hi, lo};
}

Value *emitDSToF64(IRBuilder<> &B, DSValue ds) {
  Value *hi64 = B.CreateFPExt(ds.hi, f64Ty(B), "ds.hi64");
  Value *lo64 = B.CreateFPExt(ds.lo, f64Ty(B), "ds.lo64");
  Value *joined = B.CreateFAdd(hi64, lo64, "ds.f64");
  // Tag the restore so foldDSPairRoundtrip can recover {hi,lo} exactly: every
  // DS value here is normalized, so a Dekker split of this restore returns the
  // same two limbs. Only tagged values are folded.
  if (auto *I = dyn_cast<Instruction>(joined))
    I->setMetadata("poseidon.ds.join", MDNode::get(I->getContext(), {}));
  return joined;
}

// Cancel a Dekker split applied to a tagged df64 restore (the df64 staging
// narrowing splits the stored value, and when that value is itself a restore
// the join and split cancel). Exact for a normalized pair, bit for bit; gated
// on the poseidon.ds.join tag. InstCombine will not do it because the split
// reads the join twice. `foldedLimbs` collects the limbs that replace each
// folded join, which the cost walk must re-root on (the old F64 output is
// dead).
bool foldDSPairRoundtrip(Function &F, SmallVectorImpl<Value *> *foldedLimbs) {
  SmallVector<Instruction *, 32> joins;
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      if (I.getMetadata("poseidon.ds.join"))
        joins.push_back(&I);

  bool changed = false;
  for (Instruction *J : joins) {
    auto *ehi = dyn_cast<FPExtInst>(J->getOperand(0));
    auto *elo = dyn_cast<FPExtInst>(J->getOperand(1));
    if (!ehi || !elo || ehi == elo)
      continue;
    Value *hiLimb = ehi->getOperand(0), *loLimb = elo->getOperand(0);
    if (!hiLimb->getType()->isFloatTy() || !loLimb->getType()->isFloatTy())
      continue;

    // Users of J must be exactly the Dekker split:
    //   hiT = fptrunc J ; los = fsub J, (fpext hiT) ; loT = fptrunc los
    FPTruncInst *hiT = nullptr, *loT = nullptr;
    BinaryOperator *los = nullptr;
    bool shapeOk = true;
    for (User *U : J->users()) {
      if (auto *FT = dyn_cast<FPTruncInst>(U)) {
        if (hiT || !FT->getType()->isFloatTy()) {
          shapeOk = false;
          break;
        }
        hiT = FT;
      } else if (auto *BO = dyn_cast<BinaryOperator>(U)) {
        if (los || BO->getOpcode() != Instruction::FSub ||
            !BO->getType()->isDoubleTy()) {
          shapeOk = false;
          break;
        }
        los = BO;
      } else {
        shapeOk = false;
        break;
      }
    }
    if (!shapeOk || !hiT || !los || los->getOperand(0) != J)
      continue;
    auto *hib = dyn_cast<FPExtInst>(los->getOperand(1));
    if (!hib || hib->getOperand(0) != hiT || !hib->hasOneUse() ||
        !los->hasOneUse())
      continue;
    loT = dyn_cast<FPTruncInst>(*los->user_begin());
    if (!loT || !loT->getType()->isFloatTy())
      continue;

    loT->replaceAllUsesWith(loLimb);
    loT->eraseFromParent();
    los->eraseFromParent();
    hib->eraseFromParent();
    hiT->replaceAllUsesWith(hiLimb);
    hiT->eraseFromParent();
    if (foldedLimbs) {
      foldedLimbs->push_back(hiLimb);
      foldedLimbs->push_back(loLimb);
    }
    if (J->use_empty()) {
      J->eraseFromParent();
      if (ehi->use_empty())
        ehi->eraseFromParent();
      if (elo->use_empty())
        elo->eraseFromParent();
    }
    changed = true;
  }
  if (changed && flags::Print)
    llvm::errs()
        << "[poseidon] folded df64 join/split roundtrips at the staging "
           "boundary in "
        << F.getName() << "\n";
  return changed;
}

// F32 source: exact in the hi slot, lo = 0.
static DSValue emitF32ToDS(IRBuilder<> &B, Value *f32val) {
  assert(f32val->getType()->isFloatTy() && "emitF32ToDS: input must be f32");
  return {f32val, ConstantFP::get(f32Ty(B), 0.0)};
}

// F32 target: sum the halves; as accurate as F32 allows.
static Value *emitDSToF32(IRBuilder<> &B, DSValue ds) {
  return B.CreateFAdd(ds.hi, ds.lo, "ds.f32");
}

DSValue emitToDS(IRBuilder<> &B, Value *fpval) {
  Type *T = fpval->getType();
  if (T->isFloatTy())
    return emitF32ToDS(B, fpval);
  if (T->isDoubleTy())
    return emitF64ToDS(B, fpval);
  llvm_unreachable("emitToDS: only f32 and f64 sources are supported");
}

Value *emitDSToFP(IRBuilder<> &B, DSValue ds, Type *targetTy) {
  if (targetTy->isFloatTy())
    return emitDSToF32(B, ds);
  if (targetTy->isDoubleTy())
    return emitDSToF64(B, ds);
  llvm_unreachable("emitDSToFP: only f32 and f64 targets are supported");
}

static DSValue splitConstantFP(IRBuilder<> &B, ConstantFP *CFP) {
  double val = CFP->getValueAPF().convertToDouble();
  float hi = (float)val;
  float lo = (float)(val - (double)hi);
  DSValue ds;
  ds.hi = ConstantFP::get(f32Ty(B), hi);
  ds.lo = ConstantFP::get(f32Ty(B), lo);
  return ds;
}

static DSValue getOrSplitOperand(IRBuilder<> &B, Value *op,
                                 DenseMap<Value *, DSValue> &dsMap);

// Loop-carried phi incomings are deferred: the accumulator phi and the
// accumulation feeding it are mutually dependent, and resolving the back edge
// eagerly would collapse the phi to a single double.
namespace {
struct DeferredDSPhiIn {
  PHINode *hiPhi, *loPhi;
  Value *incoming;
  BasicBlock *block;
  FastMathFlags fmf;
};
} // namespace
static SmallVector<DeferredDSPhiIn, 16> &deferredDSPhiIns() {
  static SmallVector<DeferredDSPhiIn, 16> v;
  return v;
}

// Any scalar f32/f64 value has a DS representation (materialized op, split
// constant, carried phi, or boundary emitToDS).
static bool hasDSRepresentation(Value *op) {
  Type *T = op->getType();
  return T->isFloatTy() || T->isDoubleTy();
}

static DSValue getOrSplitOperand(IRBuilder<> &B, Value *op,
                                 DenseMap<Value *, DSValue> &dsMap) {
  if (auto it = dsMap.find(op); it != dsMap.end())
    return it->second;

  if (auto *CFP = dyn_cast<ConstantFP>(op))
    return splitConstantFP(B, CFP);

  if (auto *phi = dyn_cast<PHINode>(op);
      phi && (phi->getType()->isFloatTy() || phi->getType()->isDoubleTy())) {
    bool allReachable = true;
    for (Value *in : phi->incoming_values())
      if (!hasDSRepresentation(in)) {
        allReachable = false;
        break;
      }
    if (allReachable) {
      IRBuilder<> phiB(phi);
      unsigned nIn = phi->getNumIncomingValues();
      auto *hiPhi = phiB.CreatePHI(f32Ty(B), nIn, "ds.phi.hi");
      auto *loPhi = phiB.CreatePHI(f32Ty(B), nIn, "ds.phi.lo");
      DSValue result{hiPhi, loPhi};
      dsMap[op] = result;
      for (unsigned i = 0; i < nIn; ++i) {
        Value *in = phi->getIncomingValue(i);
        BasicBlock *inBB = phi->getIncomingBlock(i);
        if (auto jt = dsMap.find(in); jt != dsMap.end()) {
          hiPhi->addIncoming(jt->second.hi, inBB);
          loPhi->addIncoming(jt->second.lo, inBB);
        } else if (auto *CFP = dyn_cast<ConstantFP>(in)) {
          DSValue inDS = splitConstantFP(B, CFP);
          hiPhi->addIncoming(inDS.hi, inBB);
          loPhi->addIncoming(inDS.lo, inBB);
        } else {
          // Not materialized yet (a loop back edge); defer to applyExpansion's
          // drain.
          deferredDSPhiIns().push_back(
              {hiPhi, loPhi, in, inBB, B.getFastMathFlags()});
        }
      }
      return result;
    }
  }

  return emitToDS(B, op);
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
    } else if (callee->hasFnAttribute("enzyme_math")) {
      mathName = callee->getFnAttribute("enzyme_math").getValueAsString();
    } else {
      StringRef name = callee->getName();
      if (name.starts_with("__nv_"))
        name = name.drop_front(5);
      if (!name.empty() && (name.back() == 'f' || name.back() == 'l'))
        name = name.drop_back(1);
      mathName = name;
    }

    if (mathName == "sqrt") {
      DSValue x = getOrSplitOperand(B, CI->getArgOperand(0), dsMap);
      return emitDSSqrt(B, x);
    }
    if (mathName == "fmuladd" || mathName == "fma") {
      DSValue a = getOrSplitOperand(B, CI->getArgOperand(0), dsMap);
      DSValue b = getOrSplitOperand(B, CI->getArgOperand(1), dsMap);
      DSValue c = getOrSplitOperand(B, CI->getArgOperand(2), dsMap);
      return emitDSFMA(B, a, b, c);
    }
  }

  return {nullptr, nullptr};
}

// df64 limbs produced for the rewritten instructions; getCompCost re-roots its
// cost walk on them because the original F64 roots are RAUW'd and DCE'd.
static SmallVector<Value *, 32> g_lastExpansionLimbs;
void takeLastExpansionLimbs(SmallVectorImpl<Value *> &out) {
  out.append(g_lastExpansionLimbs.begin(), g_lastExpansionLimbs.end());
  g_lastExpansionLimbs.clear();
}

void applyExpansion(ArrayRef<Instruction *> instsToChange,
                    const SmallPtrSetImpl<Instruction *> &allChanged,
                    DenseMap<Value *, Value *> *restoredValues) {
  DenseMap<Value *, DSValue> dsMap;
  deferredDSPhiIns().clear();
  g_lastExpansionLimbs.clear();

  for (Instruction *I : instsToChange) {
    IRBuilder<> B(I);
    FastMathFlags FMF = I->getFastMathFlags();
    FMF.setAllowReassoc(false);
    FMF.setNoSignedZeros(false);
    B.setFastMathFlags(FMF);

    DSValue ds = emitDSForInstruction(B, I, dsMap);
    if (!ds.hi) {
      errs() << "[poseidon] expansion: unsupported instruction, skipping: "
             << *I << "\n";
      continue;
    }

    dsMap[I] = ds;
    if (ds.hi)
      g_lastExpansionLimbs.push_back(ds.hi);
    if (ds.lo)
      g_lastExpansionLimbs.push_back(ds.lo);
  }

  // Resolve deferred back-edge phi incomings now the subgraph is in dsMap;
  // iterative, since resolving one incoming may carry a nested phi.
  while (!deferredDSPhiIns().empty()) {
    DeferredDSPhiIn d = deferredDSPhiIns().pop_back_val();
    IRBuilder<> predB(d.block->getTerminator());
    predB.setFastMathFlags(d.fmf);
    DSValue inDS = getOrSplitOperand(predB, d.incoming, dsMap);
    d.hiPhi->addIncoming(inDS.hi, d.block);
    d.loPhi->addIncoming(inDS.lo, d.block);
  }

  for (auto &[val, ds] : dsMap) {
    auto *I = dyn_cast<Instruction>(val);
    if (!I || !allChanged.count(I))
      continue;
    SmallVector<Use *, 4> externalUses;
    for (Use &U : I->uses()) {
      auto *userI = dyn_cast<Instruction>(U.getUser());
      if (!userI || !dsMap.count(userI) || !allChanged.count(userI))
        externalUses.push_back(&U);
    }

    if (!externalUses.empty()) {
      BasicBlock::iterator restorePos =
          isa<PHINode>(I) ? I->getParent()->getFirstNonPHIIt()
                          : std::next(BasicBlock::iterator(I));
      IRBuilder<> RestoreB(I->getParent(), restorePos);
      Value *restored = emitDSToFP(RestoreB, ds, I->getType());
      assert(restored->getType() == I->getType() &&
             "unexpected restored value type");
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
        errs()
            << "[poseidon] expansion: replacing with undef (remaining uses): "
            << *I << "\n";
        for (auto &U : I->uses())
          errs() << "  used by: " << *U.getUser() << "\n";
        I->replaceAllUsesWith(UndefValue::get(I->getType()));
      }
    }
    I->eraseFromParent();
  }

  // Eliminate the FP64 accumulator PHIs carried as df64 pairs: they are
  // non-optimizable and survive the erase above, and the external-use restore
  // would collapse the pair every iteration. Replace each with one restore of
  // its carried pair; O3 sinks it to the loop exit.
  SmallVector<PHINode *, 8> carriedPhis;
  for (auto &[val, ds] : dsMap)
    if (auto *phi = dyn_cast<PHINode>(val))
      if (ds.hi && phi->getType()->isDoubleTy())
        carriedPhis.push_back(phi);
  for (PHINode *phi : carriedPhis) {
    IRBuilder<> RB(phi->getParent(), phi->getParent()->getFirstNonPHIIt());
    phi->replaceAllUsesWith(emitDSToFP(RB, dsMap[phi], phi->getType()));
  }
  for (PHINode *phi : carriedPhis)
    phi->eraseFromParent();
}

// Wider FP32 expansions (n = 3, 4), a parallel arm sharing only the exact
// transformations with the two-component code above. Each component sequence
// reproduces, operation for operation, a routine of the Ozaki-Wakita QxW
// generator as instantiated by mX_real at Algorithm::Sloppy (the add/mul/div/
// sqrt _QTW_ and _QQW_ variants, each followed by Normalize<Regular>; mX_real
// and Ozaki-QW are BSD-3-Clause). Subtraction is addition of the negated
// operand and a fused multiply-add is materialized as multiply-then-add, as in
// the reference.

namespace {

struct ExpansionBuilder {
  IRBuilder<> &B;
  Function *fmaFn;
  Function *sqrtFn;
  explicit ExpansionBuilder(IRBuilder<> &Builder) : B(Builder) {
    Module *M = B.GetInsertBlock()->getParent()->getParent();
    fmaFn =
        Intrinsic::getOrInsertDeclaration(M, Intrinsic::fma, {B.getFloatTy()});
    // Correctly-rounded F32 sqrt (sqrt.rn.f32 on NVPTX), NOT the MUFU
    // approximation the lossy F32 tier uses: the expansion's Newton
    // corrections only remove the seed's error to second/third order, so a
    // 2-ulp seed would cap a triple-float at ~1e-14 instead of ~1e-21.
    sqrtFn =
        Intrinsic::getOrInsertDeclaration(M, Intrinsic::sqrt, {B.getFloatTy()});
  }
  Value *add(Value *a, Value *b) { return B.CreateFAdd(a, b, "mfx.a"); }
  Value *mul(Value *a, Value *b) { return B.CreateFMul(a, b, "mfx.m"); }
  Value *div(Value *a, Value *b) { return B.CreateFDiv(a, b, "mfx.d"); }
  Value *neg(Value *a) { return B.CreateFNeg(a, "mfx.n"); }
  Value *fma(Value *a, Value *b, Value *c) {
    return B.CreateCall(fmaFn, {a, b, c}, "mfx.f");
  }
  Value *sqrt(Value *a) { return B.CreateCall(sqrtFn, {a}, "mfx.q"); }
  Value *cst(double v) { return ConstantFP::get(B.getFloatTy(), v); }
  DSValue twoSum(Value *a, Value *b) { return emitTwoSum(B, a, b); }
  DSValue fastTwoSum(Value *a, Value *b) { return emitFastTwoSum(B, a, b); }
  DSValue twoProd(Value *a, Value *b) { return emitTwoProdFMA(B, a, b); }
};

// mX_real Normalize<NormalizeOption::Regular> for a 3-limb expansion at a
// non-Quasi algorithm: three in-place fastTwoSum sweeps.
void expansionNormalize3(ExpansionBuilder &b, Value *&x0, Value *&x1,
                         Value *&x2) {
  DSValue t = b.fastTwoSum(x1, x2);
  x1 = t.hi;
  x2 = t.lo;
  t = b.fastTwoSum(x0, x1);
  x0 = t.hi;
  x1 = t.lo;
  t = b.fastTwoSum(x1, x2);
  x1 = t.hi;
  x2 = t.lo;
}

// mX_real Normalize<Regular> for a 4-limb expansion at a non-Quasi algorithm.
void expansionNormalize4(ExpansionBuilder &b, Value *&x0, Value *&x1,
                         Value *&x2, Value *&x3) {
  DSValue t = b.fastTwoSum(x2, x3);
  x2 = t.hi;
  x3 = t.lo;
  t = b.fastTwoSum(x1, x2);
  x1 = t.hi;
  x2 = t.lo;
  t = b.fastTwoSum(x0, x1);
  x0 = t.hi;
  x1 = t.lo;
  t = b.fastTwoSum(x2, x3);
  x2 = t.hi;
  x3 = t.lo;
  t = b.fastTwoSum(x1, x2);
  x1 = t.hi;
  x2 = t.lo;
  t = b.fastTwoSum(x2, x3);
  x2 = t.hi;
  x3 = t.lo;
}

// QxW::add_QTW_QTW_QTW  (3 + 3 -> 3, unnormalized)
void expansionAddRaw3(ExpansionBuilder &b, Value *a0, Value *a1, Value *a2,
                      Value *b0, Value *b1, Value *b2, Value *&c0, Value *&c1,
                      Value *&c2) {
  DSValue s = b.twoSum(a0, b0);
  c0 = s.hi;
  c1 = s.lo;
  DSValue u = b.twoSum(a1, b1);
  Value *t0 = u.hi;
  c2 = u.lo;
  DSValue v = b.twoSum(c1, t0);
  c1 = v.hi;
  t0 = v.lo;
  Value *t1 = b.add(a2, b2);
  c2 = b.add(b.add(c2, t0), t1);
}

// QxW::mul_QTW_QTW_QTW  (3 * 3 -> 3, unnormalized)
void expansionMulRaw3(ExpansionBuilder &b, Value *a0, Value *a1, Value *a2,
                      Value *b0, Value *b1, Value *b2, Value *&c0, Value *&c1,
                      Value *&c2) {
  DSValue p = b.twoProd(a0, b0);
  c0 = p.hi;
  c1 = p.lo;
  DSValue q = b.twoProd(a0, b1);
  c2 = q.hi;
  Value *t0 = q.lo;
  DSValue r = b.twoProd(a1, b0);
  Value *t1 = r.hi;
  Value *t2 = r.lo;
  DSValue s = b.twoSum(c1, c2);
  c1 = s.hi;
  c2 = s.lo;
  DSValue u = b.twoSum(c1, t1);
  c1 = u.hi;
  t1 = u.lo;
  t0 = b.fma(a0, b2, t0);
  c2 = b.fma(a1, b1, c2);
  t2 = b.fma(a2, b0, t2);
  c2 = b.add(b.add(b.add(t0, c2), t2), t1);
}

// QxW::mul_PA_QTW_QTW  (2 * 3 -> 3, unnormalized)
void expansionMulRaw23(ExpansionBuilder &b, Value *a0, Value *a1, Value *b0,
                       Value *b1, Value *b2, Value *&c0, Value *&c1,
                       Value *&c2) {
  DSValue p = b.twoProd(a0, b0);
  c0 = p.hi;
  c1 = p.lo;
  DSValue q = b.twoProd(a0, b1);
  c2 = q.hi;
  Value *t0 = q.lo;
  DSValue r = b.twoProd(a1, b0);
  Value *t1 = r.hi;
  Value *t2 = r.lo;
  DSValue s = b.twoSum(c1, c2);
  c1 = s.hi;
  c2 = s.lo;
  DSValue u = b.twoSum(c1, t1);
  c1 = u.hi;
  t1 = u.lo;
  t0 = b.fma(a0, b2, t0);
  c2 = b.fma(a1, b1, c2);
  c2 = b.add(b.add(b.add(t0, c2), t2), t1);
}

// QxW::div_PA_PA_PA  (2 / 2 -> 2, unnormalized)
void expansionDivRaw22(ExpansionBuilder &b, Value *a0, Value *a1, Value *b0,
                       Value *b1, Value *&c0, Value *&c1) {
  Value *bh = b.add(b0, b1);
  c0 = b.div(a0, bh);
  c1 = b.add(b.fma(b.neg(b0), c0, a0), a1);
  c1 = b.div(b.fma(b.neg(b1), c0, c1), bh);
}

// QxW::div_QTW_QTW_QTW  (3 / 3 -> 3, unnormalized)
void expansionDivRaw3(ExpansionBuilder &b, Value *a0, Value *a1, Value *a2,
                      Value *b0, Value *b1, Value *b2, Value *&c0, Value *&c1,
                      Value *&c2) {
  Value *e40 = b.add(a1, a2);
  Value *e41 = b.add(b1, b2);
  expansionDivRaw22(b, a0, e40, b0, e41, c0, c1);
  Value *t0, *t1, *t2;
  expansionMulRaw23(b, c0, c1, b0, b1, b2, t0, t1, t2);
  Value *s0, *s1, *s2;
  expansionAddRaw3(b, a0, a1, a2, b.neg(t0), b.neg(t1), b.neg(t2), s0, s1, s2);
  Value *tn = b.add(b.add(s0, s1), s2);
  Value *td = b.add(b.add(b0, b1), b2);
  c2 = b.div(tn, td);
}

// QxW::sqrt_QTW_PA  (sqrt of a 3-limb value to 2 limbs, unnormalized)
void expansionSqrtRaw32(ExpansionBuilder &b, Value *a0, Value *a1, Value *a2,
                        Value *&c0, Value *&c1) {
  c0 = b.sqrt(a0);
  Value *num = b.add(b.add(b.fma(b.neg(c0), c0, a0), a1), a2);
  c1 = b.div(num, b.add(c0, c0));
}

// QxW::sqr_PA_QTW  (square of a 2-limb value to 3 limbs, unnormalized)
void expansionSqrRaw23(ExpansionBuilder &b, Value *a0, Value *a1, Value *&c0,
                       Value *&c1, Value *&c2) {
  DSValue p = b.twoProd(a0, a0);
  c0 = p.hi;
  c1 = p.lo;
  Value *t0 = b.add(a0, a0);
  DSValue q = b.twoProd(t0, a1);
  c2 = q.hi;
  Value *t1 = q.lo;
  DSValue s = b.twoSum(c1, c2);
  c1 = s.hi;
  c2 = s.lo;
  c2 = b.add(c2, t1);
  c2 = b.fma(a1, a1, c2);
}

// QxW::sqrt_QTW_QTW  (3 -> 3, unnormalized)
void expansionSqrtRaw3(ExpansionBuilder &b, Value *a0, Value *a1, Value *a2,
                       Value *&c0, Value *&c1, Value *&c2) {
  expansionSqrtRaw32(b, a0, a1, a2, c0, c1);
  Value *t0, *t1, *t2;
  expansionSqrRaw23(b, c0, c1, t0, t1, t2);
  Value *s0, *s1, *s2;
  expansionAddRaw3(b, a0, a1, a2, b.neg(t0), b.neg(t1), b.neg(t2), s0, s1, s2);
  Value *tn = b.add(b.add(s0, s1), s2);
  Value *td = b.mul(b.add(c0, c1), b.cst(2.0));
  c2 = b.div(tn, td);
}

// QxW::add_QQW_QQW_QQW  (4 + 4 -> 4, unnormalized)
void expansionAddRaw4(ExpansionBuilder &b, Value *a0, Value *a1, Value *a2,
                      Value *a3, Value *b0, Value *b1, Value *b2, Value *b3,
                      Value *&c0, Value *&c1, Value *&c2, Value *&c3) {
  DSValue s = b.twoSum(a0, b0);
  c0 = s.hi;
  c1 = s.lo;
  DSValue u = b.twoSum(a1, b1);
  Value *t0 = u.hi;
  c2 = u.lo;
  DSValue v = b.twoSum(a2, b2);
  Value *t1 = v.hi;
  c3 = v.lo;
  DSValue w = b.twoSum(c1, t0);
  c1 = w.hi;
  t0 = w.lo;
  DSValue x = b.twoSum(c2, t0);
  c2 = x.hi;
  t0 = x.lo;
  DSValue y = b.twoSum(c2, t1);
  c2 = y.hi;
  t1 = y.lo;
  Value *t2 = b.add(a3, b3);
  c3 = b.add(b.add(b.add(c3, t0), t1), t2);
}

// QxW::mul_QQW_QQW_QQW  (4 * 4 -> 4, unnormalized)
void expansionMulRaw4(ExpansionBuilder &b, Value *a0, Value *a1, Value *a2,
                      Value *a3, Value *b0, Value *b1, Value *b2, Value *b3,
                      Value *&c0, Value *&c1, Value *&c2, Value *&c3) {
  DSValue p = b.twoProd(a0, b0);
  c0 = p.hi;
  c1 = p.lo;
  DSValue q = b.twoProd(a0, b1);
  c2 = q.hi;
  c3 = q.lo;
  DSValue r = b.twoProd(a1, b0);
  Value *t0 = r.hi;
  Value *t1 = r.lo;
  DSValue s = b.twoSum(c1, c2);
  c1 = s.hi;
  c2 = s.lo;
  DSValue u = b.twoSum(c1, t0);
  c1 = u.hi;
  t0 = u.lo;
  DSValue v = b.twoProd(a0, b2);
  Value *t2 = v.hi;
  Value *t3 = v.lo;
  DSValue w = b.twoProd(a1, b1);
  Value *t4 = w.hi;
  Value *t5 = w.lo;
  DSValue x = b.twoProd(a2, b0);
  Value *t6 = x.hi;
  Value *t7 = x.lo;
  DSValue y = b.twoSum(c2, t0);
  c2 = y.hi;
  t0 = y.lo;
  DSValue z = b.twoSum(c2, c3);
  c2 = z.hi;
  c3 = z.lo;
  DSValue A = b.twoSum(c2, t1);
  c2 = A.hi;
  t1 = A.lo;
  DSValue C = b.twoSum(c2, t2);
  c2 = C.hi;
  t2 = C.lo;
  DSValue D = b.twoSum(c2, t4);
  c2 = D.hi;
  t4 = D.lo;
  DSValue E = b.twoSum(c2, t6);
  c2 = E.hi;
  t6 = E.lo;
  c3 = b.fma(a0, b3, c3);
  c3 = b.fma(a1, b2, c3);
  c3 = b.fma(a2, b1, c3);
  c3 = b.fma(a3, b0, c3);
  c3 = b.add(c3, t0);
  c3 = b.add(c3, t1);
  c3 = b.add(c3, t2);
  c3 = b.add(c3, t3);
  c3 = b.add(c3, t4);
  c3 = b.add(c3, t5);
  c3 = b.add(c3, t6);
  c3 = b.add(c3, t7);
}

// QxW::mul_QTW_QQW_QQW  (3 * 4 -> 4, unnormalized)
void expansionMulRaw34(ExpansionBuilder &b, Value *a0, Value *a1, Value *a2,
                       Value *b0, Value *b1, Value *b2, Value *b3, Value *&c0,
                       Value *&c1, Value *&c2, Value *&c3) {
  DSValue p = b.twoProd(a0, b0);
  c0 = p.hi;
  c1 = p.lo;
  DSValue q = b.twoProd(a0, b1);
  c2 = q.hi;
  c3 = q.lo;
  DSValue r = b.twoProd(a1, b0);
  Value *t0 = r.hi;
  Value *t1 = r.lo;
  DSValue s = b.twoSum(c1, c2);
  c1 = s.hi;
  c2 = s.lo;
  DSValue u = b.twoSum(c1, t0);
  c1 = u.hi;
  t0 = u.lo;
  DSValue v = b.twoProd(a0, b2);
  Value *t2 = v.hi;
  Value *t3 = v.lo;
  DSValue w = b.twoProd(a1, b1);
  Value *t4 = w.hi;
  Value *t5 = w.lo;
  DSValue x = b.twoProd(a2, b0);
  Value *t6 = x.hi;
  Value *t7 = x.lo;
  DSValue y = b.twoSum(c2, t0);
  c2 = y.hi;
  t0 = y.lo;
  DSValue z = b.twoSum(c2, c3);
  c2 = z.hi;
  c3 = z.lo;
  DSValue A = b.twoSum(c2, t1);
  c2 = A.hi;
  t1 = A.lo;
  DSValue C = b.twoSum(c2, t2);
  c2 = C.hi;
  t2 = C.lo;
  DSValue D = b.twoSum(c2, t4);
  c2 = D.hi;
  t4 = D.lo;
  DSValue E = b.twoSum(c2, t6);
  c2 = E.hi;
  t6 = E.lo;
  c3 = b.fma(a0, b3, c3);
  c3 = b.fma(a1, b2, c3);
  c3 = b.fma(a2, b1, c3);
  c3 = b.add(c3, t0);
  c3 = b.add(c3, t1);
  c3 = b.add(c3, t2);
  c3 = b.add(c3, t3);
  c3 = b.add(c3, t4);
  c3 = b.add(c3, t5);
  c3 = b.add(c3, t6);
  c3 = b.add(c3, t7);
}

// QxW::sqr_QTW_QQW  (square of a 3-limb value to 4 limbs, unnormalized)
void expansionSqrRaw34(ExpansionBuilder &b, Value *a0, Value *a1, Value *a2,
                       Value *&c0, Value *&c1, Value *&c2, Value *&c3) {
  DSValue p = b.twoProd(a0, a0);
  c0 = p.hi;
  c1 = p.lo;
  Value *t0 = b.add(a0, a0);
  DSValue q = b.twoProd(t0, a1);
  c2 = q.hi;
  c3 = q.lo;
  DSValue s = b.twoSum(c1, c2);
  c1 = s.hi;
  c2 = s.lo;
  DSValue u = b.twoSum(c2, c3);
  c2 = u.hi;
  c3 = u.lo;
  DSValue v = b.twoProd(t0, a2);
  Value *t1 = v.hi;
  Value *t2 = v.lo;
  DSValue w = b.twoProd(a1, a1);
  Value *t3 = w.hi;
  Value *t4 = w.lo;
  DSValue x = b.twoSum(c2, t1);
  c2 = x.hi;
  t1 = x.lo;
  DSValue y = b.twoSum(c2, t3);
  c2 = y.hi;
  t3 = y.lo;
  Value *t5 = b.add(a1, a1);
  c3 = b.add(b.add(b.add(b.add(c3, t1), t2), t3), t4);
  c3 = b.fma(t5, a2, c3);
}

// QxW::div_QQW_QQW_QQW  (4 / 4 -> 4, unnormalized)
void expansionDivRaw4(ExpansionBuilder &b, Value *a0, Value *a1, Value *a2,
                      Value *a3, Value *b0, Value *b1, Value *b2, Value *b3,
                      Value *&c0, Value *&c1, Value *&c2, Value *&c3) {
  Value *e40 = b.add(a2, a3);
  Value *e41 = b.add(b2, b3);
  expansionDivRaw3(b, a0, a1, e40, b0, b1, e41, c0, c1, c2);
  Value *t0, *t1, *t2, *t3;
  expansionMulRaw34(b, c0, c1, c2, b0, b1, b2, b3, t0, t1, t2, t3);
  Value *s0, *s1, *s2, *s3;
  expansionAddRaw4(b, a0, a1, a2, a3, b.neg(t0), b.neg(t1), b.neg(t2),
                   b.neg(t3), s0, s1, s2, s3);
  Value *tn = b.add(b.add(b.add(s0, s1), s2), s3);
  Value *td = b.add(b.add(b.add(b0, b1), b2), b3);
  c3 = b.div(tn, td);
}

// QxW::sqrt_QQW_QQW  (4 -> 4, unnormalized). QxW::sqrt_QQW_QTW is
// sqrt_QTW_QTW(a0, a1, a2 + a3, ...).
void expansionSqrtRaw4(ExpansionBuilder &b, Value *a0, Value *a1, Value *a2,
                       Value *a3, Value *&c0, Value *&c1, Value *&c2,
                       Value *&c3) {
  expansionSqrtRaw3(b, a0, a1, b.add(a2, a3), c0, c1, c2);
  Value *t0, *t1, *t2, *t3;
  expansionSqrRaw34(b, c0, c1, c2, t0, t1, t2, t3);
  Value *s0, *s1, *s2, *s3;
  expansionAddRaw4(b, a0, a1, a2, a3, b.neg(t0), b.neg(t1), b.neg(t2),
                   b.neg(t3), s0, s1, s2, s3);
  Value *tn = b.add(b.add(b.add(s0, s1), s2), s3);
  Value *td = b.mul(b.add(b.add(c0, c1), c2), b.cst(2.0));
  c3 = b.div(tn, td);
}

// Only the component counts whose QxW routines are transcribed above exist.
void expansionCheckN(unsigned n) {
  if (n != 3 && n != 4)
    report_fatal_error(
        "Poseidon expansion emitter: only 3- and 4-component FP32 expansions "
        "are implemented (2 components has its own DSValue path). Refusing to "
        "fabricate an unimplemented component count.");
}

} // namespace

ExpansionValue emitExpansionNeg(IRBuilder<> &B, const ExpansionValue &a) {
  ExpansionValue r;
  for (Value *v : a.x)
    r.x.push_back(B.CreateFNeg(v, "mfx.neg"));
  return r;
}

ExpansionValue emitExpansionAdd(IRBuilder<> &B, const ExpansionValue &a,
                                const ExpansionValue &b) {
  assert(a.n() == b.n() && "expansion operands must agree on component count");
  expansionCheckN(a.n());
  ExpansionBuilder mb(B);
  if (a.n() == 3) {
    Value *c0, *c1, *c2;
    expansionAddRaw3(mb, a.x[0], a.x[1], a.x[2], b.x[0], b.x[1], b.x[2], c0, c1,
                     c2);
    expansionNormalize3(mb, c0, c1, c2);
    return ExpansionValue{{c0, c1, c2}};
  }
  Value *c0, *c1, *c2, *c3;
  expansionAddRaw4(mb, a.x[0], a.x[1], a.x[2], a.x[3], b.x[0], b.x[1], b.x[2],
                   b.x[3], c0, c1, c2, c3);
  expansionNormalize4(mb, c0, c1, c2, c3);
  return ExpansionValue{{c0, c1, c2, c3}};
}

ExpansionValue emitExpansionSub(IRBuilder<> &B, const ExpansionValue &a,
                                const ExpansionValue &b) {
  return emitExpansionAdd(B, a, emitExpansionNeg(B, b));
}

ExpansionValue emitExpansionMul(IRBuilder<> &B, const ExpansionValue &a,
                                const ExpansionValue &b) {
  assert(a.n() == b.n() && "expansion operands must agree on component count");
  expansionCheckN(a.n());
  ExpansionBuilder mb(B);
  if (a.n() == 3) {
    Value *c0, *c1, *c2;
    expansionMulRaw3(mb, a.x[0], a.x[1], a.x[2], b.x[0], b.x[1], b.x[2], c0, c1,
                     c2);
    expansionNormalize3(mb, c0, c1, c2);
    return ExpansionValue{{c0, c1, c2}};
  }
  Value *c0, *c1, *c2, *c3;
  expansionMulRaw4(mb, a.x[0], a.x[1], a.x[2], a.x[3], b.x[0], b.x[1], b.x[2],
                   b.x[3], c0, c1, c2, c3);
  expansionNormalize4(mb, c0, c1, c2, c3);
  return ExpansionValue{{c0, c1, c2, c3}};
}

ExpansionValue emitExpansionDiv(IRBuilder<> &B, const ExpansionValue &a,
                                const ExpansionValue &b) {
  assert(a.n() == b.n() && "expansion operands must agree on component count");
  expansionCheckN(a.n());
  ExpansionBuilder mb(B);
  if (a.n() == 3) {
    Value *c0, *c1, *c2;
    expansionDivRaw3(mb, a.x[0], a.x[1], a.x[2], b.x[0], b.x[1], b.x[2], c0, c1,
                     c2);
    expansionNormalize3(mb, c0, c1, c2);
    return ExpansionValue{{c0, c1, c2}};
  }
  Value *c0, *c1, *c2, *c3;
  expansionDivRaw4(mb, a.x[0], a.x[1], a.x[2], a.x[3], b.x[0], b.x[1], b.x[2],
                   b.x[3], c0, c1, c2, c3);
  expansionNormalize4(mb, c0, c1, c2, c3);
  return ExpansionValue{{c0, c1, c2, c3}};
}

ExpansionValue emitExpansionSqrt(IRBuilder<> &B, const ExpansionValue &a) {
  expansionCheckN(a.n());
  ExpansionBuilder mb(B);
  if (a.n() == 3) {
    Value *c0, *c1, *c2;
    expansionSqrtRaw3(mb, a.x[0], a.x[1], a.x[2], c0, c1, c2);
    expansionNormalize3(mb, c0, c1, c2);
    return ExpansionValue{{c0, c1, c2}};
  }
  Value *c0, *c1, *c2, *c3;
  expansionSqrtRaw4(mb, a.x[0], a.x[1], a.x[2], a.x[3], c0, c1, c2, c3);
  expansionNormalize4(mb, c0, c1, c2, c3);
  return ExpansionValue{{c0, c1, c2, c3}};
}

// Greedy split, identical to the reference's `fromD`; an F64 source fills at
// most three limbs, the extra precision is created by the arithmetic.
ExpansionValue emitToExpansion(IRBuilder<> &B, Value *fpval, unsigned n) {
  expansionCheckN(n);
  Type *F32 = B.getFloatTy();
  Type *F64 = B.getDoubleTy();
  ExpansionValue r;
  if (fpval->getType()->isFloatTy()) {
    r.x.push_back(fpval);
    for (unsigned i = 1; i < n; ++i)
      r.x.push_back(ConstantFP::get(F32, 0.0));
    return r;
  }
  assert(fpval->getType()->isDoubleTy() &&
         "emitToExpansion: only f32/f64 sources");
  Value *rem = fpval;
  for (unsigned i = 0; i < n; ++i) {
    Value *li = B.CreateFPTrunc(rem, F32, "mfx.split");
    r.x.push_back(li);
    if (i + 1 < n) {
      Value *back = B.CreateFPExt(li, F64, "mfx.splitb");
      rem = B.CreateFSub(rem, back, "mfx.splitr");
    }
  }
  return r;
}

// Sum the limbs least significant first, as the reference's `to_d` does.
Value *emitExpansionToFP(IRBuilder<> &B, const ExpansionValue &v,
                         Type *targetTy) {
  expansionCheckN(v.n());
  if (targetTy->isFloatTy()) {
    Value *s = v.x[v.n() - 1];
    for (int i = (int)v.n() - 2; i >= 0; --i)
      s = B.CreateFAdd(v.x[i], s, "mfx.tof32");
    return s;
  }
  assert(targetTy->isDoubleTy() && "emitExpansionToFP: only f32/f64 targets");
  Type *F64 = B.getDoubleTy();
  Value *s = B.CreateFPExt(v.x[v.n() - 1], F64, "mfx.e");
  for (int i = (int)v.n() - 2; i >= 0; --i)
    s = B.CreateFAdd(B.CreateFPExt(v.x[i], F64, "mfx.e"), s, "mfx.tof64");
  return s;
}

// Materializer for n >= 3; mirrors applyExpansion without the two-component
// df64 staging machinery.

namespace {
struct DeferredExpansionPhiIn {
  SmallVector<PHINode *, 4> phis;
  Value *incoming;
  BasicBlock *block;
  FastMathFlags fmf;
};
} // namespace

static SmallVector<DeferredExpansionPhiIn, 16> &deferredExpansionPhiIns() {
  static SmallVector<DeferredExpansionPhiIn, 16> v;
  return v;
}

static ExpansionValue splitConstantFPExpansion(IRBuilder<> &B, ConstantFP *CFP,
                                               unsigned n) {
  double val = CFP->getValueAPF().convertToDouble();
  ExpansionValue r;
  double rem = val;
  for (unsigned i = 0; i < n; ++i) {
    float li = (float)rem;
    r.x.push_back(ConstantFP::get(B.getFloatTy(), li));
    rem -= (double)li;
  }
  return r;
}

static ExpansionValue
getOrSplitOperandExpansion(IRBuilder<> &B, Value *op, unsigned n,
                           DenseMap<Value *, ExpansionValue> &expansionMap) {
  if (auto it = expansionMap.find(op); it != expansionMap.end())
    return it->second;

  if (auto *CFP = dyn_cast<ConstantFP>(op))
    return splitConstantFPExpansion(B, CFP, n);

  if (auto *phi = dyn_cast<PHINode>(op);
      phi && (phi->getType()->isFloatTy() || phi->getType()->isDoubleTy())) {
    bool allReachable = true;
    for (Value *in : phi->incoming_values()) {
      Type *T = in->getType();
      if (!T->isFloatTy() && !T->isDoubleTy()) {
        allReachable = false;
        break;
      }
    }
    if (allReachable) {
      IRBuilder<> phiB(phi);
      unsigned nIn = phi->getNumIncomingValues();
      ExpansionValue result;
      SmallVector<PHINode *, 4> phis;
      for (unsigned c = 0; c < n; ++c) {
        auto *p = phiB.CreatePHI(B.getFloatTy(), nIn, "mfx.phi");
        phis.push_back(p);
        result.x.push_back(p);
      }
      expansionMap[op] = result;
      for (unsigned i = 0; i < nIn; ++i) {
        Value *in = phi->getIncomingValue(i);
        BasicBlock *inBB = phi->getIncomingBlock(i);
        if (auto jt = expansionMap.find(in); jt != expansionMap.end()) {
          for (unsigned c = 0; c < n; ++c)
            phis[c]->addIncoming(jt->second.x[c], inBB);
        } else if (auto *CFP = dyn_cast<ConstantFP>(in)) {
          ExpansionValue inExpansion = splitConstantFPExpansion(B, CFP, n);
          for (unsigned c = 0; c < n; ++c)
            phis[c]->addIncoming(inExpansion.x[c], inBB);
        } else {
          deferredExpansionPhiIns().push_back(
              {phis, in, inBB, B.getFastMathFlags()});
        }
      }
      return result;
    }
  }

  return emitToExpansion(B, op, n);
}

static ExpansionValue
emitExpansionForInstruction(IRBuilder<> &B, Instruction *I, unsigned n,
                            DenseMap<Value *, ExpansionValue> &expansionMap) {
  unsigned opcode = I->getOpcode();

  if (auto *BO = dyn_cast<BinaryOperator>(I)) {
    ExpansionValue lhs =
        getOrSplitOperandExpansion(B, BO->getOperand(0), n, expansionMap);
    ExpansionValue rhs =
        getOrSplitOperandExpansion(B, BO->getOperand(1), n, expansionMap);
    switch (opcode) {
    case Instruction::FAdd:
      return emitExpansionAdd(B, lhs, rhs);
    case Instruction::FSub:
      return emitExpansionSub(B, lhs, rhs);
    case Instruction::FMul:
      return emitExpansionMul(B, lhs, rhs);
    case Instruction::FDiv:
      return emitExpansionDiv(B, lhs, rhs);
    default:
      break;
    }
  }

  if (auto *UO = dyn_cast<UnaryOperator>(I))
    if (opcode == Instruction::FNeg)
      return emitExpansionNeg(
          B, getOrSplitOperandExpansion(B, UO->getOperand(0), n, expansionMap));

  if (auto *CI = dyn_cast<CallInst>(I)) {
    Function *callee = CI->getCalledFunction();
    if (!callee)
      return ExpansionValue{};

    StringRef mathName;
    if (callee->isIntrinsic()) {
      Intrinsic::ID id = callee->getIntrinsicID();
      if (id == Intrinsic::sqrt)
        mathName = "sqrt";
      else if (id == Intrinsic::fmuladd)
        mathName = "fmuladd";
      else if (id == Intrinsic::fma)
        mathName = "fma";
    } else if (callee->hasFnAttribute("enzyme_math")) {
      mathName = callee->getFnAttribute("enzyme_math").getValueAsString();
    } else {
      StringRef name = callee->getName();
      if (name.starts_with("__nv_"))
        name = name.drop_front(5);
      if (!name.empty() && (name.back() == 'f' || name.back() == 'l'))
        name = name.drop_back(1);
      mathName = name;
    }

    if (mathName == "sqrt")
      return emitExpansionSqrt(
          B,
          getOrSplitOperandExpansion(B, CI->getArgOperand(0), n, expansionMap));
    if (mathName == "fmuladd" || mathName == "fma") {
      ExpansionValue a =
          getOrSplitOperandExpansion(B, CI->getArgOperand(0), n, expansionMap);
      ExpansionValue b =
          getOrSplitOperandExpansion(B, CI->getArgOperand(1), n, expansionMap);
      ExpansionValue c =
          getOrSplitOperandExpansion(B, CI->getArgOperand(2), n, expansionMap);
      return emitExpansionAdd(B, emitExpansionMul(B, a, b), c);
    }
  }

  return ExpansionValue{};
}

// Separate from g_lastExpansionLimbs: one candidate can apply both an expansion
// change and a double-single change, and applyExpansion clears its list on
// entry.
static SmallVector<Value *, 32> g_lastExpLimbs;
void takeLastExpLimbs(SmallVectorImpl<Value *> &out) {
  out.append(g_lastExpLimbs.begin(), g_lastExpLimbs.end());
  g_lastExpLimbs.clear();
}
// Cleared once per candidate, not per change: a tier pair calls applyExpansion
// twice.
void resetExpLimbs() { g_lastExpLimbs.clear(); }

void applyExpansion(unsigned n, ArrayRef<Instruction *> instsToChange,
                    const SmallPtrSetImpl<Instruction *> &allChanged,
                    DenseMap<Value *, Value *> *restoredValues) {
  expansionCheckN(n);
  DenseMap<Value *, ExpansionValue> expansionMap;
  deferredExpansionPhiIns().clear();

  for (Instruction *I : instsToChange) {
    IRBuilder<> B(I);
    FastMathFlags FMF = I->getFastMathFlags();
    FMF.setAllowReassoc(false);
    FMF.setNoSignedZeros(false);
    B.setFastMathFlags(FMF);

    ExpansionValue mf = emitExpansionForInstruction(B, I, n, expansionMap);
    if (mf.x.empty()) {
      errs() << "Expansion(" << n
             << "): unsupported instruction, skipping: " << *I << "\n";
      continue;
    }
    expansionMap[I] = mf;
    for (Value *v : mf.x)
      g_lastExpLimbs.push_back(v);
  }

  while (!deferredExpansionPhiIns().empty()) {
    DeferredExpansionPhiIn d = deferredExpansionPhiIns().pop_back_val();
    IRBuilder<> predB(d.block->getTerminator());
    predB.setFastMathFlags(d.fmf);
    ExpansionValue inExpansion =
        getOrSplitOperandExpansion(predB, d.incoming, n, expansionMap);
    for (unsigned c = 0; c < n; ++c)
      d.phis[c]->addIncoming(inExpansion.x[c], d.block);
  }

  for (auto &[val, mf] : expansionMap) {
    auto *I = dyn_cast<Instruction>(val);
    if (!I || !allChanged.count(I))
      continue;
    SmallVector<Use *, 4> externalUses;
    for (Use &U : I->uses()) {
      auto *userI = dyn_cast<Instruction>(U.getUser());
      if (!userI || !expansionMap.count(userI) || !allChanged.count(userI))
        externalUses.push_back(&U);
    }
    if (!externalUses.empty()) {
      BasicBlock::iterator restorePos =
          isa<PHINode>(I) ? I->getParent()->getFirstNonPHIIt()
                          : std::next(BasicBlock::iterator(I));
      IRBuilder<> RestoreB(I->getParent(), restorePos);
      Value *restored = emitExpansionToFP(RestoreB, mf, I->getType());
      assert(restored->getType() == I->getType() &&
             "unexpected restored value type");
      for (Use *U : externalUses)
        U->set(restored);
      if (restoredValues)
        (*restoredValues)[I] = restored;
    }
  }

  for (auto it = instsToChange.rbegin(); it != instsToChange.rend(); ++it) {
    Instruction *I = *it;
    if (!expansionMap.count(I))
      continue;
    if (!I->use_empty()) {
      if (restoredValues && restoredValues->count(I))
        I->replaceAllUsesWith((*restoredValues)[I]);
      else {
        errs() << "Expansion(" << n
               << "): replacing with undef (remaining uses): " << *I << "\n";
        I->replaceAllUsesWith(UndefValue::get(I->getType()));
      }
    }
    I->eraseFromParent();
  }

  // Same reason as the df64 path: the FP64 accumulator PHIs carried as n
  // parallel F32 PHIs would otherwise force a collapse-to-double every
  // iteration just to feed them.
  SmallVector<PHINode *, 8> carriedPhis;
  for (auto &[val, mf] : expansionMap)
    if (auto *phi = dyn_cast<PHINode>(val))
      if (!mf.x.empty() && phi->getType()->isDoubleTy())
        carriedPhis.push_back(phi);
  for (PHINode *phi : carriedPhis) {
    IRBuilder<> RB(phi->getParent(), phi->getParent()->getFirstNonPHIIt());
    phi->replaceAllUsesWith(
        emitExpansionToFP(RB, expansionMap[phi], phi->getType()));
  }
  for (PHINode *phi : carriedPhis)
    phi->eraseFromParent();
}

} // namespace poseidon
