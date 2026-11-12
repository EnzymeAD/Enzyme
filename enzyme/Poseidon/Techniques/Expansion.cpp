//=- Expansion.cpp - Double-single arithmetic for Poseidon ----------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Expansion.h"
#include "Core/ExpansionRoutines.h"
#include "Pass/Session.h"
#include "Staging.h"
#include "Targets/DeviceMath.h"
#include "Targets/Target.h"
#include "Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace poseidon {

static Type *f32Ty(IRBuilder<> &B) { return B.getFloatTy(); }

namespace {

// The arithmetic the routines of Core/ExpansionRoutines.h are emitted with.
struct IRArith {
  using F32 = Value *;
  using F64 = Value *;
  using Bool = Value *;
  using Pair = DSValue;

  IRBuilder<> &B;

  Value *add(Value *a, Value *b, const char *name) {
    return B.CreateFAdd(a, b, name);
  }
  Value *sub(Value *a, Value *b, const char *name) {
    return B.CreateFSub(a, b, name);
  }
  Value *mul(Value *a, Value *b, const char *name) {
    return B.CreateFMul(a, b, name);
  }
  Value *div(Value *a, Value *b, const char *name) {
    return B.CreateFDiv(a, b, name);
  }
  Value *neg(Value *a, const char *name) { return B.CreateFNeg(a, name); }
  Value *fma(Value *a, Value *b, Value *c, const char *name) {
    return B.CreateIntrinsic(Intrinsic::fma, {B.getFloatTy()}, {a, b, c}, {},
                             name);
  }
  Value *sqrt(Value *a, const char *name) {
    return B.CreateIntrinsic(Intrinsic::sqrt, {B.getFloatTy()}, {a}, {}, name);
  }
  Value *fmaCall(Value *a, Value *b, Value *c, const char *name) {
    return B.CreateCall(declaration(Intrinsic::fma), {a, b, c}, name);
  }
  Value *sqrtCall(Value *a, const char *name) {
    return B.CreateCall(declaration(Intrinsic::sqrt), {a}, name);
  }
  Value *constant(double v) { return ConstantFP::get(B.getFloatTy(), v); }
  Value *equal(Value *a, Value *b, const char *name) {
    return B.CreateFCmpOEQ(a, b, name);
  }
  Value *both(Value *a, Value *b, const char *name) {
    return B.CreateAnd(a, b, name);
  }
  Value *select(Value *c, Value *t, Value *f, const char *name) {
    return B.CreateSelect(c, t, f, name);
  }
  Value *truncate(Value *v, const char *name) {
    return B.CreateFPTrunc(v, B.getFloatTy(), name);
  }
  Value *extend(Value *v, const char *name) {
    return B.CreateFPExt(v, B.getDoubleTy(), name);
  }

  Function *declaration(Intrinsic::ID id) {
    Module *M = B.GetInsertBlock()->getParent()->getParent();
    return Intrinsic::getOrInsertDeclaration(M, id, {B.getFloatTy()});
  }
};

using Routines = ExpansionRoutines<IRArith>;
Routines routines(IRBuilder<> &B) { return Routines(IRArith{B}); }

} // namespace

DSValue emitDSAdd(IRBuilder<> &B, DSValue x, DSValue y) {
  return routines(B).dsAdd(x, y);
}

DSValue emitDSSub(IRBuilder<> &B, DSValue x, DSValue y) {
  return routines(B).dsSub(x, y);
}

DSValue emitDSMul(IRBuilder<> &B, DSValue x, DSValue y) {
  return routines(B).dsMul(x, y);
}

DSValue emitDSFMA(IRBuilder<> &B, DSValue x, DSValue y, DSValue z) {
  return routines(B).dsFma(x, y, z);
}

DSValue emitDSDiv(IRBuilder<> &B, DSValue x, DSValue y) {
  return routines(B).dsDiv(x, y);
}

DSValue emitDSSqrt(IRBuilder<> &B, DSValue x) { return routines(B).dsSqrt(x); }

DSValue emitDSNeg(IRBuilder<> &B, DSValue x) { return routines(B).dsNeg(x); }

DSValue emitF64ToDS(IRBuilder<> &B, Value *f64val) {
  assert(f64val->getType()->isDoubleTy() && "emitF64ToDS: input must be f64");
  return routines(B).dsFromF64(f64val);
}

Value *emitDSToF64(IRBuilder<> &B, DSValue ds) {
  Value *joined = routines(B).dsToF64(ds);
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
bool foldDSPairRoundtrip(Function &F, SmallVectorImpl<Value *> *foldedLimbs,
                         bool print) {
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
  if (changed && print)
    llvm::errs()
        << "[poseidon] folded df64 join/split roundtrips at the staging "
           "boundary in "
        << F.getName() << "\n";
  return changed;
}

// F32 source: exact in the hi slot, lo = 0.
static DSValue emitF32ToDS(IRBuilder<> &B, Value *f32val) {
  assert(f32val->getType()->isFloatTy() && "emitF32ToDS: input must be f32");
  return routines(B).dsFromF32(f32val);
}

// F32 target: sum the halves; as accurate as F32 allows.
static Value *emitDSToF32(IRBuilder<> &B, DSValue ds) {
  return routines(B).dsToF32(ds);
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

// Loop-carried phi incomings are deferred: the accumulator phi and the
// accumulation feeding it are mutually dependent, and resolving the back edge
// eagerly would collapse the phi to a single double.
namespace {
struct DeferredDSPhiIn {
  PHINode *hiPhi, *loPhi;
  Value *incoming;
  BasicBlock *block;
};
} // namespace
using DeferredDSPhiIns = SmallVector<DeferredDSPhiIn, 16>;
// In insertion order: the restores of the replaced values are emitted by
// iterating it, and several can share one insertion point.
using DSMap = MapVector<Value *, DSValue>;

static DSValue getOrSplitOperand(IRBuilder<> &B, Value *op, DSMap &dsMap,
                                 DeferredDSPhiIns &deferred);

// Any scalar f32/f64 value has a DS representation (materialized op, split
// constant, carried phi, or boundary emitToDS).
static bool hasDSRepresentation(Value *op) {
  Type *T = op->getType();
  return T->isFloatTy() || T->isDoubleTy();
}

static DSValue getOrSplitOperand(IRBuilder<> &B, Value *op, DSMap &dsMap,
                                 DeferredDSPhiIns &deferred) {
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
          deferred.push_back({hiPhi, loPhi, in, inBB});
        }
      }
      return result;
    }
  }

  return emitToDS(B, op);
}

static DSValue emitDSForInstruction(IRBuilder<> &B, Instruction *I,
                                    DSMap &dsMap, DeferredDSPhiIns &deferred) {
  unsigned opcode = I->getOpcode();

  if (auto *BO = dyn_cast<BinaryOperator>(I)) {
    DSValue lhs = getOrSplitOperand(B, BO->getOperand(0), dsMap, deferred);
    DSValue rhs = getOrSplitOperand(B, BO->getOperand(1), dsMap, deferred);

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
      DSValue x = getOrSplitOperand(B, UO->getOperand(0), dsMap, deferred);
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
    } else if (StringRef device = deviceMathName(callee->getName());
               !device.empty()) {
      mathName = device;
    } else {
      StringRef name = callee->getName();
      if (!name.empty() && (name.back() == 'f' || name.back() == 'l'))
        name = name.drop_back(1);
      mathName = name;
    }

    if (mathName == "sqrt") {
      DSValue x = getOrSplitOperand(B, CI->getArgOperand(0), dsMap, deferred);
      return emitDSSqrt(B, x);
    }
    if (mathName == "fmuladd" || mathName == "fma") {
      DSValue a = getOrSplitOperand(B, CI->getArgOperand(0), dsMap, deferred);
      DSValue b = getOrSplitOperand(B, CI->getArgOperand(1), dsMap, deferred);
      DSValue c = getOrSplitOperand(B, CI->getArgOperand(2), dsMap, deferred);
      return emitDSFMA(B, a, b, c);
    }
  }

  return {nullptr, nullptr};
}

// df64 limbs produced for the rewritten instructions; getCompCost re-roots its
// cost walk on them because the original F64 roots are RAUW'd and DCE'd.
void takeLastExpansionLimbs(Session &S, SmallVectorImpl<Value *> &out) {
  out.append(S.Expansion.LastExpansionLimbs.begin(),
             S.Expansion.LastExpansionLimbs.end());
  S.Expansion.LastExpansionLimbs.clear();
}

void applyExpansion(Session &S, ArrayRef<Instruction *> instsToChange,
                    const SmallPtrSetImpl<Instruction *> &allChanged,
                    DenseMap<Value *, Value *> *restoredValues) {
  DSMap dsMap;
  DeferredDSPhiIns deferred;
  S.Expansion.LastExpansionLimbs.clear();
  // Captured before the erase loop below invalidates instsToChange.
  Function *parentF =
      instsToChange.empty() ? nullptr : instsToChange[0]->getFunction();

  // The emitted operations carry no fast-math flags, whatever the replaced
  // instruction had: an error-free transformation is exact only when every
  // operation is rounded as written.
  for (Instruction *I : instsToChange) {
    IRBuilder<> B(I);
    DSValue ds = emitDSForInstruction(B, I, dsMap, deferred);
    if (!ds.hi) {
      errs() << "[poseidon] expansion: unsupported instruction, skipping: "
             << *I << "\n";
      continue;
    }

    dsMap[I] = ds;
    if (ds.hi)
      S.Expansion.LastExpansionLimbs.push_back(ds.hi);
    if (ds.lo)
      S.Expansion.LastExpansionLimbs.push_back(ds.lo);
  }

  // Resolve deferred back-edge phi incomings now the subgraph is in dsMap;
  // iterative, since resolving one incoming may carry a nested phi.
  while (!deferred.empty()) {
    DeferredDSPhiIn d = deferred.pop_back_val();
    IRBuilder<> predB(d.block->getTerminator());
    DSValue inDS = getOrSplitOperand(predB, d.incoming, dsMap, deferred);
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
    Value *restored = emitDSToFP(RB, dsMap[phi], phi->getType());
    phi->replaceAllUsesWith(restored);
    if (restoredValues)
      (*restoredValues)[phi] = restored;
  }
  for (PHINode *phi : carriedPhis)
    phi->eraseFromParent();

  // The df64 parameter narrowing is value-preserving only once every load of
  // the buffer is a Dekker split, i.e. after the last unit over it, so it is
  // tried after each unit and self-gates until then. Here, not in the
  // post-pass, so foldDSPairRoundtrip sees the join/split pair it cancels. A
  // pricing clone is not a registered site, so the lookup fails and it bails.
  if (S.Opts.Materialization.NarrowParamStaging && parentF &&
      narrowSharedStagingParamDS(S, *parentF, parentF))
    foldDSPairRoundtrip(*parentF, /*foldedLimbs=*/nullptr,
                        S.Opts.Diagnostics.Print);
}

// Wider FP32 expansions (n = 3, 4): the QxW routines of
// Core/ExpansionRoutines.h. A fused multiply-add is materialized as
// multiply-then-add, as in the reference.

namespace {

// Only the component counts whose QxW routines are transcribed exist.
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
  r.x.resize(a.n());
  routines(B).expansionNeg(a.n(), a.x.data(), r.x.data());
  return r;
}

ExpansionValue emitExpansionAdd(IRBuilder<> &B, const ExpansionValue &a,
                                const ExpansionValue &b) {
  assert(a.n() == b.n() && "expansion operands must agree on component count");
  expansionCheckN(a.n());
  ExpansionValue r;
  r.x.resize(a.n());
  routines(B).expansionAdd(a.n(), a.x.data(), b.x.data(), r.x.data());
  return r;
}

ExpansionValue emitExpansionSub(IRBuilder<> &B, const ExpansionValue &a,
                                const ExpansionValue &b) {
  assert(a.n() == b.n() && "expansion operands must agree on component count");
  expansionCheckN(a.n());
  ExpansionValue r;
  r.x.resize(a.n());
  routines(B).expansionSub(a.n(), a.x.data(), b.x.data(), r.x.data());
  return r;
}

ExpansionValue emitExpansionMul(IRBuilder<> &B, const ExpansionValue &a,
                                const ExpansionValue &b) {
  assert(a.n() == b.n() && "expansion operands must agree on component count");
  expansionCheckN(a.n());
  ExpansionValue r;
  r.x.resize(a.n());
  routines(B).expansionMul(a.n(), a.x.data(), b.x.data(), r.x.data());
  return r;
}

ExpansionValue emitExpansionSqr(IRBuilder<> &B, const ExpansionValue &a) {
  expansionCheckN(a.n());
  ExpansionValue r;
  r.x.resize(a.n());
  routines(B).expansionSqr(a.n(), a.x.data(), r.x.data());
  return r;
}

ExpansionValue emitExpansionDiv(IRBuilder<> &B, const ExpansionValue &a,
                                const ExpansionValue &b) {
  assert(a.n() == b.n() && "expansion operands must agree on component count");
  expansionCheckN(a.n());
  ExpansionValue r;
  r.x.resize(a.n());
  routines(B).expansionDiv(a.n(), a.x.data(), b.x.data(), r.x.data());
  return r;
}

ExpansionValue emitExpansionSqrt(IRBuilder<> &B, const ExpansionValue &a) {
  expansionCheckN(a.n());
  ExpansionValue r;
  r.x.resize(a.n());
  routines(B).expansionSqrt(a.n(), a.x.data(), r.x.data());
  return r;
}

ExpansionValue emitToExpansion(IRBuilder<> &B, Value *fpval, unsigned n) {
  expansionCheckN(n);
  ExpansionValue r;
  r.x.resize(n);
  if (fpval->getType()->isFloatTy()) {
    routines(B).expansionFromF32(n, fpval, r.x.data());
    return r;
  }
  assert(fpval->getType()->isDoubleTy() &&
         "emitToExpansion: only f32/f64 sources");
  routines(B).expansionFromF64(n, fpval, r.x.data());
  return r;
}

Value *emitExpansionToFP(IRBuilder<> &B, const ExpansionValue &v,
                         Type *targetTy) {
  expansionCheckN(v.n());
  if (targetTy->isFloatTy())
    return routines(B).expansionToF32(v.n(), v.x.data());
  assert(targetTy->isDoubleTy() && "emitExpansionToFP: only f32/f64 targets");
  return routines(B).expansionToF64(v.n(), v.x.data());
}

// Materializer for n >= 3; mirrors applyExpansion without the two-component
// df64 staging machinery.

namespace {
struct DeferredExpansionPhiIn {
  SmallVector<PHINode *, 4> phis;
  Value *incoming;
  BasicBlock *block;
};
} // namespace
using DeferredExpansionPhiIns = SmallVector<DeferredExpansionPhiIn, 16>;
// In insertion order, as DSMap.
using ExpansionMap = MapVector<Value *, ExpansionValue>;

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
                           ExpansionMap &expansionMap,
                           DeferredExpansionPhiIns &deferred) {
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
          deferred.push_back({phis, in, inBB});
        }
      }
      return result;
    }
  }

  return emitToExpansion(B, op, n);
}

static ExpansionValue
emitExpansionForInstruction(IRBuilder<> &B, Instruction *I, unsigned n,
                            ExpansionMap &expansionMap,
                            DeferredExpansionPhiIns &deferred) {
  unsigned opcode = I->getOpcode();

  if (auto *BO = dyn_cast<BinaryOperator>(I)) {
    const bool sameOperand = BO->getOperand(0) == BO->getOperand(1);
    ExpansionValue lhs = getOrSplitOperandExpansion(B, BO->getOperand(0), n,
                                                    expansionMap, deferred);
    ExpansionValue rhs =
        sameOperand ? lhs
                    : getOrSplitOperandExpansion(B, BO->getOperand(1), n,
                                                 expansionMap, deferred);
    switch (opcode) {
    case Instruction::FAdd:
      return emitExpansionAdd(B, lhs, rhs);
    case Instruction::FSub:
      return emitExpansionSub(B, lhs, rhs);
    case Instruction::FMul:
      return sameOperand ? emitExpansionSqr(B, lhs)
                         : emitExpansionMul(B, lhs, rhs);
    case Instruction::FDiv:
      return emitExpansionDiv(B, lhs, rhs);
    default:
      break;
    }
  }

  if (auto *UO = dyn_cast<UnaryOperator>(I))
    if (opcode == Instruction::FNeg)
      return emitExpansionNeg(
          B, getOrSplitOperandExpansion(B, UO->getOperand(0), n, expansionMap,
                                        deferred));

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
    } else if (StringRef device = deviceMathName(callee->getName());
               !device.empty()) {
      mathName = device;
    } else {
      StringRef name = callee->getName();
      if (!name.empty() && (name.back() == 'f' || name.back() == 'l'))
        name = name.drop_back(1);
      mathName = name;
    }

    if (mathName == "sqrt")
      return emitExpansionSqrt(
          B, getOrSplitOperandExpansion(B, CI->getArgOperand(0), n,
                                        expansionMap, deferred));
    if (mathName == "fmuladd" || mathName == "fma") {
      const bool sameFactor = CI->getArgOperand(0) == CI->getArgOperand(1);
      ExpansionValue a = getOrSplitOperandExpansion(B, CI->getArgOperand(0), n,
                                                    expansionMap, deferred);
      ExpansionValue b =
          sameFactor ? a
                     : getOrSplitOperandExpansion(B, CI->getArgOperand(1), n,
                                                  expansionMap, deferred);
      ExpansionValue c = getOrSplitOperandExpansion(B, CI->getArgOperand(2), n,
                                                    expansionMap, deferred);
      return emitExpansionAdd(
          B, sameFactor ? emitExpansionSqr(B, a) : emitExpansionMul(B, a, b),
          c);
    }
  }

  return ExpansionValue{};
}

// Separate from LastExpansionLimbs: one candidate can apply both an expansion
// change and a double-single change, and applyExpansion clears its list on
// entry.
void takeLastExpLimbs(Session &S, SmallVectorImpl<Value *> &out) {
  out.append(S.Expansion.LastExpLimbs.begin(), S.Expansion.LastExpLimbs.end());
  S.Expansion.LastExpLimbs.clear();
}
// Cleared once per candidate, not per change: a tier pair calls applyExpansion
// twice.
void resetExpLimbs(Session &S) { S.Expansion.LastExpLimbs.clear(); }

void applyExpansion(Session &S, unsigned n,
                    ArrayRef<Instruction *> instsToChange,
                    const SmallPtrSetImpl<Instruction *> &allChanged,
                    DenseMap<Value *, Value *> *restoredValues) {
  expansionCheckN(n);
  ExpansionMap expansionMap;
  DeferredExpansionPhiIns deferred;

  // No fast-math flags on the emitted operations, as in the two-component arm.
  for (Instruction *I : instsToChange) {
    IRBuilder<> B(I);
    ExpansionValue mf =
        emitExpansionForInstruction(B, I, n, expansionMap, deferred);
    if (mf.x.empty()) {
      errs() << "Expansion(" << n
             << "): unsupported instruction, skipping: " << *I << "\n";
      continue;
    }
    expansionMap[I] = mf;
    for (Value *v : mf.x)
      S.Expansion.LastExpLimbs.push_back(v);
  }

  while (!deferred.empty()) {
    DeferredExpansionPhiIn d = deferred.pop_back_val();
    IRBuilder<> predB(d.block->getTerminator());
    ExpansionValue inExpansion = getOrSplitOperandExpansion(
        predB, d.incoming, n, expansionMap, deferred);
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
    Value *restored = emitExpansionToFP(RB, expansionMap[phi], phi->getType());
    phi->replaceAllUsesWith(restored);
    if (restoredValues)
      (*restoredValues)[phi] = restored;
  }
  for (PHINode *phi : carriedPhis)
    phi->eraseFromParent();
}

#if LLVM_VERSION_MAJOR < 24
// Before LLVM 24 fast contraction and fast-math attributes also reach
// operations without flags, so expansion code is not rounded as written there.
// Returns why for F, or "".
static std::string fastFPReason(const Function &F) {
  SmallVector<StringRef, 4> attrs;
#if LLVM_VERSION_MAJOR < 22
  attrs.append({"unsafe-fp-math", "approx-func-fp-math"}); // read up to LLVM 21
#endif
#if LLVM_VERSION_MAJOR < 23
  attrs.append({"no-nans-fp-math", "no-infs-fp-math"}); // read up to LLVM 22
#endif
  for (StringRef attr : attrs)
    if (F.getFnAttribute(attr).getValueAsString() == "true")
      return ("the function carries \"" + attr +
              "\"=\"true\", which this LLVM applies to every floating-point "
              "operation of the function, with fast-math flags or without")
          .str();
  for (const Instruction &I : instructions(F))
    if (isa<FPMathOperator>(I) && I.getFastMathFlags().allowContract()) {
      const auto *CI = dyn_cast<CallInst>(&I);
      const Function *callee = CI ? CI->getCalledFunction() : nullptr;
      return ("its floating-point operations carry 'contract' (the first " +
              (callee ? callee->getName() : StringRef(I.getOpcodeName())) +
              "), as under -ffp-contract=fast or -ffast-math, with which this "
              "LLVM's code generator also fuses multiply-add pairs that carry "
              "no flag")
          .str();
    }
  return "";
}
#endif

bool expansionsRoundedAsWritten(const Options &O, Function &F,
                                bool hasSubgraphs) {
  bool expansionSafe = true;
#if LLVM_VERSION_MAJOR < 24
  if (O.Candidates.EnablePT && O.Candidates.EnableMultifloat &&
      Target::of(F).isDevice() && hasSubgraphs) {
    const std::string reason = fastFPReason(F);
    expansionSafe = reason.empty();
    if (!expansionSafe) {
      llvm::errs() << "[poseidon] WARNING: " << F.getName()
                   << ": floating-point expansion candidates NOT PROPOSED -- "
                   << reason
                   << ". An expansion is exact only if each of its operations "
                      "is rounded as written; it needs -ffp-contract=on or "
                      "-ffp-contract=off, without -ffast-math.\n";
      if (O.Model.StrictMode)
        reportUserError(F, F.getSubprogram(),
                        "strict mode: site '" + F.getName() + "': " + reason +
                            "; floating-point expansions need "
                            "-ffp-contract=on or -ffp-contract=off");
    }
  }
#endif
  return expansionSafe;
}

} // namespace poseidon
