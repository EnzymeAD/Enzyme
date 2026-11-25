//===- Expansion.cpp - floating-point expansions over FP32 ----------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Expansion.h"
#include "Core/ExpansionRoutines.h"
#include "Core/Options.h"
#include "Enzyme/Utils.h"
#include "IR/Utils.h"
#include "Targets/MathLibrary.h"
#include "Targets/Target.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>

using namespace llvm;

namespace poseidon {

namespace {

// The two components of a two-component expansion, the routines' pair.
struct DSValue {
  Value *hi;
  Value *lo;
};

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
using PairOp1 = DSValue (Routines::*)(DSValue);
using PairOp2 = DSValue (Routines::*)(DSValue, DSValue);
using WideOp1 = void (Routines::*)(unsigned, Value *const *, Value **);
using WideOp2 = void (Routines::*)(unsigned, Value *const *, Value *const *,
                                   Value **);

} // namespace

static Routines routines(IRBuilder<> &B) { return Routines(IRArith{B}); }

// The metadata on the FP64 restore of a pair, which a Dekker split recovers
// exactly.
static constexpr StringLiteral kPairJoin = "poseidon.ds.join";

// Only the component counts whose routines are transcribed exist.
static void checkComponents(unsigned n) {
  if (n < 2 || n > 4)
    report_fatal_error("Poseidon expansion emitter: only 2-, 3- and "
                       "4-component FP32 expansions are implemented. Refusing "
                       "to fabricate an unimplemented component count.");
}

static DSValue pairOf(const ExpansionValue &v) {
  assert(v.n() == 2 && "a pair is a two-component expansion");
  return {v.x[0], v.x[1]};
}

static ExpansionValue expansionOf(DSValue p) {
  ExpansionValue r;
  r.x.assign({p.hi, p.lo});
  return r;
}

static ExpansionValue emitUnary(IRBuilder<> &B, const ExpansionValue &a,
                                PairOp1 pairOp, WideOp1 wideOp) {
  Routines R = routines(B);
  if (a.n() == 2)
    return expansionOf((R.*pairOp)(pairOf(a)));
  ExpansionValue r;
  r.x.resize(a.n());
  (R.*wideOp)(a.n(), a.x.data(), r.x.data());
  return r;
}

static ExpansionValue emitBinary(IRBuilder<> &B, const ExpansionValue &a,
                                 const ExpansionValue &b, PairOp2 pairOp,
                                 WideOp2 wideOp) {
  assert(a.n() == b.n() && "expansion operands must agree on component count");
  Routines R = routines(B);
  if (a.n() == 2)
    return expansionOf((R.*pairOp)(pairOf(a), pairOf(b)));
  ExpansionValue r;
  r.x.resize(a.n());
  (R.*wideOp)(a.n(), a.x.data(), b.x.data(), r.x.data());
  return r;
}

ExpansionValue emitExpansionAdd(IRBuilder<> &B, const ExpansionValue &a,
                                const ExpansionValue &b) {
  return emitBinary(B, a, b, &Routines::dsAdd, &Routines::expansionAdd);
}

ExpansionValue emitExpansionSub(IRBuilder<> &B, const ExpansionValue &a,
                                const ExpansionValue &b) {
  return emitBinary(B, a, b, &Routines::dsSub, &Routines::expansionSub);
}

ExpansionValue emitExpansionMul(IRBuilder<> &B, const ExpansionValue &a,
                                const ExpansionValue &b) {
  return emitBinary(B, a, b, &Routines::dsMul, &Routines::expansionMul);
}

ExpansionValue emitExpansionSqr(IRBuilder<> &B, const ExpansionValue &a) {
  if (a.n() == 2)
    return emitExpansionMul(B, a, a);
  ExpansionValue r;
  r.x.resize(a.n());
  routines(B).expansionSqr(a.n(), a.x.data(), r.x.data());
  return r;
}

ExpansionValue emitExpansionDiv(IRBuilder<> &B, const ExpansionValue &a,
                                const ExpansionValue &b) {
  return emitBinary(B, a, b, &Routines::dsDiv, &Routines::expansionDiv);
}

ExpansionValue emitExpansionSqrt(IRBuilder<> &B, const ExpansionValue &a) {
  return emitUnary(B, a, &Routines::dsSqrt, &Routines::expansionSqrt);
}

ExpansionValue emitExpansionNeg(IRBuilder<> &B, const ExpansionValue &a) {
  return emitUnary(B, a, &Routines::dsNeg, &Routines::expansionNeg);
}

ExpansionValue emitExpansionFMA(IRBuilder<> &B, const ExpansionValue &a,
                                const ExpansionValue &b,
                                const ExpansionValue &c) {
  if (a.n() == 2)
    return expansionOf(routines(B).dsFma(pairOf(a), pairOf(b), pairOf(c)));
  ExpansionValue product = emitExpansionMul(B, a, b);
  return emitExpansionAdd(B, product, c);
}

ExpansionValue emitExpansionFMASqr(IRBuilder<> &B, const ExpansionValue &a,
                                   const ExpansionValue &c) {
  if (a.n() == 2)
    return emitExpansionFMA(B, a, a, c);
  ExpansionValue square = emitExpansionSqr(B, a);
  return emitExpansionAdd(B, square, c);
}

ExpansionValue emitToExpansion(IRBuilder<> &B, Value *fpval, unsigned n) {
  checkComponents(n);
  const bool f32 = fpval->getType()->isFloatTy();
  assert((f32 || fpval->getType()->isDoubleTy()) &&
         "emitToExpansion: only f32/f64 sources");
  Routines R = routines(B);
  if (n == 2)
    return expansionOf(f32 ? R.dsFromF32(fpval) : R.dsFromF64(fpval));
  ExpansionValue r;
  r.x.resize(n);
  if (f32)
    R.expansionFromF32(n, fpval, r.x.data());
  else
    R.expansionFromF64(n, fpval, r.x.data());
  return r;
}

Value *emitExpansionToFP(IRBuilder<> &B, const ExpansionValue &v,
                         Type *targetTy) {
  Routines R = routines(B);
  if (targetTy->isFloatTy())
    return v.n() == 2 ? R.dsToF32(pairOf(v))
                      : R.expansionToF32(v.n(), v.x.data());
  assert(targetTy->isDoubleTy() && "emitExpansionToFP: only f32/f64 targets");
  if (v.n() != 2)
    return R.expansionToF64(v.n(), v.x.data());
  Value *joined = R.dsToF64(pairOf(v));
  // Every pair here is normalized, so a Dekker split of this restore returns
  // the same two limbs; only tagged restores are folded.
  if (auto *I = dyn_cast<Instruction>(joined))
    I->setMetadata(kPairJoin, MDNode::get(I->getContext(), {}));
  return joined;
}

void DekkerSplit::replaceWith(Value *hiV, Value *loV) {
  hi->replaceAllUsesWith(hiV);
  lo->replaceAllUsesWith(loV);
  lo->eraseFromParent();
  los->eraseFromParent();
  hib->eraseFromParent();
  hi->eraseFromParent();
}

bool matchDekkerSplits(Value *V, bool allowFCmp,
                       SmallVectorImpl<DekkerSplit> &splits) {
  splits.clear();
  SmallVector<FPTruncInst *, 4> his;
  SmallVector<BinaryOperator *, 4> loss;
  for (User *U : V->users()) {
    auto *FT = dyn_cast<FPTruncInst>(U);
    auto *BO = dyn_cast<BinaryOperator>(U);
    if (FT && FT->getType()->isFloatTy())
      his.push_back(FT);
    else if (BO && BO->getOpcode() == Instruction::FSub &&
             BO->getOperand(0) == V && BO->getType()->isDoubleTy())
      loss.push_back(BO);
    else if (!allowFCmp || !isa<FCmpInst>(U))
      return false;
  }
  if (his.empty() || his.size() != loss.size())
    return false;
  SmallPtrSet<FPTruncInst *, 4> paired;
  for (BinaryOperator *los : loss) {
    auto *hib = dyn_cast<FPExtInst>(los->getOperand(1));
    if (!hib || !hib->hasOneUse() || !los->hasOneUse())
      return false;
    auto *hi = dyn_cast<FPTruncInst>(hib->getOperand(0));
    auto *lo = dyn_cast<FPTruncInst>(*los->user_begin());
    if (!hi || !is_contained(his, hi) || !paired.insert(hi).second || !lo ||
        !lo->getType()->isFloatTy())
      return false;
    splits.push_back({hi, lo, hib, los});
  }
  return true;
}

// Exact for a normalized pair, bit for bit, so only joins tagged by
// emitExpansionToFP are folded. InstCombine does not fold the pair because the
// split reads the join twice.
bool foldDSPairRoundtrip(Function &F, SmallVectorImpl<Value *> *foldedLimbs,
                         bool print) {
  SmallVector<Instruction *> joins;
  for (Instruction &I : instructions(F))
    if (I.getMetadata(kPairJoin))
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
    SmallVector<DekkerSplit, 1> splits;
    if (!matchDekkerSplits(J, /*allowFCmp=*/false, splits) ||
        splits.size() != 1)
      continue;

    splits.front().replaceWith(hiLimb, loLimb);
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
        << F.getName() << '\n';
  return changed;
}

// The math function a call to `callee` computes: sqrt, fma or fmuladd for an
// intrinsic, else mathFunctionOf's answer for its name, the one its enzyme_math
// attribute gives when it has one.
static StringRef mathNameOf(Function &callee) {
  if (callee.isIntrinsic()) {
    Intrinsic::ID id = callee.getIntrinsicID();
    if (id == Intrinsic::sqrt)
      return "sqrt";
    if (id == Intrinsic::fmuladd)
      return "fmuladd";
    if (id == Intrinsic::fma)
      return "fma";
    return StringRef();
  }
  return mathFunctionOf(getFuncName(&callee));
}

namespace {

// A loop-carried incoming of the limb phis `phis`, resolved once the subgraph
// is converted: the accumulator phi and the accumulation feeding it are
// mutually dependent, and resolving the back edge eagerly would collapse the
// phi to a single double.
struct DeferredPhiIn {
  SmallVector<PHINode *, 4> phis;
  Value *incoming;
  BasicBlock *block;
};

} // namespace

// In insertion order: the restores of the replaced values are emitted by
// iterating it, and several can share one insertion point.
using ExpansionMap = MapVector<Value *, ExpansionValue>;

// The greedy split of a constant into n FP32 limbs.
static ExpansionValue splitConstantFP(IRBuilder<> &B, ConstantFP *CFP,
                                      unsigned n) {
  double val = CFP->getValueAPF().convertToDouble();
  ExpansionValue r;
  double rem = val;
  for (unsigned i = 0; i < n; ++i) {
    float limb = static_cast<float>(rem);
    r.x.push_back(ConstantFP::get(B.getFloatTy(), limb));
    rem -= static_cast<double>(limb);
  }
  return r;
}

// Any scalar f32/f64 value has an expansion (materialized op, split constant,
// carried phi, or boundary emitToExpansion).
static bool hasExpansion(Value *op) {
  Type *T = op->getType();
  return T->isFloatTy() || T->isDoubleTy();
}

// The name of limb c of a carried phi, after the routines' instructions: ds.*
// for a pair, mfx.* for a wider expansion.
static const char *limbPhiName(unsigned n, unsigned c) {
  if (n == 2)
    return c == 0 ? "ds.phi.hi" : "ds.phi.lo";
  return "mfx.phi";
}

// The phi `phi` carried as n limb phis, recorded in `expansionMap`; an
// incoming not converted yet is a loop back edge, left in `deferred`.
static ExpansionValue expandPhi(IRBuilder<> &B, PHINode *phi, unsigned n,
                                ExpansionMap &expansionMap,
                                SmallVectorImpl<DeferredPhiIn> &deferred) {
  IRBuilder<> phiB(phi);
  unsigned nIn = phi->getNumIncomingValues();
  ExpansionValue result;
  SmallVector<PHINode *, 4> phis;
  for (unsigned c = 0; c < n; ++c) {
    PHINode *limb = phiB.CreatePHI(B.getFloatTy(), nIn, limbPhiName(n, c));
    phis.push_back(limb);
    result.x.push_back(limb);
  }
  expansionMap[phi] = result;
  for (unsigned i = 0; i < nIn; ++i) {
    Value *in = phi->getIncomingValue(i);
    BasicBlock *inBB = phi->getIncomingBlock(i);
    if (auto jt = expansionMap.find(in); jt != expansionMap.end()) {
      for (unsigned c = 0; c < n; ++c)
        phis[c]->addIncoming(jt->second.x[c], inBB);
    } else if (auto *CFP = dyn_cast<ConstantFP>(in)) {
      ExpansionValue inExpansion = splitConstantFP(B, CFP, n);
      for (unsigned c = 0; c < n; ++c)
        phis[c]->addIncoming(inExpansion.x[c], inBB);
    } else {
      deferred.push_back({phis, in, inBB});
    }
  }
  return result;
}

// The expansion of `op`: its converted value, a split constant, a carried
// phi, or the conversion of a value from outside the subgraph.
static ExpansionValue
getOrSplitOperand(IRBuilder<> &B, Value *op, unsigned n,
                  ExpansionMap &expansionMap,
                  SmallVectorImpl<DeferredPhiIn> &deferred) {
  if (auto it = expansionMap.find(op); it != expansionMap.end())
    return it->second;
  if (auto *CFP = dyn_cast<ConstantFP>(op))
    return splitConstantFP(B, CFP, n);
  if (auto *phi = dyn_cast<PHINode>(op);
      phi && hasExpansion(phi) && all_of(phi->incoming_values(), hasExpansion))
    return expandPhi(B, phi, n, expansionMap, deferred);
  return emitToExpansion(B, op, n);
}

// I as an n-component expansion, or none for an operation without a routine.
// A wider expansion converts an operand repeated in I once and squares a
// product of it; a pair converts each read and multiplies.
static ExpansionValue
emitExpansionForInstruction(IRBuilder<> &B, Instruction *I, unsigned n,
                            ExpansionMap &expansionMap,
                            SmallVectorImpl<DeferredPhiIn> &deferred) {
  auto operand = [&](Value *op) {
    return getOrSplitOperand(B, op, n, expansionMap, deferred);
  };
  const bool shareRepeated = n != 2;
  unsigned opcode = I->getOpcode();

  if (auto *BO = dyn_cast<BinaryOperator>(I)) {
    const bool sameOperand =
        shareRepeated && BO->getOperand(0) == BO->getOperand(1);
    ExpansionValue lhs = operand(BO->getOperand(0));
    ExpansionValue rhs = sameOperand ? lhs : operand(BO->getOperand(1));
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

  if (isa<UnaryOperator>(I) && opcode == Instruction::FNeg) {
    ExpansionValue x = operand(I->getOperand(0));
    return emitExpansionNeg(B, x);
  }

  if (auto *CI = dyn_cast<CallInst>(I)) {
    Function *callee = CI->getCalledFunction();
    if (!callee)
      return ExpansionValue{};

    StringRef mathName = mathNameOf(*callee);

    if (mathName == "sqrt") {
      ExpansionValue x = operand(CI->getArgOperand(0));
      return emitExpansionSqrt(B, x);
    }
    if (mathName == "fmuladd" || mathName == "fma") {
      const bool sameFactor =
          shareRepeated && CI->getArgOperand(0) == CI->getArgOperand(1);
      ExpansionValue a = operand(CI->getArgOperand(0));
      ExpansionValue b = sameFactor ? a : operand(CI->getArgOperand(1));
      ExpansionValue c = operand(CI->getArgOperand(2));
      return sameFactor ? emitExpansionFMASqr(B, a, c)
                        : emitExpansionFMA(B, a, b, c);
    }
  }

  return ExpansionValue{};
}

// Gives every use of a converted instruction outside the conversion the
// instruction's restore, recorded in `restoredValues` if given.
static void restoreExternalUses(ExpansionMap &expansionMap,
                                const SmallPtrSetImpl<Instruction *> &changed,
                                DenseMap<Value *, Value *> *restoredValues) {
  for (auto &[val, v] : expansionMap) {
    auto *I = dyn_cast<Instruction>(val);
    if (!I || !changed.count(I))
      continue;
    SmallVector<Use *, 4> externalUses;
    for (Use &U : I->uses()) {
      auto *userI = dyn_cast<Instruction>(U.getUser());
      if (!userI || !expansionMap.count(userI) || !changed.count(userI))
        externalUses.push_back(&U);
    }
    if (externalUses.empty())
      continue;
    BasicBlock::iterator restorePos = std::next(BasicBlock::iterator(I));
    if (isa<PHINode>(I))
      // NOLINTNEXTLINE(enzyme-first-non-phi): ahead of the debug records
      restorePos = I->getParent()->getFirstNonPHIIt();
    IRBuilder<> RestoreB = builderAt(I->getParent(), restorePos);
    Value *restored = emitExpansionToFP(RestoreB, v, I->getType());
    assert(restored->getType() == I->getType() &&
           "unexpected restored value type");
    for (Use *U : externalUses)
      U->set(restored);
    if (restoredValues)
      (*restoredValues)[I] = restored;
  }
}

// Replaces each FP64 phi carried as an expansion, which the conversion leaves,
// by one restore of its limbs: restoring it per use would collapse the
// expansion every iteration.
static void restoreCarriedPhis(ExpansionMap &expansionMap,
                               DenseMap<Value *, Value *> *restoredValues) {
  SmallVector<PHINode *, 8> carriedPhis;
  for (auto &[val, v] : expansionMap)
    if (auto *phi = dyn_cast<PHINode>(val))
      if (!v.x.empty() && phi->getType()->isDoubleTy())
        carriedPhis.push_back(phi);
  for (PHINode *phi : carriedPhis) {
    // NOLINTNEXTLINE(enzyme-first-non-phi): ahead of the debug records
    BasicBlock::iterator at = phi->getParent()->getFirstNonPHIIt();
    IRBuilder<> RB = builderAt(phi->getParent(), at);
    Value *restored = emitExpansionToFP(RB, expansionMap[phi], phi->getType());
    phi->replaceAllUsesWith(restored);
    if (restoredValues)
      (*restoredValues)[phi] = restored;
  }
  for (PHINode *phi : carriedPhis)
    phi->eraseFromParent();
}

void applyExpansion(unsigned n, ArrayRef<Instruction *> instsToChange,
                    DenseMap<Value *, Value *> *restoredValues,
                    SmallVectorImpl<Value *> *limbs, bool print) {
  checkComponents(n);
  SmallPtrSet<Instruction *, 8> changed(instsToChange.begin(),
                                        instsToChange.end());
  ExpansionMap expansionMap;
  SmallVector<DeferredPhiIn, 16> deferred;

  // The emitted operations carry no fast-math flags, whatever the replaced
  // instruction had: an error-free transformation is exact only when every
  // operation is rounded as written.
  for (Instruction *I : instsToChange) {
    IRBuilder<> B(I);
    ExpansionValue v =
        emitExpansionForInstruction(B, I, n, expansionMap, deferred);
    if (v.x.empty()) {
      if (print)
        llvm::errs() << "[poseidon] expansion: unsupported instruction, "
                        "skipping: "
                     << *I << '\n';
      continue;
    }
    expansionMap[I] = v;
    if (limbs)
      limbs->append(v.x.begin(), v.x.end());
  }

  // Resolving one back-edge incoming may carry a nested phi, which defers
  // another.
  while (!deferred.empty()) {
    DeferredPhiIn d = deferred.pop_back_val();
    IRBuilder<> predB(d.block->getTerminator());
    ExpansionValue inExpansion =
        getOrSplitOperand(predB, d.incoming, n, expansionMap, deferred);
    for (unsigned c = 0; c < n; ++c)
      d.phis[c]->addIncoming(inExpansion.x[c], d.block);
  }

  restoreExternalUses(expansionMap, changed, restoredValues);
  for (Instruction *I : reverse(instsToChange)) {
    if (!expansionMap.count(I))
      continue;
    if (!I->use_empty()) {
      assert(restoredValues && restoredValues->count(I) &&
             "every reader of an expanded instruction is expanded or reads "
             "its restore");
      I->replaceAllUsesWith((*restoredValues)[I]);
    }
    I->eraseFromParent();
  }
  restoreCarriedPhis(expansionMap, restoredValues);
}

#if LLVM_VERSION_MAJOR < 24
// Before LLVM 24 fast contraction and fast-math attributes also reach
// operations without flags, so expansion code is not rounded as written there:
// why for F, if so.
static std::optional<std::string> fastFPReason(const Function &F) {
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
  return std::nullopt;
}
#endif

bool expansionsRoundedAsWritten(const Options &O, Function &F,
                                bool hasSubgraphs) {
  bool expansionSafe = true;
#if LLVM_VERSION_MAJOR < 24
  if (O.Candidates.EnablePT && O.Candidates.EnableMultifloat &&
      Target::of(F).isDevice() && hasSubgraphs) {
    const std::optional<std::string> reason = fastFPReason(F);
    expansionSafe = !reason;
    if (reason) {
      llvm::errs() << "[poseidon] WARNING: " << F.getName()
                   << ": floating-point expansion candidates NOT PROPOSED -- "
                   << *reason
                   << ". An expansion is exact only if each of its operations "
                      "is rounded as written; it needs -ffp-contract=on or "
                      "-ffp-contract=off, without -ffast-math.\n";
      if (O.Model.StrictMode)
        reportUserError(F, F.getSubprogram(),
                        "strict mode: site '" + F.getName() + "': " + *reason +
                            "; floating-point expansions need "
                            "-ffp-contract=on or -ffp-contract=off");
    }
  }
#endif
  return expansionSafe;
}

} // namespace poseidon
