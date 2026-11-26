//===- Precision.cpp - precision candidates of an FP subgraph -------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The precision unit of an FP subgraph: assignments of its operations to
// narrower or wider formats and to floating-point expansions.
//
//===----------------------------------------------------------------------===//

#include "Precision.h"
#include "Core/FPCore.h"
#include "Expansion.h"
#include "ExpansionModel.h"
#include "Model/CostModel.h"
#include "Model/Evaluators.h"
#include "Model/Sampling.h"
#include "Narrowing.h"
#include "Profile/ProfileReader.h"
#include "Sites/Canonicalize.h"
#include "Solve/Explain.h"
#include "Solve/Session.h"
#include "Support/Diagnostics.h"
#include "Support/Parallel.h"
#include "Support/Utils.h"
#include "Targets/MathLibrary.h"
#include "Targets/Target.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace llvm;

namespace poseidon {

//===----------------------------------------------------------------------===//
// Materializing a candidate
//===----------------------------------------------------------------------===//

// The call CI of a library function, in `newType` with the arguments
// `newArgs`: the function's variant for that type, through FP32 for a
// half-precision type the library has no variant for.
static Value *changedLibmCall(IRBuilder<> &B, CallInst *CI,
                              ArrayRef<Value *> newArgs, Type *newType) {
  StringRef funcName = CI->getCalledFunction()->getName();
  std::string newFuncName = getLibmFunctionForPrecision(funcName, newType);
  if (newFuncName.empty())
    report_fatal_error("Poseidon: no " + formatOf(newType)->Name +
                       " variant of the library call " + getValueStr(CI));

  Type *funcType = newType;
  const bool needsPromotion = newType->isHalfTy() || newType->isBFloatTy();
  if (needsPromotion)
    funcType = Type::getFloatTy(newType->getContext());

  SmallVector<Value *, 4> callArgs;
  for (Value *arg : newArgs) {
    if (needsPromotion && arg->getType()->isFloatingPointTy() &&
        arg->getType() != funcType)
      callArgs.push_back(B.CreateFPExt(arg, funcType, "poseidon.promote"));
    else
      callArgs.push_back(arg);
  }

  SmallVector<Type *, 4> funcArgTypes(callArgs.size(), funcType);
  FunctionCallee newFuncCallee = CI->getModule()->getOrInsertFunction(
      newFuncName, FunctionType::get(funcType, funcArgTypes, false));
  Value *callResult =
      createMathCall(B, cast<Function>(newFuncCallee.getCallee()), callArgs);
  if (needsPromotion)
    return B.CreateFPTrunc(callResult, newType, "poseidon.demote");
  return callResult;
}

// The call CI in `newType` with the arguments `newArgs`.
static Value *changedCall(IRBuilder<> &B, CallInst *CI,
                          ArrayRef<Value *> newArgs, Type *newType) {
  Function *calledFunc = CI->getCalledFunction();
  assert(calledFunc && "isOptimizable admits only calls of a known callee");
  if (calledFunc->isIntrinsic()) {
    Intrinsic::ID intrinsicID = calledFunc->getIntrinsicID();
    SmallVector<Type *, 2> overloadedTypes = {newType};
    if (intrinsicID == Intrinsic::powi)
      overloadedTypes.push_back(CI->getArgOperand(1)->getType());
    Function *newFunc = Intrinsic::getOrInsertDeclaration(
        CI->getModule(), intrinsicID, overloadedTypes);
    return B.CreateCall(newFunc, newArgs);
  }

  // An F32-downcast sqrt becomes the target's approximate square root where it
  // has one, instead of a correctly-rounded library call that would dominate
  // sqrt-heavy kernels; the accuracy model charges the approximation's error.
  Intrinsic::ID approxSqrt = approxSqrtF32Of(*CI);
  if (newType->isFloatTy() && approxSqrt != Intrinsic::not_intrinsic) {
    Function *fastSqrt =
        Intrinsic::getOrInsertDeclaration(CI->getModule(), approxSqrt, {});
    return B.CreateCall(fastSqrt, {newArgs[0]});
  }

  // fmax and fmin become maxnum and minnum, which have the same IEEE-754
  // semantics and are modelled: libdevice is linked before this pass, so a
  // fresh __nv_fmaxf declaration would reach ptxas undefined.
  StringRef base = mathFunctionOf(calledFunc->getName());
  if ((base == "fmax" || base == "fmin") && newArgs.size() == 2)
    return B.CreateBinaryIntrinsic(base == "fmax" ? Intrinsic::maxnum
                                                  : Intrinsic::minnum,
                                   newArgs[0], newArgs[1]);

  return changedLibmCall(B, CI, newArgs, newType);
}

// Emits I in the precision of `change`, reading the operands already changed
// from `oldToNew` and casting the others, and records the new value there.
static void changePrecision(Instruction *I, PrecisionChange &change,
                            MapVector<Value *, Value *> &oldToNew) {
  assert(isOptimizable(*I) &&
         "Trying to tune an instruction is not isOptimizable");

  IRBuilder<> Builder(I);
  Builder.setFastMathFlags(I->getFastMathFlags());
  Type *newType = storageTypeOf(I->getContext(), change.newType);

  // An operand in the new precision: its changed value, or a cast of it.
  auto convert = [&](Value *operand) -> Value * {
    if (Value *changed = oldToNew.lookup(operand))
      return changed;
    if (operand->getType()->isIntegerTy())
      return operand;
    if (isa<Constant>(operand))
      return Builder.CreateFPCast(operand, newType, "poseidon.const.fpcast");
    assert((isa<Argument>(operand) || isa<Instruction>(operand)) &&
           "an operand is a constant, an argument or an instruction");
    return Builder.CreateFPCast(operand, newType, "poseidon.fpcast");
  };

  if (isa<UnaryOperator, BinaryOperator>(I)) {
    SmallVector<Value *, 2> newOps;
    for (Value *operand : I->operands())
      newOps.push_back(convert(operand));
    oldToNew[I] = Builder.CreateNAryOp(I->getOpcode(), newOps);
    return;
  }
  auto *CI = dyn_cast<CallInst>(I);
  if (!CI)
    report_fatal_error(Twine("Poseidon: a precision change of an operation "
                             "that is neither arithmetic nor a call: ") +
                       getValueStr(I));
  SmallVector<Value *, 4> newArgs;
  for (Value *arg : CI->args())
    newArgs.push_back(convert(arg));
  oldToNew[I] = changedCall(Builder, CI, newArgs, newType);
}

namespace {

// Where a candidate's changes are applied: the site, or its clone under VMap.
class ChangeTarget {
public:
  ChangeTarget(Session &S, Subgraph &subgraph, ValueToValueMapTy *VMap,
               FPGraph *graph);

  // The operations `change` converts that are still to be changed, in the
  // clone under VMap.
  SetVector<Instruction *> operationsOf(const PrecisionChange &change) const;

  // Converts `sorted`, in operand order, to an n-component expansion.
  // `components` receives the components of each converted instruction.
  void expand(unsigned n, ArrayRef<Instruction *> sorted,
              SmallVectorImpl<Value *> &components);

  // Converts `insts`, `sorted` in operand order, to the format of `change`,
  // giving their readers outside the change the value cast back.
  void changeFormat(PrecisionChange &change,
                    const SetVector<Instruction *> &insts,
                    ArrayRef<Instruction *> sorted);

private:
  Session &S;
  Subgraph &subgraph;
  ValueToValueMapTy *VMap;
  FPGraph *graph;
  SetVector<Instruction *> operations;
  DenseMap<Value *, Value *> clonedToOriginal;
};

} // namespace

ChangeTarget::ChangeTarget(Session &S, Subgraph &subgraph,
                           ValueToValueMapTy *VMap, FPGraph *graph)
    : S(S), subgraph(subgraph), VMap(VMap), graph(graph) {
  if (!VMap) {
    operations = subgraph.operations;
    return;
  }
  for (Instruction *I : subgraph.operations) {
    assert(VMap->count(I) && "the clone maps every operation");
    if (auto *MappedI = dyn_cast_or_null<Instruction>(VMap->lookup(I))) {
      operations.insert(MappedI);
      clonedToOriginal[MappedI] = I;
    }
  }
}

SetVector<Instruction *>
ChangeTarget::operationsOf(const PrecisionChange &change) const {
  SetVector<Instruction *> insts;
  for (FPLLValue *node : change.nodes) {
    if (!node || !node->value)
      continue;
    auto *I = cast<Instruction>(node->value);
    if (VMap) {
      assert(VMap->count(I) && "the clone maps every operation");
      I = dyn_cast_or_null<Instruction>(VMap->lookup(I));
      if (!I)
        continue;
    }
    // An operation an algebraic rewrite erased is no longer among them.
    if (operations.contains(I))
      insts.insert(I);
  }
  return insts;
}

void ChangeTarget::expand(unsigned n, ArrayRef<Instruction *> sorted,
                          SmallVectorImpl<Value *> &components) {
  DenseMap<Value *, Value *> restoredValues;
  Function *F = sorted.empty() ? nullptr : sorted.front()->getFunction();
  applyExpansion(n, sorted, &restoredValues, &components,
                 S.Opts.Diagnostics.Print);
  // The pair narrowing preserves values only once every load of the buffer is
  // a Dekker split, so it is tried after each pair change and declines until
  // then. It runs here so that foldExpansionRoundtrip sees the join/split pair
  // it cancels. A pricing clone is not a registered site, and the narrowing
  // declines on it.
  if (n == 2 && F && narrowToPairs(S, *F, *F, /*announce=*/!VMap))
    foldExpansionRoundtrip(*F, /*foldedComponents=*/nullptr,
                           S.Opts.Diagnostics.Print);
  if (VMap) {
    for (auto &[oldClonedI, restoredV] : restoredValues)
      if (clonedToOriginal.count(oldClonedI))
        (*VMap)[clonedToOriginal[oldClonedI]] = restoredV;
    return;
  }
  for (Instruction *I : sorted)
    subgraph.operations.remove(I);
  if (graph)
    for (auto &[erased, restored] : restoredValues)
      graph->replaceErased(erased, restored);
}

void ChangeTarget::changeFormat(PrecisionChange &change,
                                const SetVector<Instruction *> &insts,
                                ArrayRef<Instruction *> sorted) {
  MapVector<Value *, Value *> oldToNew;
  for (Instruction *I : sorted)
    changePrecision(I, change, oldToNew);

  for (auto &KV : oldToNew) {
    auto *oldI = cast<Instruction>(KV.first);
    Value *newV = KV.second;
    if (Value *casted = restoreOutside(
            oldI, [&](Instruction *U) { return insts.contains(U); },
            [&](IRBuilder<> &B) {
              return B.CreateFPCast(
                  newV, storageTypeOf(B.getContext(), change.oldType));
            })) {
      if (VMap) {
        assert(VMap->count(clonedToOriginal[oldI]) &&
               "a changed clone operation maps back to an operation");
        (*VMap)[clonedToOriginal[oldI]] = casted;
      } else if (graph) {
        graph->replaceErased(oldI, casted);
      }
    }
    if (!oldI->use_empty())
      oldI->replaceAllUsesWith(PoisonValue::get(oldI->getType()));
    oldI->eraseFromParent();
    if (!VMap)
      subgraph.operations.remove(oldI);
  }
}

// If `VMap` is passed, map `llvm::Value`s in `subgraph` to their cloned
// values and change outputs in VMap to new casted outputs. Otherwise `graph`,
// if passed, learns of each erased operation whose users read a new value.
void PrecisionCandidate::apply(Session &S, Subgraph &subgraph,
                               ValueToValueMapTy *VMap, FPGraph *graph,
                               SmallVectorImpl<Value *> *components) {
  ChangeTarget target(S, subgraph, VMap, graph);
  SmallVector<Value *, 32> pairComponents, widerComponents;
  for (PrecisionChange &change : changes) {
    SetVector<Instruction *> insts = target.operationsOf(change);
    SmallVector<Instruction *, 8> sorted;
    topoSort(insts, sorted);
    if (unsigned n = expansionComponents(change.newType))
      target.expand(n, sorted, n == 2 ? pairComponents : widerComponents);
    else
      target.changeFormat(change, insts, sorted);
  }
  if (components) {
    components->append(pairComponents.begin(), pairComponents.end());
    components->append(widerComponents.begin(), widerComponents.end());
  }
}

// The FP32 value V extends, if V is such an extension.
static Value *floatExtSource(Value *V) {
  if (auto *FE = dyn_cast<FPExtInst>(V))
    if (FE->getOperand(0)->getType()->isFloatTy())
      return FE->getOperand(0);
  return nullptr;
}

// V as a float constant, if it is one exactly.
static Constant *floatConstant(Value *V) {
  if (isa<PoisonValue>(V))
    return PoisonValue::get(Type::getFloatTy(V->getContext()));
  if (isa<UndefValue>(V))
    return UndefValue::get(Type::getFloatTy(V->getContext()));
  auto *CFP = dyn_cast<ConstantFP>(V);
  if (!CFP || !CFP->getType()->isDoubleTy())
    return nullptr;
  APFloat v = CFP->getValueAPF();
  bool losesInfo = false;
  v.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven, &losesInfo);
  if (losesInfo)
    return nullptr;
  return ConstantFP::get(V->getContext(), v);
}

// The FP64 PHIs of F, in order, that the largest such set closes over: every
// incoming is a member, a float extension or a float constant.
static SmallVector<PHINode *, 16> demotablePhis(Function &F) {
  SmallVector<PHINode *, 16> phis;
  SmallPtrSet<Value *, 16> demotable;
  for (BasicBlock &BB : F)
    for (PHINode &P : BB.phis())
      if (P.getType()->isDoubleTy()) {
        phis.push_back(&P);
        demotable.insert(&P);
      }
  bool shrunk = true;
  while (shrunk) {
    shrunk = false;
    for (PHINode *P : phis) {
      if (!demotable.count(P))
        continue;
      if (all_of(P->incoming_values(), [&](Value *IV) {
            return demotable.count(IV) || floatExtSource(IV) ||
                   floatConstant(IV);
          }))
        continue;
      demotable.erase(P);
      shrunk = true;
    }
  }
  SmallVector<PHINode *, 16> demoted;
  for (PHINode *P : phis)
    if (demotable.count(P))
      demoted.push_back(P);
  return demoted;
}

// Demote FP64 PHIs whose every incoming is fpext-from-float or a
// float-representable constant. InstCombine folds fptrunc(fpext(x)) -> x but
// stops at a loop-carried PHI, so an FP32 accumulator the solver chose would
// otherwise keep its fptrunc/fpext roundtrip.
bool demoteFPCastPHIs(Function &F) {
  SmallVector<PHINode *, 16> demoted = demotablePhis(F);
  if (demoted.empty())
    return false;

  Type *FloatTy = Type::getFloatTy(F.getContext());
  DenseMap<PHINode *, PHINode *> floatPhiOf;
  for (PHINode *P : demoted)
    floatPhiOf[P] = PHINode::Create(FloatTy, P->getNumIncomingValues(),
                                    P->getName() + ".f32", P->getIterator());
  for (PHINode *P : demoted) {
    PHINode *NP = floatPhiOf[P];
    for (unsigned i = 0; i < P->getNumIncomingValues(); ++i) {
      Value *IV = P->getIncomingValue(i);
      Value *fv = nullptr;
      if (auto *IVP = dyn_cast<PHINode>(IV))
        fv = floatPhiOf.lookup(IVP);
      if (!fv)
        fv = floatExtSource(IV);
      if (!fv)
        fv = floatConstant(IV);
      assert(fv && "demoteFPCastPHIs: non-leaf incoming survived closure");
      NP->addIncoming(fv, P->getIncomingBlock(i));
    }
  }
  for (PHINode *P : demoted) {
    PHINode *NP = floatPhiOf[P];
    BasicBlock *BB = NP->getParent();
    IRBuilder<> B = builderAt(BB, BB->getFirstInsertionPt());
    P->replaceAllUsesWith(B.CreateFPExt(NP, Type::getDoubleTy(F.getContext())));
  }
  for (PHINode *P : demoted)
    P->eraseFromParent();
  return true;
}

//===----------------------------------------------------------------------===//
// Accuracy of a candidate
//===----------------------------------------------------------------------===//

namespace {

// A candidate's mean error at each output over the samples, or that it
// returns a non-finite value where the exact result is finite.
struct CandidateErrors {
  struct Mean {
    double sum = 0.0;
    unsigned count = 0;
  };
  bool nonFinite = false;
  std::unordered_map<FPNode *, Mean> at;
};

} // namespace

// The errors of `candidate` at `outputs`, whose exact values at `points` are
// `refs`.
static CandidateErrors
scoreCandidate(const PrecisionCandidate &candidate, ArrayRef<FPNode *> outputs,
               ArrayRef<const SubgraphSamples::Reference *> refs,
               ArrayRef<SubgraphSamples::Point> points) {
  CandidateErrors out;
  for (FPNode *node : outputs)
    out.at[node] = CandidateErrors::Mean();
  SmallVector<double, 8> results;
  for (size_t i = 0; i < points.size(); ++i) {
    getFPValues(outputs, points[i], results, candidate.changes);
    for (size_t j = 0; j < outputs.size(); ++j) {
      double goldVal = refs[j]->gold[i];
      double result = results[j];
      if (std::isfinite(goldVal) && !std::isfinite(result)) {
        out.nonFinite = true;
        return out;
      }
      double error = sampleError(goldVal, result);
      if (!std::isnan(error)) {
        out.at[outputs[j]].sum += error;
        ++out.at[outputs[j]].count;
      }
    }
  }
  return out;
}

// Sets the accuracy cost of PU and of each candidate, the gradient-weighted
// mean error at the outputs over the subgraph's samples, and drops the
// candidates that cannot be scored. With no usable sample the unit keeps the
// subgraph as written.
static void priceAccuracy(const Options &O, PrecisionUnit &PU, FPGraph &G) {
  SubgraphSamples &samples = samplesOf(O, *PU.subgraph, G);
  SmallVector<FPNode *, 4> outputs;
  SmallVector<const FPNode *, 4> nodes;
  SmallVector<double, 4> grads;
  for (Instruction *output : PU.subgraph->outputs) {
    FPNode *node = G.valueToNodeMap[output];
    outputs.push_back(node);
    nodes.push_back(node);
    grads.push_back(node->grad);
  }
  std::optional<BaselineAccuracy> baseline =
      PU.priceBaseline(samples, nodes, grads, O.Diagnostics.Print);
  if (!baseline)
    return;
  ArrayRef<SubgraphSamples::Point> sampledPoints = baseline->points;
  SmallVector<const SubgraphSamples::Reference *, 4> refs;
  for (auto [node, out] : zip(outputs, baseline->outputs)) {
    PU.perOutputInitialAccCost[node] = out.error * std::fabs(out.grad);
    refs.push_back(out.ref);
  }

  SmallVector<CandidateErrors, 0> scored(PU.candidates.size());
  forEachIndex(O.Model.Threads, PU.candidates.size(), [&](size_t c) {
    scored[c] = scoreCandidate(PU.candidates[c], outputs, refs, sampledPoints);
  });

  SmallVector<PrecisionCandidate, 4> kept;
  for (auto [candidate, errors] : zip(PU.candidates, scored)) {
    if (errors.nonFinite) {
      PU.dropped.push_back({candidate.desc,
                            "the accuracy model found that it returns a "
                            "non-finite value at a sampled input where the "
                            "exact result is finite"});
      continue;
    }
    candidate.accuracyCost = 0.0;
    for (FPNode *node : outputs) {
      const CandidateErrors::Mean &mean = errors.at[node];
      assert(mean.count != 0 && "No valid sample found for candidate subgraph");
      candidate.perOutputAccCost[node] =
          mean.sum / mean.count * std::fabs(node->grad);
      candidate.accuracyCost += candidate.perOutputAccCost[node];
    }
    assert(!std::isnan(candidate.accuracyCost) &&
           "a kept candidate has an accuracy cost");
    kept.push_back(std::move(candidate));
  }
  PU.candidates = std::move(kept);
}

//===----------------------------------------------------------------------===//
// Cost of a candidate
//===----------------------------------------------------------------------===//

// Whether an operation is priced at its own measured frequency. A merged unit
// can span a loop body and code that runs once per thread, which one
// execution count misprices, so merging shared-staging subgraphs implies it.
static bool freqWeightedPricing(const Options &O) {
  return O.Model.PrecisionFrequencyWeighted ||
         O.Candidates.SubgraphMergeStaging;
}

namespace {

// A clone of the site with a candidate applied to the operations of a
// subgraph and simplified as the materializer's output is, with handles on
// what the cost walk starts from. Its IR does not depend on the outputs, so
// one clone prices the candidate over any set of the subgraph's outputs.
class CandidateClone {
public:
  CandidateClone(Session &S, Subgraph &subgraph,
                 ArrayRef<Instruction *> outputs, PrecisionCandidate &pt);
  ~CandidateClone() {
    cloneLI.reset();
    cloneDT.reset();
    FClone->eraseFromParent();
  }

  // Why the clone cannot be priced; empty when it can.
  const std::string &unpriceable() const { return unpricedReason; }

  // The cost of the cone of outputs[i] for each i in `which`, in that order,
  // and of the expansion roots. Only for a clone that can be priced.
  Expected<double> cost(Session &S, const Subgraph &subgraph,
                        ArrayRef<size_t> which);

private:
  // Applies `pt` to the clone with the materializer's clean-ups, and holds
  // the walk's inputs and roots: the outputs, then the components the expansion
  // changes leave, which are the unit's outputs in the emitted code.
  void applyCandidate(Session &S, Subgraph &subgraph,
                      ArrayRef<Instruction *> outputs, PrecisionCandidate &pt,
                      Function &F, ValueToValueMapTy &VMap);
  // Simplifies the clone as the target's pipeline does, keeping each handle
  // on its value's copy.
  void simplify(Session &S);
  // Tags each handle's instruction, and moves a handle the last pass left on
  // a constant to the copy of its tag.
  void retrackHandles();
  // Why no cost walk from a root reaches its work; empty when one does.
  std::string lostRoot(ArrayRef<Instruction *> outputs);
  // The instruction root i resolves to, if any.
  Instruction *rootAt(size_t i) const;
  // The instructions the walk from the roots in `roots` reaches before an
  // input, in the order it reaches them.
  SmallVector<Instruction *, 32> cone(ArrayRef<size_t> roots) const;

  Function *FClone = nullptr;
  SmallVector<WeakTrackingVH, 8> trackedInputs;
  // A handle per output, then one per expansion root; each with the FP32
  // operand of the extension it tracks, should the extension fold away.
  SmallVector<WeakTrackingVH, 8> trackedOutputs;
  SmallVector<WeakTrackingVH, 8> trackedOutputFallbacks;
  size_t numOutputs = 0;
  // The metadata kind of the tags, and the tag of each handle's instruction.
  unsigned tagKind = 0;
  SmallVector<MDNode *, 16> inputTags, rootTags, fallbackTags;
  std::optional<DominatorTree> cloneDT;
  std::optional<LoopInfo> cloneLI;
  // The cost-model price of each instruction walked so far.
  DenseMap<const Instruction *, double> prices;
  std::string unpricedReason;
};

} // namespace

// Whether a simplification left the handle on a constant.
static bool lost(const WeakTrackingVH &VH) {
  return VH.pointsToAliveValue() && !isa<Instruction>((Value *)VH);
}

CandidateClone::CandidateClone(Session &S, Subgraph &subgraph,
                               ArrayRef<Instruction *> outputs,
                               PrecisionCandidate &pt) {
  assert(!outputs.empty() && "a candidate is priced over some output");
  Function *F = outputs[0]->getFunction();
  ValueToValueMapTy VMap;
  FClone = CloneFunction(F, VMap);
  FClone->setName(F->getName() + "_clone");
  applyCandidate(S, subgraph, outputs, pt, *F, VMap);
  simplify(S);
  unpricedReason = lostRoot(outputs);
  cloneDT.emplace(*FClone);
  cloneLI.emplace(*cloneDT);
}

void CandidateClone::applyCandidate(Session &S, Subgraph &subgraph,
                                    ArrayRef<Instruction *> outputs,
                                    PrecisionCandidate &pt, Function &F,
                                    ValueToValueMapTy &VMap) {
  const Options &O = S.Opts;
  // The clone gets the materializer's carried-PHI elimination, so the df64
  // restore is not billed per iteration.
  SmallVector<Value *, 32> expandedComponents;
  bool ptHasExpansion = false, wantsPairs = false;
  for (PrecisionChange &ch : pt.changes) {
    ptHasExpansion |= expansionComponents(ch.newType) != 0;
    wantsPairs |= ch.newType == NumberFormat::Expansion2;
  }
  pt.apply(S, subgraph, &VMap, /*graph=*/nullptr, &expandedComponents);
  // An Expansion2 change that converts nothing (unsupported instructions)
  // leaves no components, and the untouched FP64 outputs are still the right
  // roots.
  const bool expansionMaterialized = !expandedComponents.empty();

  // Run the materializer's cleanups, so that the price charges only the casts
  // the materializer emits. FClone has no call sites, so F, whose arguments are
  // FClone's, supplies the parameter-to-global mapping. A df64 candidate wants
  // the pair layout and any other the halved one, so one parameter arm runs.
  const ParamNarrowing paramArm =
      wantsPairs ? ParamNarrowing::Pairs : ParamNarrowing::FP32;
  // The df64 narrowing turns a staged load's split, some of whose halves are
  // components, into a pair load: the components follow it through handles.
  SmallVector<WeakTrackingVH, 32> componentHandles(expandedComponents.begin(),
                                                   expandedComponents.end());
  narrowSharedStaging(S, *FClone, /*announce=*/false, paramArm, F);
  expandedComponents.clear();
  for (Value *L : componentHandles)
    if (L)
      expandedComponents.push_back(L);
  // Cancel the join/split roundtrip, as the emitted code does.
  SmallVector<Value *, 8> dsFoldRoots;
  foldExpansionRoundtrip(*FClone, &dsFoldRoots, O.Diagnostics.Print);

  for (Value *input : subgraph.inputs)
    if (VMap.count(input))
      trackedInputs.emplace_back(VMap[input]);
  // An output without a clone is held by an empty handle, which roots nothing.
  for (Instruction *output : outputs)
    trackedOutputs.emplace_back(VMap.count(output) ? VMap[output]
                                                   : WeakTrackingVH());
  numOutputs = trackedOutputs.size();
  // The F64 handles above are dead once the expansion is materialized, so
  // without the components the candidate walks an empty cone and prices as
  // free.
  size_t liveExpansionRoots = 0;
  for (ArrayRef<Value *> componentList :
       {ArrayRef<Value *>(expandedComponents), ArrayRef<Value *>(dsFoldRoots)})
    for (Value *L : componentList)
      if (auto *LI = dyn_cast<Instruction>(L))
        if (LI->getParent()) {
          trackedOutputs.emplace_back(L);
          ++liveExpansionRoots;
        }
  if (ptHasExpansion && expansionMaterialized && liveExpansionRoots == 0)
    report_fatal_error(
        "getCompCost: expansion candidate materialized df64 values but left "
        "no live root for the cost walk, which would price it as free. "
        "Refusing to fabricate a cost.");
  if (O.Diagnostics.Print && ptHasExpansion && !expansionMaterialized)
    llvm::errs() << "[poseidon] the expansion change converted no "
                    "instruction; pricing the unchanged cone\n";

  // Runs after the value handles are seeded: the demote RAUWs each PHI with
  // fpext(new PHI) before erasing it, so a tracked output migrates to the
  // boundary ext.
  demoteFPCastPHIs(*FClone);

  // Once the accumulator roundtrips fold, a tracked output ext can be DCE'd
  // without a RAUW and the handle nulls; keep its FP32 operand as a fallback
  // root so the accumulation cone stays priced.
  trackedOutputFallbacks.resize(trackedOutputs.size());
  for (size_t i = 0; i < trackedOutputs.size(); ++i)
    if (trackedOutputs[i].pointsToAliveValue())
      if (auto *FE = dyn_cast_or_null<FPExtInst>((Value *)trackedOutputs[i]))
        trackedOutputFallbacks[i] = FE->getOperand(0);
}

void CandidateClone::simplify(Session &S) {
  // On a target without divergent branches the simplification can unswitch
  // a loop on a condition it does not vary in, and delete the body of the
  // version that never runs it: a handle on that body then holds poison, and
  // the walk would price an empty cone. The version that runs the body copies
  // its instructions with their metadata, so after each pass a handle left on
  // a constant moves to the one copy of its tag, where later passes'
  // replacements carry it.
  tagKind = FClone->getContext().getMDKindID("poseidon.cost.handle");
  inputTags.assign(trackedInputs.size(), nullptr);
  rootTags.assign(trackedOutputs.size(), nullptr);
  fallbackTags.assign(trackedOutputs.size(), nullptr);
  retrackHandles();
  simplifyFunction(*FClone, S.PB, OptimizationLevel::O3,
                   [&] { retrackHandles(); });
}

void CandidateClone::retrackHandles() {
  auto tag = [&](WeakTrackingVH &VH, MDNode *&Tag) {
    auto *I = dyn_cast_or_null<Instruction>((Value *)VH);
    if (!I)
      return;
    if (MDNode *Has = I->getMetadata(tagKind)) {
      Tag = Has;
      return;
    }
    if (!Tag)
      Tag = MDNode::getDistinct(FClone->getContext(), {});
    I->setMetadata(tagKind, Tag);
  };
  for (auto [VH, Tag] : zip(trackedInputs, inputTags))
    tag(VH, Tag);
  for (auto [VH, Tag] : zip(trackedOutputs, rootTags))
    tag(VH, Tag);
  for (auto [VH, Tag] : zip(trackedOutputFallbacks, fallbackTags))
    tag(VH, Tag);
  if (none_of(trackedInputs, lost) && none_of(trackedOutputs, lost) &&
      none_of(trackedOutputFallbacks, lost))
    return;

  // The walk stops at every copy of an input, but a root moves only to a
  // single copy: two are two loop versions, of which a thread runs one.
  DenseMap<MDNode *, SmallVector<Instruction *, 1>> copies;
  for (Instruction &I : instructions(*FClone))
    if (MDNode *Tag = I.getMetadata(tagKind))
      copies[Tag].push_back(&I);
  for (size_t j = 0, e = trackedInputs.size(); j < e; ++j)
    if (inputTags[j] && lost(trackedInputs[j]))
      for (auto [k, I] : enumerate(copies.lookup(inputTags[j]))) {
        if (k == 0) {
          trackedInputs[j] = I;
          continue;
        }
        trackedInputs.emplace_back(I);
        inputTags.push_back(inputTags[j]);
      }
  auto moveRoot = [&](WeakTrackingVH &VH, MDNode *Tag) {
    if (!Tag || !lost(VH))
      return;
    SmallVector<Instruction *, 1> found = copies.lookup(Tag);
    if (found.size() == 1)
      VH = found.front();
  };
  for (size_t i = 0; i < trackedOutputs.size(); ++i) {
    moveRoot(trackedOutputs[i], rootTags[i]);
    moveRoot(trackedOutputFallbacks[i], fallbackTags[i]);
  }
}

std::string CandidateClone::lostRoot(ArrayRef<Instruction *> outputs) {
  // A root left on a constant has no single copy: its cost is unknown.
  for (size_t i = 0; i < trackedOutputs.size(); ++i) {
    if (isa_and_nonnull<Instruction>((Value *)trackedOutputs[i]) ||
        isa_and_nonnull<Instruction>((Value *)trackedOutputFallbacks[i]) ||
        (!lost(trackedOutputs[i]) && !lost(trackedOutputFallbacks[i])))
      continue;
    std::string what = "an expansion root of the candidate";
    if (i < outputs.size()) {
      std::string loc = getSourceLocationStr(outputs[i]);
      what = "the output " + (loc.empty() ? getValueStr(outputs[i]) : loc);
    }
    size_t n = 0;
    for (Instruction &I : instructions(*FClone))
      if (MDNode *Tag = I.getMetadata(tagKind))
        n += Tag == rootTags[i] || Tag == fallbackTags[i];
    return "the simplified clone of the site replaced " + what +
           " by a constant while it was still used, and has " +
           std::to_string(n) + " copies of it instead of one";
  }
  // A fold can also rebuild a root's computation in new instructions and
  // erase the root unreplaced, as narrowing fptrunc(op(fpext a, fpext b)) to
  // op(a, b) does. With every root gone so, the work is still there but no
  // handle reaches it.
  if (none_of(seq<size_t>(0, trackedOutputs.size()),
              [&](size_t i) { return rootAt(i); }))
    return "the simplified clone of the site erased every output of the "
           "candidate while computing it in other instructions";
  return "";
}

Instruction *CandidateClone::rootAt(size_t i) const {
  if (auto *I = dyn_cast_or_null<Instruction>((Value *)trackedOutputs[i]))
    return I;
  return dyn_cast_or_null<Instruction>((Value *)trackedOutputFallbacks[i]);
}

SmallVector<Instruction *, 32>
CandidateClone::cone(ArrayRef<size_t> roots) const {
  SmallPtrSet<Value *, 8> clonedInputs;
  for (const WeakTrackingVH &VH : trackedInputs)
    if (VH.pointsToAliveValue() && (Value *)VH)
      clonedInputs.insert(VH);

  SmallSetVector<Value *, 8> clonedOutputs;
  for (size_t i : roots)
    if (Instruction *I = rootAt(i))
      clonedOutputs.insert(I);
  // price() kept only clones with a root; a subset of outputs left none.
  if (clonedOutputs.empty())
    report_fatal_error("Poseidon: none of the outputs a precision candidate is "
                       "priced over is left in its simplified clone. Refusing "
                       "to fabricate a cost.");

  return backwardCone(clonedOutputs.getArrayRef(),
                      [&](Value *V) { return clonedInputs.contains(V); });
}

Expected<double> CandidateClone::cost(Session &S, const Subgraph &subgraph,
                                      ArrayRef<size_t> which) {
  assert(unpricedReason.empty() && "the clone cannot be priced");
  const Options &O = S.Opts;

  // In output order: the walk sums the cost in the order it reaches the
  // instructions.
  SmallVector<size_t, 8> roots(which.begin(), which.end());
  for (size_t i = numOutputs; i < trackedOutputs.size(); ++i)
    roots.push_back(i);
  SmallVector<Instruction *, 32> coneInsts = cone(roots);

  // A conversion the simplified clone hoists out of every loop runs at most
  // once per thread per launch, so it is scaled by outBoundaryFreqScale when
  // the cone's other work is in a loop. One left in a loop runs at least once
  // per iteration of it and counts as the walk's other instructions do.
  unsigned bodyDepth = 0;
  for (Instruction *I : coneInsts)
    if (!isa<FPExtInst, FPTruncInst>(I))
      bodyDepth = std::max(bodyDepth, cloneLI->getLoopDepth(I->getParent()));

  const bool freqWeighted =
      freqWeightedPricing(O) && subgraph.depthExec.size() > 1;
  double cost = 0.0;
  for (Instruction *I : coneInsts) {
    auto It = prices.find(I);
    if (It == prices.end()) {
      Expected<double> price = instructionCost(S.CostModel, I);
      if (!price)
        return price.takeError();
      It = prices.try_emplace(I, *price).first;
    }
    double c = It->second;
    unsigned d = cloneLI->getLoopDepth(I->getParent());
    if (freqWeighted) {
      // The measured frequency of this depth subsumes the boundary-cast scale
      // and also reaches the arithmetic emitted outside the body loop, where a
      // df64 boundary conversion lives.
      c *= subgraph.freqWeightAtDepth(d);
    } else if (isa<FPExtInst, FPTruncInst>(I) && bodyDepth > 0 && d == 0) {
      c *= subgraph.outBoundaryFreqScale;
    }
    cost += c;
  }
  return cost;
}

// The cost of `pt` per execution of `subgraph`; a UserError for a cost-model
// row it lacks, any other error saying why the clone cannot be priced.
static Expected<double> priceCandidate(Session &S, Subgraph &subgraph,
                                       PrecisionCandidate &pt) {
  CandidateClone clone(S, subgraph, subgraph.outputs.getArrayRef(), pt);
  if (!clone.unpriceable().empty())
    return createStringError(clone.unpriceable());
  return clone.cost(S, subgraph,
                    to_vector(seq<size_t>(0, subgraph.outputs.size())));
}

//===----------------------------------------------------------------------===//
// Proposing candidates
//===----------------------------------------------------------------------===//

// The percentage steps of the three-tier sweep, and the finest step of the
// two-tier one.
static constexpr int kThreeTierStep = 20;
static constexpr int kMinTwoTierStep = 5;

// The tiers of a candidate, widest first.
static const std::array<NumberFormat, 2> kCanonicalPairs[] = {
    {NumberFormat::FP64, NumberFormat::FP32},
    {NumberFormat::FP64, NumberFormat::Expansion2},
    {NumberFormat::FP64, NumberFormat::FP16},
    {NumberFormat::FP64, NumberFormat::BF16},
    {NumberFormat::Expansion2, NumberFormat::FP32},
    {NumberFormat::Expansion2, NumberFormat::FP16},
    {NumberFormat::Expansion2, NumberFormat::BF16},
    {NumberFormat::FP32, NumberFormat::FP16},
    {NumberFormat::FP32, NumberFormat::BF16},
    {NumberFormat::FP64, NumberFormat::Expansion3},
    {NumberFormat::Expansion3, NumberFormat::Expansion2},
    {NumberFormat::Expansion3, NumberFormat::FP32},
    {NumberFormat::FP64, NumberFormat::Expansion4},
    {NumberFormat::Expansion4, NumberFormat::Expansion3},
    {NumberFormat::Expansion4, NumberFormat::Expansion2},
    {NumberFormat::Expansion4, NumberFormat::FP32},
};

static const std::array<NumberFormat, 3> kCanonicalTriples[] = {
    {NumberFormat::FP64, NumberFormat::Expansion2, NumberFormat::FP32},
    {NumberFormat::FP64, NumberFormat::Expansion2, NumberFormat::FP16},
    {NumberFormat::FP64, NumberFormat::Expansion2, NumberFormat::BF16},
    {NumberFormat::FP64, NumberFormat::FP32, NumberFormat::FP16},
    {NumberFormat::FP64, NumberFormat::FP32, NumberFormat::BF16},
    {NumberFormat::Expansion2, NumberFormat::FP32, NumberFormat::FP16},
    {NumberFormat::Expansion2, NumberFormat::FP32, NumberFormat::BF16},
};

// `hwScalar` is the target's scalar types; null off the GPU.
static bool precAllowed(const Options &O, NumberFormat t, bool gpuMode,
                        bool expansionSafe,
                        const std::unordered_set<std::string> *hwScalar) {
  if (unsigned nComp = expansionComponents(t)) {
    if (!gpuMode || !expansionSafe)
      return false;
    if (!O.Candidates.PrecisionExpansions)
      return false;
    if (nComp > O.Candidates.PrecisionExpansionComponents)
      return false;
  }
  if (t == NumberFormat::FP16 && (!gpuMode || !hwScalar->count("half")))
    return false;
  if (t == NumberFormat::BF16 && (!gpuMode || !hwScalar->count("bf16")))
    return false;
  return true;
}

// Whether every tier of a tier pair or triple is allowed.
static bool tiersAllowed(const Options &O, ArrayRef<NumberFormat> tiers,
                         bool gpuMode, bool expansionSafe,
                         const std::unordered_set<std::string> *hwScalar) {
  return all_of(tiers, [&](NumberFormat t) {
    return precAllowed(O, t, gpuMode, expansionSafe, hwScalar);
  });
}

// The tiers of a candidate with the percentage of the operations each takes,
// as its description lists them: "FP64(20%) + FP32(80%)".
static std::string fmtTiers(ArrayRef<std::pair<NumberFormat, int>> tierPcts) {
  std::string s;
  for (const auto &[t, pct] : tierPcts) {
    if (!s.empty())
      s += " + ";
    s += formatOf(t).Name.str() + "(" + std::to_string(pct) + "%)";
  }
  return s;
}

// Records each operation's measured frequency in `subgraph`, so the baseline
// walk and every candidate's weight each instruction by its own, snapped onto
// the classes of loop depth the candidate walk sees.
static void measureFrequencies(const Options &O, Function &F,
                               Subgraph &subgraph, const FPGraph &G) {
  DominatorTree freqDT(F);
  LoopInfo freqLI(freqDT);
  auto note = [&](Instruction *I) {
    auto it = G.valueToNodeMap.find(I);
    if (it == G.valueToNodeMap.end() || !it->second ||
        it->second->executions == 0)
      return;
    uint64_t e = it->second->executions;
    subgraph.opExec[I] = e;
    unsigned d = freqLI.getLoopDepth(I->getParent());
    auto dit = subgraph.depthExec.find(d);
    if (dit == subgraph.depthExec.end())
      subgraph.depthExec[d] = e;
    else
      dit->second = std::max(dit->second, e);
  };
  for (Instruction *op : subgraph.operations)
    note(op);
  for (Instruction *output : subgraph.outputs)
    note(output);
  // A materialized instruction keeps only its loop depth.
  for (auto &kv : subgraph.opExec)
    if (auto *I = dyn_cast<Instruction>(kv.first)) {
      auto dit = subgraph.depthExec.find(freqLI.getLoopDepth(I->getParent()));
      if (dit != subgraph.depthExec.end())
        kv.second = dit->second;
    }
  if (O.Diagnostics.Print && subgraph.depthExec.size() > 1) {
    llvm::errs() << "[poseidon] frequency-heterogeneous subgraph "
                    "(normalizer "
                 << subgraph.execNormalizer << ", measured depth:exec";
    for (auto &kv : subgraph.depthExec)
      llvm::errs() << ' ' << kv.first << ':' << kv.second;
    llvm::errs() << "); pricing each instruction at its own measured "
                    "frequency\n";
  }
}

// How often an output-boundary conversion executes per execution of a body
// that runs `executions` times: at most once per thread per launch, so 1 for
// an elementwise subgraph.
static std::optional<double>
outBoundaryFreqScale(const SiteProfileHeader &profileHeader,
                     uint64_t executions) {
  if (executions == 0 || profileHeader.launchCount == 0)
    return std::nullopt;
  double threads = 1.0;
  for (uint32_t dim : profileHeader.maxBlockDim)
    threads *= std::max<uint32_t>(1, dim);
  for (uint32_t dim : profileHeader.maxGridDim)
    threads *= std::max<uint32_t>(1, dim);
  return std::min(1.0, threads * (double)profileHeader.launchCount /
                           (double)executions);
}

namespace {

// The candidates of one precision unit: its operations split into tiers by
// profiled sensitivity, the most sensitive in the widest tier.
class TierSweep {
public:
  TierSweep(const Options &O, PrecisionUnit &PU, NumberFormat current)
      : O(O), PU(PU), current(current) {}

  // Two tiers, the share of the wider from 0% up in steps of
  // -poseidon-precision-tier-step, over `sortedAsc` (ascending sensitivity).
  void twoTier(ArrayRef<NumberFormat> tiers, ArrayRef<FPLLValue *> sortedAsc,
               StringRef label);
  // Three tiers, the shares in steps of kThreeTierStep.
  void threeTier(ArrayRef<NumberFormat> tiers, ArrayRef<FPLLValue *> sortedAsc,
                 StringRef label);

private:
  // The description of a candidate of `label` (All or Funcs) with `tiers`,
  // announced under -poseidon-print.
  std::string describe(StringRef label, StringRef kind,
                       const std::string &tiers, size_t N) const;
  // Adds the candidate that computes the operations of `sortedAsc` from the
  // cumulative share `cuts[i - 1]` to `cuts[i]` (percent, most sensitive
  // first; the last tier ends at 100) in `tiers[i]`, unless it moves nothing.
  void add(ArrayRef<NumberFormat> tiers, ArrayRef<int> cuts,
           ArrayRef<FPLLValue *> sortedAsc, StringRef label, StringRef kind);

  const Options &O;
  PrecisionUnit &PU;
  NumberFormat current;
};

} // namespace

std::string TierSweep::describe(StringRef label, StringRef kind,
                                const std::string &tiers, size_t N) const {
  if (O.Diagnostics.Print)
    llvm::errs() << "[poseidon] Created " << label << " " << kind
                 << " precision candidate: " << tiers << " (N=" << N << ")\n";
  std::string desc = label.str();
  if (!desc.empty())
    desc += " ";
  return desc + tiers;
}

void TierSweep::add(ArrayRef<NumberFormat> tiers, ArrayRef<int> cuts,
                    ArrayRef<FPLLValue *> sortedAsc, StringRef label,
                    StringRef kind) {
  const size_t N = sortedAsc.size();
  SmallVector<std::pair<NumberFormat, int>, 3> shares;
  SmallVector<PrecisionChange, 3> changes;
  int prevPct = 0;
  size_t prevK = 0;
  for (auto [i, type] : enumerate(tiers)) {
    const int pct = i < cuts.size() ? cuts[i] : 100;
    const size_t k = N * pct / 100;
    shares.emplace_back(type, pct - prevPct);
    SetVector<FPLLValue *> ops(sortedAsc.end() - k, sortedAsc.end() - prevK);
    if (type != current && !ops.empty())
      changes.emplace_back(ops, current, type,
                           expansionComponents(type) ? expansionArithmetic
                                                     : nullptr);
    prevPct = pct;
    prevK = k;
  }
  std::string desc = describe(label, kind, fmtTiers(shares), N);
  if (changes.empty())
    return;
  PrecisionCandidate cand{std::move(changes), std::move(desc)};
  cand.ordinal = PU.candidates.size();
  PU.candidates.push_back(std::move(cand));
}

void TierSweep::twoTier(ArrayRef<NumberFormat> tiers,
                        ArrayRef<FPLLValue *> sortedAsc, StringRef label) {
  const size_t N = sortedAsc.size();
  const int step = std::max(kMinTwoTierStep, O.Candidates.PrecisionTierStep);
  size_t prev = N + 1;
  for (int pct = 0; pct <= 100 - step; pct += step) {
    size_t k = N * pct / 100;
    if (k == prev)
      continue;
    prev = k;
    add(tiers, {pct}, sortedAsc, label, "two-tier");
  }
}

void TierSweep::threeTier(ArrayRef<NumberFormat> tiers,
                          ArrayRef<FPLLValue *> sortedAsc, StringRef label) {
  const size_t N = sortedAsc.size();
  const int step = kThreeTierStep;
  for (int pctHi = step; pctHi <= 100 - 2 * step; pctHi += step) {
    for (int pctHiMid = pctHi + step; pctHiMid <= 100 - step;
         pctHiMid += step) {
      size_t k0 = N * pctHi / 100;
      size_t k1 = N * pctHiMid / 100;
      if (k0 == 0 || k0 >= k1 || k1 >= N)
        continue;
      add(tiers, {pctHi, pctHiMid}, sortedAsc, label, "three-tier");
    }
  }
}

namespace {
// What the precision technique keeps for a compile.
struct PrecisionState : TechniqueState {
  static inline char ID;
  // The libm functions whose single-precision call, with its two casts, the
  // cost model prices below the double-precision call; set on first use.
  std::optional<std::unordered_set<std::string>> NarrowableFuncs;
};
} // namespace

// The libm calls that are cheaper computed at FP32 and converted back than at
// FP64 on this device. A model without a row this needs is an error at `F`.
static Expected<const std::unordered_set<std::string> *>
narrowableFuncs(Session &S, const Function &F) {
  std::optional<std::unordered_set<std::string>> &cached =
      S.get<PrecisionState>().NarrowableFuncs;
  if (!cached) {
    std::unordered_set<std::string> funcs;
    for (const MathFunction &libm : mathFunctions()) {
      if (!libm.Libm)
        continue;
      StringRef func = libm.Name;
      const std::pair<StringRef, StringRef> rows[] = {
          {func, "float"},
          {func, "double"},
          {"fptrunc_double_to_float", "double"},
          {"fpext_float_to_double", "float"}};
      double costs[4];
      for (auto [cost, row] : zip(costs, rows)) {
        Expected<double> price =
            rowCost(S.CostModel, F, F.getSubprogram(), row.first, row.second);
        if (!price)
          return price.takeError();
        cost = *price;
      }
      auto [costFP32, costFP64, costFPTrunc, costFPExt] = costs;
      double costFPCast = costFPTrunc + costFPExt;
      if (costFP32 + costFPCast < costFP64)
        funcs.insert(func.str());
    }
    cached = std::move(funcs);
  }
  return &*cached;
}

Error proposePrecisionCandidates(
    Session &S, Function &F, Subgraph &subgraph, FPGraph &G,
    const SiteProfileHeader &profileHeader, bool expansionSafe,
    SmallVectorImpl<PrecisionUnit> &precisionUnits) {
  const Options &O = S.Opts;
  const uint64_t o0Exec = G.valueToNodeMap[subgraph.outputs[0]]->executions;

  // Filled before PU is built, whose baseline walk reads them.
  subgraph.opExec.clear();
  subgraph.depthExec.clear();
  subgraph.execNormalizer = o0Exec;
  if (freqWeightedPricing(O))
    measureFrequencies(O, F, subgraph, G);

  PrecisionUnit PU(subgraph);
  if (Error E = PU.priceAsWritten(S.CostModel))
    return E;
  PU.executions = o0Exec;
  if (std::optional<double> scale =
          outBoundaryFreqScale(profileHeader, PU.executions))
    subgraph.outBoundaryFreqScale = *scale;

  // The operations, and the calls of the library functions the cost model
  // prices per precision, by ascending sensitivity. Stable: which tier an
  // operation of a tied sensitivity lands in follows the subgraph's order.
  Expected<const std::unordered_set<std::string> *> funcs =
      narrowableFuncs(S, F);
  if (!funcs)
    return funcs.takeError();
  const std::unordered_set<std::string> &NarrowableFuncs = **funcs;
  SetVector<FPLLValue *> funcsSet, allSet;
  for (Instruction *I : subgraph.operations) {
    auto *node = cast<FPLLValue>(G.valueToNodeMap[I]);
    allSet.insert(node);
    if (NarrowableFuncs.count(operationName(*node)) != 0) {
      funcsSet.insert(node);
      if (O.Diagnostics.Print)
        llvm::errs() << "[poseidon] library call with a precision variant: "
                     << *I << '\n';
    }
  }
  SmallVector<FPLLValue *> sortedFuncs(funcsSet.begin(), funcsSet.end());
  SmallVector<FPLLValue *> sortedAllOps(allSet.begin(), allSet.end());
  auto bySens = [](const FPLLValue *a, const FPLLValue *b) {
    return a->sens < b->sens;
  };
  stable_sort(sortedFuncs, bySens);
  stable_sort(sortedAllOps, bySens);

  const bool gpuMode = Target::of(F).isDevice();
  const std::unordered_set<std::string> *hwScalar = nullptr;
  if (gpuMode) {
    Expected<const std::unordered_set<std::string> *> types =
        scalarTypes(S.CostModel, F);
    if (!types)
      return types.takeError();
    hwScalar = *types;
  }
  TierSweep sweep(O, PU, formatOf(subgraph.outputs[0]->getType())->Kind);
  for (ArrayRef<NumberFormat> tp : kCanonicalPairs) {
    if (!tiersAllowed(O, tp, gpuMode, expansionSafe, hwScalar))
      continue;
    sweep.twoTier(tp, sortedAllOps, "All");
    if (!sortedFuncs.empty())
      sweep.twoTier(tp, sortedFuncs, "Funcs");
  }
  if (O.Candidates.PrecisionThreeTier)
    for (ArrayRef<NumberFormat> tr : kCanonicalTriples)
      if (tiersAllowed(O, tr, gpuMode, expansionSafe, hwScalar))
        sweep.threeTier(tr, sortedAllOps, "All");

  // Zero candidates for a nonempty subgraph means every tier was filtered
  // out (a configuration problem), which must not pass as a no-op frontier.
  if (PU.candidates.empty() && !subgraph.operations.empty()) {
    std::string msg =
        ("a floating-point subgraph of " + Twine(subgraph.operations.size()) +
         " operations has no precision candidate, since the "
         "cost model allows no tier combination; check its "
         "scalar_types header")
            .str();
    if (O.Model.StrictMode)
      return userError(F, "strict mode: " + msg);
    diagnose(F, msg, DS_Warning);
  }

  precisionUnits.push_back(std::move(PU));
  return Error::success();
}

Error PrecisionUnit::priceAsWritten(CostModelState &CM) {
  Expected<double> cost = compCost(
      CM,
      SmallVector<Value *>(subgraph->outputs.begin(), subgraph->outputs.end()),
      subgraph->inputs, &subgraph->opExec, subgraph->execNormalizer);
  if (!cost)
    return cost.takeError();
  initialCompCost = *cost;
  return Error::success();
}

double PrecisionUnit::getCompCostDelta(size_t candidateIndex) {
  return (candidates[candidateIndex].CompCost - initialCompCost) *
         (double)executions;
}

// The units among `steps` that rewrite outputs of `subgraph`, in step order.
static SmallVector<const SiteUnit *> rewritesOf(const Subgraph *subgraph,
                                                ArrayRef<SolutionStep> steps) {
  SmallVector<const SiteUnit *> rewrites;
  for (const SolutionStep &step : steps) {
    const SiteUnit &unit = unitOf(step);
    if (unit.subgraphOf() == subgraph && !unit.rewrittenOutputs().empty())
      rewrites.push_back(&unit);
  }
  return rewrites;
}

// What the rewrites of `subgraph` among `steps` leave of it.
static Subgraph withoutRewrites(const Subgraph &subgraph,
                                ArrayRef<SolutionStep> steps) {
  Subgraph left = subgraph;
  for (const SiteUnit *unit : rewritesOf(&subgraph, steps))
    unit->removeRewritten(left);
  return left;
}

// The operations of `subgraph` each change of `candidate` converts, a null
// after each change's.
static std::vector<const Instruction *>
changedOperations(const PrecisionCandidate &candidate,
                  const Subgraph &subgraph) {
  std::vector<const Instruction *> changed;
  for (const PrecisionChange &change : candidate.changes) {
    for (const FPLLValue *node : change.nodes)
      if (node && node->value &&
          subgraph.operations.contains(cast<Instruction>(node->value)))
        changed.push_back(cast<Instruction>(node->value));
    changed.push_back(nullptr);
  }
  return changed;
}

PrecisionUnit::RepriceKey
PrecisionUnit::repriceKey(size_t candidateIndex,
                          ArrayRef<SolutionStep> steps) const {
  SmallVector<const SiteUnit *> rewrites = rewritesOf(subgraph, steps);
  return RepriceKey{candidateIndex, {rewrites.begin(), rewrites.end()}};
}

void PrecisionUnit::priceRequests(
    Session &S, ArrayRef<std::pair<size_t, ArrayRef<SolutionStep>>> requests) {
  // A candidate's clone depends only on the operations it changes, so the
  // requests that leave a candidate the same operations share one clone and
  // differ in the outputs its cost walk starts from.
  struct Request {
    RepriceKey key;
    SmallVector<Value *> outputs;
    SmallVector<size_t, 8> which;
  };
  struct Group {
    size_t candidateIndex;
    Subgraph subgraph;
    SmallVector<Request, 1> requests;
  };
  SmallVector<Group, 0> groups;
  std::map<std::pair<size_t, std::vector<const Instruction *>>, size_t> groupOf;
  std::set<RepriceKey> requested;
  DenseMap<const Instruction *, size_t> outputIndex;
  for (auto [i, output] : enumerate(subgraph->outputs))
    outputIndex[output] = i;

  for (auto [candidateIndex, steps] : requests) {
    RepriceKey key = repriceKey(candidateIndex, steps);
    if (compCostDeltaCache.count(key) || !requested.insert(key).second)
      continue;

    Subgraph newSubgraph = withoutRewrites(*subgraph, steps);
    if (newSubgraph.outputs.empty()) {
      compCostDeltaCache[key] = 0;
      continue;
    }

    auto [it, fresh] = groupOf.try_emplace(
        {candidateIndex,
         changedOperations(candidates[candidateIndex], newSubgraph)},
        groups.size());
    if (fresh)
      groups.push_back({candidateIndex, newSubgraph, {}});
    Request &R = groups[it->second].requests.emplace_back();
    R.key = std::move(key);
    R.outputs.assign(newSubgraph.outputs.begin(), newSubgraph.outputs.end());
    for (Instruction *output : newSubgraph.outputs)
      R.which.push_back(outputIndex.lookup(output));
  }

  // What the outputs left cost as written, shared by every candidate.
  std::map<SmallVector<Value *>, double> initialCompCosts;
  for (Group &G : groups) {
    CandidateClone clone(S, G.subgraph, subgraph->outputs.getArrayRef(),
                         candidates[G.candidateIndex]);
    // price() kept only candidates priced over the whole subgraph.
    if (!clone.unpriceable().empty())
      report_fatal_error("Poseidon: precision candidate '" +
                         Twine(candidates[G.candidateIndex].desc) +
                         "' is priced over its whole subgraph but not over "
                         "what a rewrite leaves of it: " +
                         clone.unpriceable() +
                         ". Refusing to fabricate a cost.");
    for (Request &R : G.requests) {
      // The solve cannot unwind: a cost it cannot compute is taken as 0 and
      // its error kept for the caller of the solve.
      auto [initial, fresh] = initialCompCosts.try_emplace(R.outputs);
      if (fresh) {
        Expected<double> cost =
            compCost(S.CostModel, R.outputs, subgraph->inputs,
                     &subgraph->opExec, subgraph->execNormalizer);
        if (cost) {
          initial->second = *cost;
        } else {
          pricingError.set(cost.takeError());
          initial->second = 0.0;
        }
      }
      double initialCompCost = initial->second;
      Expected<double> clonePrice = clone.cost(S, *subgraph, R.which);
      if (!clonePrice) {
        pricingError.set(clonePrice.takeError());
        compCostDeltaCache[R.key] = 0.0;
        continue;
      }
      double candidateCompCost = *clonePrice;
      double adjustedCostDelta =
          (candidateCompCost - initialCompCost) * (double)executions;
      // With a one-point frontier these lines record why each candidate lost.
      if (S.Opts.Diagnostics.Print)
        llvm::errs() << "[poseidon] precision candidate '"
                     << candidates[G.candidateIndex].desc
                     << "': initial=" << initialCompCost
                     << " cand=" << candidateCompCost << " exec=" << executions
                     << " dcost=" << adjustedCostDelta
                     << " dacc=" << getAccCostDelta(G.candidateIndex) << '\n';
      compCostDeltaCache[R.key] = adjustedCostDelta;
    }
  }
}

void PrecisionUnit::priceAhead(ArrayRef<ArrayRef<SolutionStep>> priors) {
  SmallVector<std::pair<size_t, ArrayRef<SolutionStep>>, 0> requests;
  for (ArrayRef<SolutionStep> steps : priors)
    for (size_t i = 0; i < candidates.size(); ++i)
      requests.push_back({i, steps});
  priceRequests(session(), requests);
}

double PrecisionUnit::costDelta(size_t i, ArrayRef<SolutionStep> prior) {
  priceRequests(session(), {{i, prior}});
  return compCostDeltaCache.at(repriceKey(i, prior));
}

double PrecisionUnit::errorDelta(size_t candidateIndex,
                                 ArrayRef<SolutionStep> steps) {
  auto &valueToNodeMap = graph().valueToNodeMap;
  RepriceKey key = repriceKey(candidateIndex, steps);
  if (auto cacheIt = accCostDeltaCache.find(key);
      cacheIt != accCostDeltaCache.end())
    return cacheIt->second;

  double totalCandidateAccCost = 0.0;
  double totalInitialAccCost = 0.0;

  // The outputs the rewrites among `steps` replace are not this unit's.
  SmallPtrSet<FPNode *, 8> stepNodes;
  for (const SiteUnit *unit : rewritesOf(subgraph, steps))
    for (Value *root : unit->rewrittenOutputs()) {
      auto it = valueToNodeMap.find(root);
      assert(it != valueToNodeMap.end() && it->second &&
             "a rewritten output has a node");
      stepNodes.insert(it->second);
    }

  for (auto &[node, cost] : perOutputInitialAccCost)
    if (!stepNodes.count(node))
      totalInitialAccCost += cost;
  for (auto &[node, cost] : candidates[candidateIndex].perOutputAccCost)
    if (!stepNodes.count(node))
      totalCandidateAccCost += cost;

  double adjustedAccCostDelta = totalCandidateAccCost - totalInitialAccCost;

  accCostDeltaCache[key] = adjustedAccCostDelta;
  return adjustedAccCostDelta;
}

bool PrecisionUnit::pricedAgainst(const OptimizationUnit &earlier) const {
  const SiteUnit &unit = cast<SiteUnit>(earlier);
  return unit.subgraphOf() == subgraph && !unit.rewrittenOutputs().empty();
}

static constexpr size_t kMaxListedOps = 6;

static std::string precisionWords(NumberFormat t) {
  if (unsigned k = expansionComponents(t))
    return formatv("a {0}-component float expansion", k).str();
  const NumberFormatInfo &format = formatOf(t);
  return (format.Words.empty() ? format.Name : format.Words).str();
}

static std::string opWord(StringRef op) {
  if (op == "+" || op == "-" || op == "*" || op == "/")
    return ("'" + op + "'").str();
  if (op == "neg")
    return "negation";
  return op.str();
}

static const Instruction *
firstInIROrder(Function &F, const SmallPtrSetImpl<const Value *> &set) {
  for (Instruction &I : instructions(F))
    if (set.contains(&I))
      return &I;
  return nullptr;
}

Error PrecisionUnit::explain(Explanation &E, UnitExplanation &U) {
  const Subgraph &sg = *subgraph;
  Function &F = *graph().F;
  auto nodeOf = [&](Value *V) -> FPNode * {
    auto it = graph().valueToNodeMap.find(V);
    return it == graph().valueToNodeMap.end() ? nullptr : it->second;
  };
  CRendering how = E.rendering(sg);
  DenseMap<const FPNode *, unsigned> opIndex;
  unsigned k = 0;
  for (Instruction &I : instructions(F))
    if (sg.operations.contains(&I)) {
      if (FPNode *N = nodeOf(&I))
        opIndex[N] = k;
      ++k;
    }

  auto describe = [&](Alternative &A, PrecisionCandidate &cand) {
    bool expansion = false;
    DenseMap<const FPNode *, std::string> dtypes;
    SmallPtrSet<const Value *, 8> changedInsts;
    SmallVector<std::string, 4> parts;
    for (const PrecisionChange &ch : cand.changes) {
      expansion |= expansionComponents(ch.newType) != 0;
      json::Object c;
      c["from"] = formatOf(ch.oldType).Name.str();
      c["to"] = formatOf(ch.newType).Name.str();
      json::Array idx;
      SmallVector<std::string, 4> names;
      MapVector<std::string, unsigned, std::map<std::string, unsigned>> counts;
      for (FPLLValue *node : ch.nodes) {
        auto it = opIndex.find(node);
        if (it != opIndex.end())
          idx.push_back((int64_t)it->second);
        changedInsts.insert(node->value);
        SourceLoc L = locOf(node->value);
        std::string w = opWord(operationName(*node));
        ++counts[w];
        if (L.valid())
          w += formatv(" ({0}:{1})", L.line, L.col).str();
        names.push_back(w);
        A.changedOps.push_back(w);
        if (!expansionComponents(ch.newType))
          dtypes[node] = dtypeOfType(storageTypeOf(F.getContext(), ch.newType));
      }
      c["ops"] = std::move(idx);
      A.changes.push_back(std::move(c));
      if (names.size() > kMaxListedOps) {
        names.clear();
        for (const auto &[w, n] : counts)
          names.push_back(n == 1 ? w : formatv("{0} x {1}", n, w).str());
      }
      parts.push_back(joinWords(names) + " in " + precisionWords(ch.newType));
      A.changedTo = precisionWords(ch.newType);
    }
    A.kind = expansion ? "expansion" : "precision";
    A.summary = cand.desc + ": " + joinWords(parts) + " (" +
                std::to_string(A.changedOps.size()) + " of " +
                std::to_string(sg.operations.size()) + " operations)";
    if (A.changedOps.size() == sg.operations.size() && cand.changes.size() == 1)
      A.changedOps = {
          formatv("all {0} operations", sg.operations.size()).str()};
    else if (A.changedOps.size() > kMaxListedOps)
      A.changedOps = {formatv("{0} of the {1} operations", A.changedOps.size(),
                              sg.operations.size())
                          .str()};
    A.at = firstInIROrder(F, changedInsts);
    if (!A.at)
      A.at = sg.outputs[0];
    if (!expansion) {
      CRendering typed = how;
      typed.dtypes = &dtypes;
      for (Instruction *O : sg.outputs)
        if (FPNode *N = nodeOf(O))
          A.c.push_back(renderC(*N, typed, dtypeOfType(O->getType())));
    }
    A.baseAcc = initialAccCost;
    A.baseCost = U.baseCost;
  };
  for (size_t j = 0; j < candidates.size(); ++j) {
    PrecisionCandidate &cand = candidates[j];
    Alternative A;
    A.id = candidateId(j);
    A.item = this;
    A.index = j;
    describe(A, cand);
    for (unsigned o = 0; o < sg.outputs.size(); ++o)
      A.outputs.push_back(o);
    std::optional<int64_t> cost;
    if (cand.CompCost != std::numeric_limits<double>::max()) {
      Expected<int64_t> c = E.onAxis(getCompCostDelta(j));
      if (!c)
        return c.takeError();
      cost = *c;
    }
    double acc = getAccCostDelta(j);
    // Applied after rewrites of its subgraph's outputs, the candidate is
    // priced over what they leave.
    RepriceKey key = repriceKey(j, E.appliedSteps());
    if (!key.rewrites.empty() &&
        llvm::any_of(E.appliedSteps(), [&](const SolutionStep &step) {
          return step.unit == this && step.candidateIndex == j;
        })) {
      auto c = compCostDeltaCache.find(key);
      cost = std::nullopt;
      if (c != compCostDeltaCache.end()) {
        Expected<int64_t> repriced = E.onAxis(c->second);
        if (!repriced)
          return repriced.takeError();
        cost = *repriced;
      }
      auto a = accCostDeltaCache.find(key);
      acc = a == accCostDeltaCache.end()
                ? std::numeric_limits<double>::quiet_NaN()
                : a->second;
    }
    A.costDelta = cost;
    E.setAccuracy(A, acc);
    U.alts.push_back(std::move(A));
  }
  for (const auto &[desc, why] : dropped) {
    Alternative A;
    A.kind = "precision";
    A.summary = desc;
    A.at = sg.outputs[0];
    A.standing = Standing::Dropped;
    A.reason = "pricing dropped it: " + why;
    U.alts.push_back(std::move(A));
  }
  if (!unpricedReason.empty())
    U.unpriced(unpricedReason);
  return Error::success();
}

std::string PrecisionUnit::choiceText(size_t i) const {
  const PrecisionCandidate &cand = candidates[i];
  return "precision change " + cand.desc + " (#" +
         std::to_string(cand.ordinal) + ")";
}

Error PrecisionUnit::materialize(size_t i) {
  assert(i < candidates.size() && "the solver names one of the candidates");
  // An operation this change erases may be an input of another subgraph whose
  // rewrite a later step materializes: the rewrite reads the value the
  // operation's users now read, as the units were priced.
  candidates[i].apply(session(), *subgraph, nullptr, &graph());
  return Error::success();
}

Error PrecisionUnit::price() {
  SmallVector<PrecisionCandidate, 8> priced;
  std::string firstWhy;
  for (PrecisionCandidate &c : candidates) {
    Expected<double> cost = priceCandidate(session(), *subgraph, c);
    if (cost) {
      c.CompCost = *cost;
      priced.push_back(std::move(c));
      continue;
    }
    Error E = cost.takeError();
    if (E.isA<UserError>())
      return E;
    std::string why = "its cost cannot be computed: " + toString(std::move(E));
    if (firstWhy.empty())
      firstWhy = why;
    dropped.push_back({c.desc, std::move(why)});
  }
  if (!firstWhy.empty() && session().Opts.Diagnostics.Print)
    llvm::errs() << "[poseidon] "
                 << subgraph->outputs[0]->getFunction()->getName() << ": "
                 << (candidates.size() - priced.size()) << " of "
                 << candidates.size()
                 << " precision candidate(s) dropped, first because "
                 << firstWhy << '\n';
  candidates = std::move(priced);
  priceAccuracy(session().Opts, *this, graph());
  return Error::success();
}

void PrecisionUnit::scaleErrors(double factor) {
  SubgraphUnit::scaleErrors(factor);
  for (auto &[node, cost] : perOutputInitialAccCost)
    cost *= factor;
  for (PrecisionCandidate &cand : candidates)
    for (auto &[node, cost] : cand.perOutputAccCost)
      cost *= factor;
}

void PrecisionUnit::printPriced(raw_ostream &OS) {
  OS << "[poseidon] priced precision candidates\n";
  OS << "Initial AccuracyCost: " << initialAccCost << '\n';
  OS << "Initial ComputationCost: " << initialCompCost << '\n';
  OS << "Candidates:\n";
  OS << "accuracy delta\t\tcost delta\t\tdescription\n"
     << "---------------------------\n";
  for (size_t i = 0; i < candidates.size(); ++i) {
    PrecisionCandidate &candidate = candidates[i];
    OS << getAccCostDelta(i) << "\t\t";
    if (candidate.CompCost == std::numeric_limits<double>::max())
      OS << "unpriced";
    else
      OS << getCompCostDelta(i);
    OS << "\t\t" << candidate.desc << '\n';
  }
}

} // namespace poseidon
