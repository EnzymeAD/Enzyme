//===- Precision.cpp - precision candidates of an FP subgraph -------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
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
#include "Core/Parallel.h"
#include "Expansion.h"
#include "Herbie.h"
#include "IR/Utils.h"
#include "Model/CostModel.h"
#include "Model/Evaluators.h"
#include "Model/Sampling.h"
#include "Pass/Session.h"
#include "Profile/ProfileRead.h"
#include "Sites/Canonicalize.h"
#include "Staging.h"
#include "Targets/MathLibrary.h"
#include "Targets/Target.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMap.h"
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

// Emits I in the precision of `change`, reading the operands already changed
// from `oldToNew` and casting the others, and records the new value there.
static void changePrecision(Instruction *I, PrecisionChange &change,
                            MapVector<Value *, Value *> &oldToNew) {
  assert(isOptimizable(*I) &&
         "Trying to tune an instruction is not isOptimizable");

  IRBuilder<> Builder(I);
  Builder.setFastMathFlags(I->getFastMathFlags());
  Type *newType = getLLVMFPType(change.newType, I->getContext());
  Value *newI = nullptr;

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

  if (isa<UnaryOperator>(I) || isa<BinaryOperator>(I)) {
    SmallVector<Value *, 2> newOps;
    for (Value *operand : I->operands())
      newOps.push_back(convert(operand));
    newI = Builder.CreateNAryOp(I->getOpcode(), newOps);
  } else if (auto *CI = dyn_cast<CallInst>(I)) {
    SmallVector<Value *, 4> newArgs;
    for (Value *arg : CI->args())
      newArgs.push_back(convert(arg));
    auto *calledFunc = CI->getCalledFunction();
    if (calledFunc && calledFunc->isIntrinsic()) {
      Intrinsic::ID intrinsicID = calledFunc->getIntrinsicID();
      if (intrinsicID != Intrinsic::not_intrinsic) {
        if (intrinsicID == Intrinsic::powi) {
          SmallVector<Type *, 2> overloadedTypes;
          overloadedTypes.push_back(newType);
          overloadedTypes.push_back(CI->getArgOperand(1)->getType());
          Function *newFunc = Intrinsic::getOrInsertDeclaration(
              CI->getModule(), intrinsicID, overloadedTypes);
          newI = Builder.CreateCall(newFunc, newArgs);
        } else {
          Function *newFunc = Intrinsic::getOrInsertDeclaration(
              CI->getModule(), intrinsicID, newType);
          newI = Builder.CreateCall(newFunc, newArgs);
        }
      } else {
        llvm::errs() << "PT: Unknown intrinsic: " << *CI << '\n';
        llvm_unreachable("changePrecision: Unknown intrinsic call to change");
      }
    } else {
      assert(calledFunc && "isOptimizable admits only calls of a known callee");
      StringRef funcName = calledFunc->getName();

      // An F32-downcast sqrt becomes the target's approximate square root
      // where it has one, instead of a correctly-rounded library call that
      // would dominate sqrt-heavy kernels; the accuracy model charges the
      // approximation's error.
      {
        Intrinsic::ID approxSqrt = approxSqrtF32Of(*CI);
        if (newType->isFloatTy() && approxSqrt != Intrinsic::not_intrinsic) {
          Function *fastSqrt = Intrinsic::getOrInsertDeclaration(
              CI->getModule(), approxSqrt, {});
          newI = Builder.CreateCall(fastSqrt, {newArgs[0]});
          oldToNew[I] = newI;
          return;
        }
      }

      // fmax and fmin become maxnum and minnum, which have the same IEEE-754
      // semantics and are modelled: libdevice is linked before this pass, so
      // a fresh __nv_fmaxf declaration would reach ptxas undefined.
      {
        StringRef base = mathFunctionOf(funcName);
        if ((base == "fmax" || base == "fmin") && newArgs.size() == 2) {
          Intrinsic::ID id =
              (base == "fmax") ? Intrinsic::maxnum : Intrinsic::minnum;
          newI = Builder.CreateBinaryIntrinsic(id, newArgs[0], newArgs[1]);
          oldToNew[I] = newI;
          return;
        }
      }

      std::string newFuncName = getLibmFunctionForPrecision(funcName, newType);

      if (!newFuncName.empty()) {
        Module *M = CI->getModule();

        Type *funcType = newType;
        bool needsPromotion = newType->isHalfTy() || newType->isBFloatTy();
        if (needsPromotion)
          funcType = Type::getFloatTy(newType->getContext());

        SmallVector<Value *, 4> callArgs;
        for (auto *arg : newArgs) {
          if (needsPromotion && arg->getType()->isFloatingPointTy() &&
              arg->getType() != funcType)
            callArgs.push_back(
                Builder.CreateFPExt(arg, funcType, "poseidon.promote"));
          else
            callArgs.push_back(arg);
        }

        SmallVector<Type *, 4> funcArgTypes(callArgs.size(), funcType);
        FunctionCallee newFuncCallee = M->getOrInsertFunction(
            newFuncName, FunctionType::get(funcType, funcArgTypes, false));

        if (Function *newFunc = dyn_cast<Function>(newFuncCallee.getCallee())) {
          Value *callResult = createMathCall(Builder, newFunc, callArgs);
          if (needsPromotion)
            newI =
                Builder.CreateFPTrunc(callResult, newType, "poseidon.demote");
          else
            newI = callResult;
        } else {
          llvm::errs() << "PT: Failed to get "
                       << getPrecisionChangeTypeString(change.newType)
                       << " libm function for: " << *CI << '\n';
          llvm_unreachable("changePrecision: Failed to get libm function");
        }
      } else {
        llvm::errs() << "PT: Unknown function call: " << *CI << '\n';
        llvm_unreachable("changePrecision: Unknown function call to change");
      }
    }

  } else {
    llvm::errs() << "Unexpectedly isOptimizable instruction: " << *I << '\n';
    llvm_unreachable("Unexpectedly isOptimizable instruction");
  }

  oldToNew[I] = newI;
}

// If `VMap` is passed, map `llvm::Value`s in `subgraph` to their cloned
// values and change outputs in VMap to new casted outputs. Otherwise `graph`,
// if passed, learns of each erased operation whose users read a new value.
void PTCandidate::apply(Session &S, Subgraph &subgraph, ValueToValueMapTy *VMap,
                        FPGraph *graph, SmallVectorImpl<Value *> *limbs) {
  SetVector<Instruction *> operations;
  DenseMap<Value *, Value *> clonedToOriginal;
  if (VMap) {
    for (auto *I : subgraph.operations) {
      assert(VMap->count(I) && "the clone maps every operation");
      if (Value *Mapped = VMap->lookup(I)) {
        if (auto *MappedI = dyn_cast<Instruction>(Mapped)) {
          operations.insert(MappedI);
          clonedToOriginal[MappedI] = I;
        }
      }
    }
  } else {
    operations = subgraph.operations;
  }

  SmallVector<Value *, 32> pairLimbs, widerLimbs;
  for (auto &change : changes) {
    MapVector<Value *, Value *> oldToNew;

    SetVector<Instruction *> instsToChange;
    for (auto node : change.nodes) {
      if (!node || !node->value) {
        continue;
      }
      auto *I = cast<Instruction>(node->value);
      if (VMap) {
        assert(VMap->count(I) && "the clone maps every operation");
        if (Value *Mapped = VMap->lookup(I)) {
          if (auto *MappedI = dyn_cast<Instruction>(Mapped)) {
            I = MappedI;
          } else {
            continue;
          }
        } else {
          continue;
        }
      }
      if (!operations.contains(I)) {
        // Already erased by an algebraic rewrite.
        continue;
      }
      instsToChange.insert(I);
    }

    SmallVector<Instruction *, 8> instsToChangeSorted;
    topoSort(instsToChange, instsToChangeSorted);

    if (unsigned nComp = expansionComponents(change.newType)) {
      SmallPtrSet<Instruction *, 8> allChangedSet(instsToChange.begin(),
                                                  instsToChange.end());
      DenseMap<Value *, Value *> restoredValues;
      Function *F = instsToChangeSorted.empty()
                        ? nullptr
                        : instsToChangeSorted.front()->getFunction();
      applyExpansion(nComp, instsToChangeSorted, allChangedSet, &restoredValues,
                     nComp == 2 ? &pairLimbs : &widerLimbs);
      // The parameter narrowing preserves values only once every load of the
      // buffer is a Dekker split, so it is tried after each pair change and
      // declines until then. It runs here so that foldDSPairRoundtrip sees the
      // join/split pair it cancels. A pricing clone is not a registered site,
      // and the narrowing declines on it.
      if (nComp == 2 && F &&
          narrowSharedStagingParamDS(S, *F, *F, /*announce=*/!VMap))
        foldDSPairRoundtrip(*F, /*foldedLimbs=*/nullptr,
                            S.Opts.Diagnostics.Print);
      if (VMap) {
        for (auto &[oldClonedI, restoredV] : restoredValues) {
          if (clonedToOriginal.count(oldClonedI)) {
            (*VMap)[clonedToOriginal[oldClonedI]] = restoredV;
          }
        }
      } else {
        for (auto *I : instsToChangeSorted)
          subgraph.operations.remove(I);
        if (graph)
          for (auto &[erased, restored] : restoredValues)
            graph->replaceErased(erased, restored);
      }
      continue;
    }

    for (auto *I : instsToChangeSorted) {
      changePrecision(I, change, oldToNew);
    }

    for (auto &[oldV, newV] : oldToNew) {
      if (!isa<Instruction>(oldV)) {
        continue;
      }

      if (!instsToChange.contains(cast<Instruction>(oldV))) {
        continue;
      }

      SmallPtrSet<Instruction *, 8> users;
      for (auto *user : oldV->users()) {
        assert(isa<Instruction>(user) &&
               "PT: Unexpected non-instruction user of a changed instruction");
        if (!instsToChange.contains(cast<Instruction>(user))) {
          users.insert(cast<Instruction>(user));
        }
      }

      Value *casted = nullptr;
      if (!users.empty()) {
        IRBuilder<> builder =
            builderAt(cast<Instruction>(oldV)->getParent(),
                      ++BasicBlock::iterator(cast<Instruction>(oldV)));
        casted = builder.CreateFPCast(
            newV, getLLVMFPType(change.oldType, builder.getContext()));

        if (VMap) {
          assert(VMap->count(clonedToOriginal[oldV]) &&
                 "a changed clone operation maps back to an operation");
          (*VMap)[clonedToOriginal[oldV]] = casted;
        } else if (graph) {
          graph->replaceErased(oldV, casted);
        }
      }

      for (auto *user : users) {
        user->replaceUsesOfWith(oldV, casted);
      }

      // No external uses of the old value remain: every new value was already
      // cast back and substituted.
      for (auto *user : oldV->users()) {
        assert(instsToChange.contains(cast<Instruction>(user)) &&
               "PT: Unexpected external user of a changed instruction");
      }

      if (!oldV->use_empty()) {
        oldV->replaceAllUsesWith(UndefValue::get(oldV->getType()));
      }

      cast<Instruction>(oldV)->eraseFromParent();

      if (!VMap)
        subgraph.operations.remove(cast<Instruction>(oldV));
    }
  }
  if (limbs) {
    limbs->append(pairLimbs.begin(), pairLimbs.end());
    limbs->append(widerLimbs.begin(), widerLimbs.end());
  }
}

// Demote FP64 PHIs whose every incoming is fpext-from-float or a
// float-representable constant. InstCombine folds fptrunc(fpext(x)) -> x but
// stops at a loop-carried PHI, so an FP32 accumulator the solver chose would
// otherwise keep its fptrunc/fpext roundtrip.
bool demoteFPCastPHIs(llvm::Function &F) {
  using namespace llvm;
  LLVMContext &Ctx = F.getContext();
  Type *DoubleTy = Type::getDoubleTy(Ctx);
  Type *FloatTy = Type::getFloatTy(Ctx);

  auto floatExtSrc = [&](Value *V) -> Value * {
    if (auto *FE = dyn_cast<FPExtInst>(V))
      if (FE->getOperand(0)->getType()->isFloatTy())
        return FE->getOperand(0);
    return nullptr;
  };
  auto floatReprConst = [&](Value *V, Constant *&out) -> bool {
    if (isa<UndefValue>(V)) {
      out = UndefValue::get(FloatTy);
      return true;
    }
    auto *CFP = dyn_cast<ConstantFP>(V);
    if (!CFP || !CFP->getType()->isDoubleTy())
      return false;
    APFloat v = CFP->getValueAPF();
    bool losesInfo = false;
    v.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven, &losesInfo);
    if (losesInfo)
      return false;
    out = ConstantFP::get(Ctx, v);
    return true;
  };

  SmallVector<PHINode *, 16> phis;
  SmallPtrSet<Value *, 16> inSet;
  for (BasicBlock &BB : F)
    for (PHINode &P : BB.phis())
      if (P.getType()->isDoubleTy()) {
        phis.push_back(&P);
        inSet.insert(&P);
      }
  if (phis.empty())
    return false;

  bool shrunk = true;
  while (shrunk) {
    shrunk = false;
    for (PHINode *P : phis) {
      if (!inSet.count(P))
        continue;
      for (Value *IV : P->incoming_values()) {
        Constant *c = nullptr;
        if (inSet.count(IV) || floatExtSrc(IV) || floatReprConst(IV, c))
          continue;
        inSet.erase(P);
        shrunk = true;
        break;
      }
    }
  }
  SmallVector<PHINode *, 16> D;
  for (PHINode *P : phis)
    if (inSet.count(P))
      D.push_back(P);
  if (D.empty())
    return false;

  DenseMap<PHINode *, PHINode *> nmap;
  for (PHINode *P : D) {
    PHINode *NP = PHINode::Create(FloatTy, P->getNumIncomingValues(),
                                  P->getName() + ".f32", P->getIterator());
    nmap[P] = NP;
  }
  for (PHINode *P : D) {
    PHINode *NP = nmap[P];
    for (unsigned i = 0; i < P->getNumIncomingValues(); ++i) {
      Value *IV = P->getIncomingValue(i);
      Constant *c = nullptr;
      Value *fv = nullptr;
      if (auto *IVP = dyn_cast<PHINode>(IV))
        if (nmap.count(IVP))
          fv = nmap[IVP];
      if (!fv)
        if (Value *s = floatExtSrc(IV))
          fv = s;
      if (!fv && floatReprConst(IV, c))
        fv = c;
      assert(fv && "demoteFPCastPHIs: non-leaf incoming survived closure");
      NP->addIncoming(fv, P->getIncomingBlock(i));
    }
  }
  for (PHINode *P : D) {
    PHINode *NP = nmap[P];
    BasicBlock *BB = NP->getParent();
    IRBuilder<> B = builderAt(BB, BB->getFirstInsertionPt());
    P->replaceAllUsesWith(B.CreateFPExt(NP, DoubleTy));
  }
  for (PHINode *P : D)
    P->eraseFromParent();
  return true;
}

static void setUnifiedAccuracyCost(
    const Options &O, PrecisionUnit &CS,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap) {
  SubgraphSamples &samples = samplesOf(O, *CS.subgraph, valueToNodeMap);
  ArrayRef<SubgraphSamples::Point> sampledPoints = samples.points();

  // No usable sample means the subgraph cannot be priced: drop its candidates
  // and keep the original body.
  auto unpriceable = [&](StringRef why) {
    CS.unpricedReason = why.str();
    llvm::errs() << "[poseidon] subgraph cannot be priced (" << why
                 << "); discarding its " << CS.candidates.size()
                 << " precision candidate(s) and keeping the original body.\n";
    CS.candidates.clear();
    CS.initialAccCost = 0.0;
  };
  if (sampledPoints.empty()) {
    unpriceable("the profiled input box is empty or unbounded");
    return;
  }

  SmallVector<FPNode *, 4> outputs;
  SmallVector<const SubgraphSamples::Reference *, 4> refs;
  SmallVector<std::optional<double>, 4> origErrors;
  for (auto *output : CS.subgraph->outputs) {
    FPNode *node = valueToNodeMap[output].get();
    outputs.push_back(node);
    CS.perOutputInitialAccCost[node] = 0.;
    refs.push_back(&samples.reference(node));
    origErrors.push_back(refs.back()->meanError());
  }
  if (!llvm::all_of(origErrors, [](auto e) { return e.has_value(); })) {
    unpriceable("every sample of the original expression scored NaN");
    return;
  }
  CS.initialAccCost = 0.0;
  for (auto [node, origError] : zip(outputs, origErrors)) {
    CS.perOutputInitialAccCost[node] = *origError * std::fabs(node->grad);
    CS.initialAccCost += CS.perOutputInitialAccCost[node];
  }
  assert(!std::isnan(CS.initialAccCost) &&
         "the subgraph as written has an accuracy cost");

  // Each candidate is scored on its own, concurrently.
  struct RunningAccArith {
    double sum = 0.0;
    unsigned count = 0;
  };
  struct Scored {
    bool discard = false;
    std::unordered_map<FPNode *, RunningAccArith> candAcc;
  };
  SmallVector<Scored, 0> scored(CS.candidates.size());
  forEachIndex(O.Model.Threads, CS.candidates.size(), [&](size_t c) {
    const PTCandidate &candidate = CS.candidates[c];
    Scored &out = scored[c];
    for (auto *node : outputs)
      out.candAcc[node] = RunningAccArith();

    SmallVector<double, 8> results;
    for (size_t i = 0; i < sampledPoints.size(); ++i) {
      getFPValues(outputs, sampledPoints[i], results, candidate.changes);
      for (size_t j = 0; j < outputs.size(); ++j) {
        double goldVal = refs[j]->gold[i];
        double result = results[j];
        if (std::isfinite(goldVal) && !std::isfinite(result)) {
          out.discard = true;
          break;
        }
        double error = sampleError(goldVal, result);
        if (!std::isnan(error)) {
          out.candAcc[outputs[j]].sum += error;
          ++out.candAcc[outputs[j]].count;
        }
      }
      if (out.discard)
        break;
    }
  });

  SmallVector<PTCandidate, 4> newCandidates;
  for (auto [candidate, out] : zip(CS.candidates, scored)) {
    if (!out.discard) {
      candidate.accuracyCost = 0.0;
      for (auto *node : outputs) {
        auto &ra = out.candAcc[node];
        assert(ra.count != 0 && "No valid sample found for candidate subgraph");
        double red = ra.sum / ra.count;
        candidate.perOutputAccCost[node] = red * std::fabs(node->grad);
        candidate.accuracyCost += candidate.perOutputAccCost[node];
      }
      assert(!std::isnan(candidate.accuracyCost) &&
             "a kept candidate has an accuracy cost");
      newCandidates.push_back(std::move(candidate));
    } else {
      CS.dropped.push_back({candidate.desc,
                            "the accuracy model found that it returns a "
                            "non-finite value at a sampled input where the "
                            "exact result is finite"});
    }
  }
  CS.candidates = std::move(newCandidates);
}

// Whether an operation is priced at its own measured frequency. A merged unit
// can span a loop body and code that runs once per thread, which one
// execution count misprices, so merging shared-staging subgraphs implies it.
static bool freqWeightedPricing(const Options &O) {
  return O.Model.FreqWeightedPricing || O.Candidates.MergeSharedStaging;
}

namespace {

// A clone of the site with a candidate applied to the operations of a
// subgraph and simplified as the materializer's output is, with handles on
// what the cost walk starts from. Its IR does not depend on the outputs, so
// one clone prices the candidate over any set of the subgraph's outputs.
class CandidateClone {
public:
  CandidateClone(Session &S, Subgraph &subgraph,
                 ArrayRef<Instruction *> outputs, PTCandidate &pt);
  ~CandidateClone() {
    cloneLI.reset();
    cloneDT.reset();
    FClone->eraseFromParent();
  }

  // Why the clone cannot be priced; empty when it can.
  const std::string &unpriceable() const { return unpricedReason; }

  // The cost of the cone of outputs[i] for each i in `which`, in that order,
  // and of the expansion roots. Only for a clone that can be priced.
  double cost(Session &S, const Subgraph &subgraph, ArrayRef<size_t> which);

private:
  // The instruction root i resolves to, if any.
  Instruction *rootAt(size_t i) const;

  Function *FClone = nullptr;
  SmallVector<WeakTrackingVH, 8> trackedInputs;
  // A handle per output, then one per expansion root; each with the FP32
  // operand of the extension it tracks, should the extension fold away.
  SmallVector<WeakTrackingVH, 8> trackedOutputs;
  SmallVector<WeakTrackingVH, 8> trackedOutputFallbacks;
  size_t numOutputs = 0;
  std::optional<DominatorTree> cloneDT;
  std::optional<LoopInfo> cloneLI;
  // The cost-model price of each instruction walked so far.
  DenseMap<const Instruction *, double> prices;
  std::string unpricedReason;
};

} // namespace

CandidateClone::CandidateClone(Session &S, Subgraph &subgraph,
                               ArrayRef<Instruction *> outputs,
                               PTCandidate &pt) {
  const Options &O = S.Opts;
  assert(!outputs.empty() && "a candidate is priced over some output");

  Function *F = outputs[0]->getFunction();

  ValueToValueMapTy VMap;
  FClone = CloneFunction(F, VMap);
  FClone->setName(F->getName() + "_clone");

  // The clone gets the materializer's carried-PHI elimination, so the df64
  // restore is not billed per iteration; the walk is re-rooted below on the
  // limbs the expansion changes produce.
  SmallVector<Value *, 32> expansionLimbs;
  bool ptHasExpansion = false, wantsDS = false;
  for (auto &ch : pt.changes) {
    ptHasExpansion |= expansionComponents(ch.newType) != 0;
    wantsDS |= ch.newType == PrecisionChangeType::Expansion2;
  }
  pt.apply(S, subgraph, &VMap, /*graph=*/nullptr, &expansionLimbs);
  // An Expansion2 change that converts nothing (unsupported instructions)
  // leaves no limbs, and the untouched FP64 outputs are still the right roots.
  const bool expansionMaterialized = !expansionLimbs.empty();

  // Run the materializer's cleanups, so that the price charges only the casts
  // the materializer emits. FClone has no call sites, so F, whose arguments are
  // FClone's, supplies the parameter-to-global mapping. A df64 candidate wants
  // the pair layout and any other the halved one, so one parameter arm runs.
  const ParamArm paramArm = wantsDS ? ParamArm::DS : ParamArm::FP32;
  // The df64 narrowing turns a staged load's split, some of whose halves are
  // limbs, into a pair load: the limbs follow it through handles.
  SmallVector<WeakTrackingVH, 32> limbHandles(expansionLimbs.begin(),
                                              expansionLimbs.end());
  applyStagingNarrowing(S, *FClone, /*announce=*/false, paramArm, *F);
  expansionLimbs.clear();
  for (Value *L : limbHandles)
    if (L)
      expansionLimbs.push_back(L);
  // Cancel the join/split roundtrip, as the emitted code does.
  SmallVector<Value *, 8> dsFoldRoots;
  foldDSPairRoundtrip(*FClone, &dsFoldRoots, O.Diagnostics.Print);

  for (auto &input : subgraph.inputs) {
    if (VMap.count(input))
      trackedInputs.emplace_back(VMap[input]);
  }
  // An output without a clone is held by an empty handle, which roots nothing.
  for (Instruction *output : outputs)
    trackedOutputs.emplace_back(VMap.count(output) ? VMap[output]
                                                   : WeakTrackingVH());
  numOutputs = trackedOutputs.size();
  // Re-root on the df64 limbs: they are the unit's outputs in the emitted code
  // and the F64 handles above are dead, so without them the candidate walks an
  // empty cone and prices as free.
  size_t liveExpansionRoots = 0;
  SmallVector<ArrayRef<Value *>, 2> limbLists = {
      ArrayRef<Value *>(expansionLimbs), ArrayRef<Value *>(dsFoldRoots)};
  for (ArrayRef<Value *> limbList : limbLists)
    for (Value *L : limbList)
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
    llvm::errs() << "  (expansion change converted no instruction; pricing "
                    "the unchanged cone)\n";

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

  // On a target without divergent branches the simplification can unswitch
  // a loop on a condition it does not vary in, and delete the body of the
  // version that never runs it: a handle on that body then holds poison, and
  // the walk would price an empty cone. The version that runs the body copies
  // its instructions with their metadata. So after each pass the instruction of
  // each handle is tagged, and a handle left on a constant moves to the one
  // copy of its tag, where later passes' replacements carry it.
  LLVMContext &Ctx = FClone->getContext();
  const unsigned TagKind = Ctx.getMDKindID("poseidon.cost.handle");
  SmallVector<MDNode *, 16> inputTags(trackedInputs.size());
  SmallVector<MDNode *, 16> rootTags(trackedOutputs.size());
  SmallVector<MDNode *, 16> fallbackTags(trackedOutputs.size());
  auto tag = [&](WeakTrackingVH &VH, MDNode *&Tag) {
    auto *I = dyn_cast_or_null<Instruction>((Value *)VH);
    if (!I)
      return;
    if (MDNode *Has = I->getMetadata(TagKind)) {
      Tag = Has;
      return;
    }
    if (!Tag)
      Tag = MDNode::getDistinct(Ctx, {});
    I->setMetadata(TagKind, Tag);
  };
  auto lost = [](const WeakTrackingVH &VH) {
    return VH.pointsToAliveValue() && !isa<Instruction>((Value *)VH);
  };
  // The walk stops at every copy of an input, but a root moves only to a
  // single copy: two are two loop versions, of which a thread runs one.
  auto trackHandles = [&] {
    for (auto [VH, Tag] : zip(trackedInputs, inputTags))
      tag(VH, Tag);
    for (auto [VH, Tag] : zip(trackedOutputs, rootTags))
      tag(VH, Tag);
    for (auto [VH, Tag] : zip(trackedOutputFallbacks, fallbackTags))
      tag(VH, Tag);
    if (none_of(trackedInputs, lost) && none_of(trackedOutputs, lost) &&
        none_of(trackedOutputFallbacks, lost))
      return;
    DenseMap<MDNode *, SmallVector<Instruction *, 1>> copies;
    for (Instruction &I : instructions(*FClone))
      if (MDNode *Tag = I.getMetadata(TagKind))
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
  };
  trackHandles();
  simplifyFunction(*FClone, S.PB, OptimizationLevel::O3, trackHandles);

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
      if (MDNode *Tag = I.getMetadata(TagKind))
        n += Tag == rootTags[i] || Tag == fallbackTags[i];
    unpricedReason = "the simplified clone of the site replaced " + what +
                     " by a constant while it was still used, and has " +
                     std::to_string(n) + " copies of it instead of one";
    break;
  }
  // A fold can also rebuild a root's computation in new instructions and
  // erase the root unreplaced, as narrowing fptrunc(op(fpext a, fpext b)) to
  // op(a, b) does. With every root gone so, the work is still there but no
  // handle reaches it.
  if (unpricedReason.empty() && !any_of(seq<size_t>(0, trackedOutputs.size()),
                                        [&](size_t i) { return rootAt(i); }))
    unpricedReason = "the simplified clone of the site erased every output "
                     "of the candidate while computing it in other "
                     "instructions";

  cloneDT.emplace(*FClone);
  cloneLI.emplace(*cloneDT);
}

Instruction *CandidateClone::rootAt(size_t i) const {
  if (auto *I = dyn_cast_or_null<Instruction>((Value *)trackedOutputs[i]))
    return I;
  return dyn_cast_or_null<Instruction>((Value *)trackedOutputFallbacks[i]);
}

double CandidateClone::cost(Session &S, const Subgraph &subgraph,
                            ArrayRef<size_t> which) {
  assert(unpricedReason.empty() && "the clone cannot be priced");
  const Options &O = S.Opts;

  SmallPtrSet<Value *, 8> clonedInputs;
  for (auto &VH : trackedInputs) {
    if (!VH.pointsToAliveValue())
      continue;
    Value *V = VH;
    if (V)
      clonedInputs.insert(V);
  }

  // In output order: the walk below sums the cost in the order it reaches the
  // instructions.
  SmallVector<size_t, 8> roots(which.begin(), which.end());
  for (size_t i = numOutputs; i < trackedOutputs.size(); ++i)
    roots.push_back(i);
  SmallSetVector<Value *, 8> clonedOutputs;
  for (size_t i : roots)
    if (Instruction *I = rootAt(i))
      clonedOutputs.insert(I);
  // price() kept only clones with a root; a subset of outputs left none.
  if (clonedOutputs.empty())
    report_fatal_error("Poseidon: none of the outputs a precision candidate is "
                       "priced over is left in its simplified clone. Refusing "
                       "to fabricate a cost.");

  SmallPtrSet<Value *, 8> seen;
  SmallVector<Value *, 8> todo;
  SmallVector<Instruction *, 32> coneInsts;

  todo.insert(todo.end(), clonedOutputs.begin(), clonedOutputs.end());
  while (!todo.empty()) {
    auto cur = todo.pop_back_val();
    if (!seen.insert(cur).second)
      continue;

    if (clonedInputs.contains(cur))
      continue;

    if (auto *I = dyn_cast<Instruction>(cur)) {
      coneInsts.push_back(I);

      auto operands =
          isa<CallInst>(I) ? cast<CallInst>(I)->args() : I->operands();
      for (auto &operand : operands) {
        todo.push_back(operand);
      }
    }
  }

  // A conversion the simplified clone hoists out of every loop runs at most
  // once per thread per launch, so it is scaled by outBoundaryFreqScale when
  // the cone's other work is in a loop. One left in a loop runs at least once
  // per iteration of it and counts as the walk's other instructions do.
  unsigned bodyDepth = 0;
  for (Instruction *I : coneInsts)
    if (!isa<FPExtInst>(I) && !isa<FPTruncInst>(I))
      bodyDepth = std::max(bodyDepth, cloneLI->getLoopDepth(I->getParent()));

  const bool freqWeighted =
      freqWeightedPricing(O) && subgraph.depthExec.size() > 1;
  double cost = 0.0;
  for (Instruction *I : coneInsts) {
    auto [price, fresh] = prices.try_emplace(I);
    if (fresh)
      price->second = getInstructionCompCost(S.CostModel, I);
    double c = price->second;
    unsigned d = cloneLI->getLoopDepth(I->getParent());
    if (freqWeighted) {
      // The measured frequency of this depth subsumes the boundary-cast scale
      // and also reaches the arithmetic emitted outside the body loop, where a
      // df64 boundary conversion lives.
      c *= subgraph.freqWeightAtDepth(d);
    } else if ((isa<FPExtInst>(I) || isa<FPTruncInst>(I)) && bodyDepth > 0 &&
               d == 0) {
      c *= subgraph.outBoundaryFreqScale;
    }
    cost += c;
  }
  return cost;
}

// The cost of `pt` per execution of `subgraph`, or why it cannot be computed.
static Expected<double> priceCandidate(Session &S, Subgraph &subgraph,
                                       PTCandidate &pt) {
  CandidateClone clone(S, subgraph, subgraph.outputs.getArrayRef(), pt);
  if (!clone.unpriceable().empty())
    return createStringError(clone.unpriceable());
  SmallVector<size_t, 8> all;
  for (size_t i = 0; i < subgraph.outputs.size(); ++i)
    all.push_back(i);
  return clone.cost(S, subgraph, all);
}

static constexpr int kThreeTierStep = 20;

namespace {

struct TierPair {
  PrecisionChangeType hi;
  PrecisionChangeType lo;
};

struct TierTriple {
  PrecisionChangeType hi;
  PrecisionChangeType mid;
  PrecisionChangeType lo;
};

} // namespace

static const TierPair kCanonicalPairs[] = {
    {PrecisionChangeType::FP64, PrecisionChangeType::FP32},
    {PrecisionChangeType::FP64, PrecisionChangeType::Expansion2},
    {PrecisionChangeType::FP64, PrecisionChangeType::FP16},
    {PrecisionChangeType::FP64, PrecisionChangeType::BF16},
    {PrecisionChangeType::Expansion2, PrecisionChangeType::FP32},
    {PrecisionChangeType::Expansion2, PrecisionChangeType::FP16},
    {PrecisionChangeType::Expansion2, PrecisionChangeType::BF16},
    {PrecisionChangeType::FP32, PrecisionChangeType::FP16},
    {PrecisionChangeType::FP32, PrecisionChangeType::BF16},
    {PrecisionChangeType::FP64, PrecisionChangeType::Expansion3},
    {PrecisionChangeType::Expansion3, PrecisionChangeType::Expansion2},
    {PrecisionChangeType::Expansion3, PrecisionChangeType::FP32},
    {PrecisionChangeType::FP64, PrecisionChangeType::Expansion4},
    {PrecisionChangeType::Expansion4, PrecisionChangeType::Expansion3},
    {PrecisionChangeType::Expansion4, PrecisionChangeType::Expansion2},
    {PrecisionChangeType::Expansion4, PrecisionChangeType::FP32},
};

static const TierTriple kCanonicalTriples[] = {
    {PrecisionChangeType::FP64, PrecisionChangeType::Expansion2,
     PrecisionChangeType::FP32},
    {PrecisionChangeType::FP64, PrecisionChangeType::Expansion2,
     PrecisionChangeType::FP16},
    {PrecisionChangeType::FP64, PrecisionChangeType::Expansion2,
     PrecisionChangeType::BF16},
    {PrecisionChangeType::FP64, PrecisionChangeType::FP32,
     PrecisionChangeType::FP16},
    {PrecisionChangeType::FP64, PrecisionChangeType::FP32,
     PrecisionChangeType::BF16},
    {PrecisionChangeType::Expansion2, PrecisionChangeType::FP32,
     PrecisionChangeType::FP16},
    {PrecisionChangeType::Expansion2, PrecisionChangeType::FP32,
     PrecisionChangeType::BF16},
};

// `hwScalar` is the target's scalar types; null off the GPU.
static bool precAllowed(const Options &O, PrecisionChangeType t, bool gpuMode,
                        bool expansionSafe,
                        const std::unordered_set<std::string> *hwScalar) {
  if (unsigned nComp = expansionComponents(t)) {
    if (!gpuMode || !expansionSafe)
      return false;
    if (!O.Candidates.EnableMultifloat)
      return false;
    if (nComp > O.Candidates.ExpansionComponents)
      return false;
  }
  if (t == PrecisionChangeType::FP16 && (!gpuMode || !hwScalar->count("half")))
    return false;
  if (t == PrecisionChangeType::BF16 && (!gpuMode || !hwScalar->count("bf16")))
    return false;
  return true;
}

// Whether every tier of a tier pair or triple is allowed.
static bool tiersAllowed(const Options &O, ArrayRef<PrecisionChangeType> tiers,
                         bool gpuMode, bool expansionSafe,
                         const std::unordered_set<std::string> *hwScalar) {
  return all_of(tiers, [&](PrecisionChangeType t) {
    return precAllowed(O, t, gpuMode, expansionSafe, hwScalar);
  });
}

// The tiers of a candidate with the percentage of the operations each takes,
// as its description lists them: "FP64(20%) + FP32(80%)".
static std::string
fmtTiers(ArrayRef<std::pair<PrecisionChangeType, int>> tierPcts) {
  std::string s;
  for (const auto &[t, pct] : tierPcts) {
    if (!s.empty())
      s += " + ";
    s += getPrecisionChangeTypeString(t).str() + "(" + std::to_string(pct) +
         "%)";
  }
  return s;
}

void proposePrecisionCandidates(
    Session &S, Function &F, Subgraph &subgraph,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    const FunctionProfileHeader &profileHeader, bool expansionSafe,
    SmallVectorImpl<PrecisionUnit> &CSs) {
  const Options &O = S.Opts;
  auto *o0 = subgraph.outputs[0];
  const uint64_t o0Exec = valueToNodeMap[o0]->executions;

  // Filled before CS is built, so its baseline walk and every candidate's
  // weight each instruction by its own measured frequency.
  subgraph.opExec.clear();
  subgraph.depthExec.clear();
  subgraph.execNormalizer = o0Exec;
  if (freqWeightedPricing(O)) {
    DominatorTree freqDT(F);
    LoopInfo freqLI(freqDT);
    auto note = [&](Instruction *I) {
      auto it = valueToNodeMap.find(I);
      if (it == valueToNodeMap.end() || !it->second ||
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
    for (auto *op : subgraph.operations)
      note(op);
    for (auto *o : subgraph.outputs)
      note(o);
    // A materialized instruction keeps only its loop depth, so the baseline
    // is snapped onto the same depth classes as the candidate walk.
    for (auto &kv : subgraph.opExec)
      if (auto *I = dyn_cast<Instruction>(kv.first)) {
        auto dit = subgraph.depthExec.find(freqLI.getLoopDepth(I->getParent()));
        if (dit != subgraph.depthExec.end())
          kv.second = dit->second;
      }
    if (O.Diagnostics.Print && subgraph.depthExec.size() > 1) {
      llvm::errs() << "[poseidon] frequency-heterogeneous subgraph "
                      "(normalizer "
                   << o0Exec << ", measured depth:exec";
      for (auto &kv : subgraph.depthExec)
        llvm::errs() << ' ' << kv.first << ':' << kv.second;
      llvm::errs() << "); pricing each instruction at its own measured "
                      "frequency\n";
    }
  }

  PrecisionUnit CS(S.CostModel, subgraph);
  CS.executions = o0Exec;

  // Synthetic output-boundary casts execute at most once per thread per
  // launch, while the body executes CS.executions times; derive the scale
  // from the profile header (1 for elementwise subgraphs).
  if (CS.executions > 0 && profileHeader.launchCount > 0) {
    double thr = 1.0;
    for (int d = 0; d < 3; ++d)
      thr *= std::max<uint32_t>(1, profileHeader.maxBlockDim[d]);
    for (int d = 0; d < 3; ++d)
      thr *= std::max<uint32_t>(1, profileHeader.maxGridDim[d]);
    subgraph.outBoundaryFreqScale = std::min(
        1.0, thr * (double)profileHeader.launchCount / (double)CS.executions);
  }

  const auto &PTFuncs = getPTFuncs(S.CostModel, F);
  SetVector<FPLLValue *> funcsSet, allSet;
  for (auto *I : subgraph.operations) {
    assert(isa<FPLLValue>(valueToNodeMap[I].get()) &&
           "Corrupted FPNode for original instructions");
    auto node = cast<FPLLValue>(valueToNodeMap[I].get());
    allSet.insert(node);
    if (PTFuncs.count(node->op) != 0) {
      funcsSet.insert(node);
      llvm::errs() << "[poseidon] PT Function identified: " << *I << '\n';
    }
  }
  SmallVector<FPLLValue *> sortedFuncs(funcsSet.begin(), funcsSet.end());
  SmallVector<FPLLValue *> sortedAllOps(allSet.begin(), allSet.end());
  // Stable: which tier an operation of a tied sensitivity lands in follows the
  // subgraph's order, not the sort algorithm's.
  auto bySens = [](const auto &a, const auto &b) { return a->sens < b->sens; };
  llvm::stable_sort(sortedFuncs, bySens);
  llvm::stable_sort(sortedAllOps, bySens);

  const bool gpuMode = Target::of(F).isDevice();
  const std::unordered_set<std::string> *hwScalar =
      gpuMode ? &getScalarTypes(S.CostModel, F) : nullptr;
  PrecisionChangeType curr =
      getPrecisionChangeType(subgraph.outputs[0]->getType());

  auto emitCandidate =
      [&](SmallVectorImpl<std::pair<PrecisionChangeType,
                                    SetVector<FPLLValue *>>> &assignment,
          std::string desc) {
        SmallVector<PrecisionChange, 3> changes;
        for (auto &kv : assignment) {
          if (kv.first != curr && !kv.second.empty())
            changes.emplace_back(kv.second, curr, kv.first);
        }
        if (changes.empty())
          return;
        PTCandidate cand{std::move(changes), std::move(desc)};
        cand.ordinal = CS.candidates.size();
        CS.candidates.push_back(std::move(cand));
      };

  // The description of a candidate of `label` (All or Funcs) with `tiers`,
  // announced under -poseidon-print.
  auto describe = [&](StringRef label, StringRef kind, const std::string &tiers,
                      size_t N) {
    if (O.Diagnostics.Print)
      llvm::errs() << "Created " << label << " " << kind
                   << " PT candidate: " << tiers << " (N=" << N << ")\n";
    std::string desc = label.str();
    if (!desc.empty())
      desc += " ";
    return desc + tiers;
  };

  auto sweepTwoTier = [&](TierPair tp, ArrayRef<FPLLValue *> sortedAsc,
                          StringRef label) {
    const size_t N = sortedAsc.size();
    const int step = std::max(5, O.Candidates.TwoTierStep);
    size_t prev = N + 1;
    for (int pct = 0; pct <= 100 - step; pct += step) {
      size_t k = N * pct / 100;
      if (k == prev)
        continue;
      prev = k;

      SetVector<FPLLValue *> hiOps(sortedAsc.end() - k, sortedAsc.end());
      SetVector<FPLLValue *> loOps(sortedAsc.begin(), sortedAsc.end() - k);

      std::string desc = describe(
          label, "two-tier", fmtTiers({{tp.hi, pct}, {tp.lo, 100 - pct}}), N);

      SmallVector<std::pair<PrecisionChangeType, SetVector<FPLLValue *>>, 2>
          assignment;
      assignment.emplace_back(tp.hi, std::move(hiOps));
      assignment.emplace_back(tp.lo, std::move(loOps));
      emitCandidate(assignment, std::move(desc));
    }
  };

  auto sweepThreeTier = [&](TierTriple tr, ArrayRef<FPLLValue *> sortedAsc,
                            StringRef label) {
    const size_t N = sortedAsc.size();
    const int step = kThreeTierStep;
    for (int pctHi = step; pctHi <= 100 - 2 * step; pctHi += step) {
      for (int pctHiMid = pctHi + step; pctHiMid <= 100 - step;
           pctHiMid += step) {
        size_t k0 = N * pctHi / 100;
        size_t k1 = N * pctHiMid / 100;
        if (k0 == 0 || k0 >= k1 || k1 >= N)
          continue;

        SetVector<FPLLValue *> hiOps(sortedAsc.end() - k0, sortedAsc.end());
        SetVector<FPLLValue *> midOps(sortedAsc.end() - k1,
                                      sortedAsc.end() - k0);
        SetVector<FPLLValue *> loOps(sortedAsc.begin(), sortedAsc.end() - k1);

        std::string desc = describe(label, "three-tier",
                                    fmtTiers({{tr.hi, pctHi},
                                              {tr.mid, pctHiMid - pctHi},
                                              {tr.lo, 100 - pctHiMid}}),
                                    N);

        SmallVector<std::pair<PrecisionChangeType, SetVector<FPLLValue *>>, 3>
            assignment;
        assignment.emplace_back(tr.hi, std::move(hiOps));
        assignment.emplace_back(tr.mid, std::move(midOps));
        assignment.emplace_back(tr.lo, std::move(loOps));
        emitCandidate(assignment, std::move(desc));
      }
    }
  };

  for (TierPair tp : kCanonicalPairs) {
    if (!tiersAllowed(O, {tp.hi, tp.lo}, gpuMode, expansionSafe, hwScalar))
      continue;
    sweepTwoTier(tp, sortedAllOps, "All");
    if (!sortedFuncs.empty())
      sweepTwoTier(tp, sortedFuncs, "Funcs");
  }

  if (O.Candidates.EnableThreeTier) {
    for (TierTriple tr : kCanonicalTriples) {
      if (!tiersAllowed(O, {tr.hi, tr.mid, tr.lo}, gpuMode, expansionSafe,
                        hwScalar))
        continue;
      sweepThreeTier(tr, sortedAllOps, "All");
    }
  }

  // Zero candidates for a nonempty subgraph means every tier was filtered
  // out (a configuration problem), which must not pass as a no-op frontier.
  if (CS.candidates.empty() && !subgraph.operations.empty()) {
    llvm::errs() << "[poseidon] WARNING: subgraph with "
                 << subgraph.operations.size() << " FP operations in "
                 << F.getName()
                 << " synthesized ZERO precision-tuning candidates (all "
                    "tier combinations filtered); the solve cannot "
                    "propose any rewrite here. Check the cost model's "
                    "scalar_types header.\n";
    if (O.Model.StrictMode)
      reportUserError(F, F.getSubprogram(),
                      "strict mode: an FP subgraph of " + F.getName() +
                          " has no precision candidate with precision tuning "
                          "enabled; check the scalar_types header of the cost "
                          "model.");
  }

  CSs.push_back(std::move(CS));
}

double PrecisionUnit::getCompCostDelta(size_t candidateIndex) {
  return (candidates[candidateIndex].CompCost - initialCompCost) *
         (double)executions;
}

PrecisionUnit::RepriceKey
PrecisionUnit::repriceKey(size_t candidateIndex,
                          ArrayRef<SolutionStep> steps) const {
  RewriteUnitSet RewriteUnits;
  for (const auto &step : steps) {
    if (auto *co = dyn_cast<RewriteUnit>(step.unit)) {
      if (co->subgraph == subgraph) {
        RewriteUnits.insert(co);
      }
    }
  }
  return RepriceKey{candidateIndex, RewriteUnits};
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

    Subgraph newSubgraph = *this->subgraph;

    for (auto &step : steps) {
      if (auto *co = dyn_cast<RewriteUnit>(step.unit)) {
        const auto &CO = *co;
        if (CO.subgraph == subgraph) {
          newSubgraph.operations.remove_if(
              [&CO](Instruction *I) { return CO.erasableInsts.contains(I); });
          for (Value *root : CO.roots)
            newSubgraph.outputs.remove(cast<Instruction>(root));
        }
      }
    }

    if (newSubgraph.outputs.empty()) {
      compCostDeltaCache[key] = 0;
      continue;
    }

    std::vector<const Instruction *> changed;
    for (const PrecisionChange &change : candidates[candidateIndex].changes) {
      for (const auto *node : change.nodes)
        if (node && node->value &&
            newSubgraph.operations.contains(cast<Instruction>(node->value)))
          changed.push_back(cast<Instruction>(node->value));
      changed.push_back(nullptr);
    }
    auto [it, fresh] = groupOf.try_emplace({candidateIndex, std::move(changed)},
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
      auto [initial, fresh] = initialCompCosts.try_emplace(R.outputs);
      if (fresh)
        initial->second =
            getCompCost(S.CostModel, R.outputs, subgraph->inputs,
                        &subgraph->opExec, subgraph->execNormalizer);
      double initialCompCost = initial->second;
      double candidateCompCost = clone.cost(S, *subgraph, R.which);
      double adjustedCostDelta =
          (candidateCompCost - initialCompCost) * (double)executions;
      // With a one-point frontier these lines record why each candidate lost.
      if (S.Opts.Diagnostics.Print)
        llvm::errs() << "CS candidate '" << candidates[G.candidateIndex].desc
                     << "': initial=" << initialCompCost
                     << " cand=" << candidateCompCost << " exec=" << executions
                     << " Δcost=" << adjustedCostDelta
                     << " Δacc=" << getAccCostDelta(G.candidateIndex) << '\n';
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

  auto cacheIt = accCostDeltaCache.find(key);
  if (cacheIt != accCostDeltaCache.end()) {
    return cacheIt->second;
  }

  double totalCandidateAccCost = 0.0;
  double totalInitialAccCost = 0.0;

  SmallPtrSet<FPNode *, 8> stepNodes;
  for (const auto &step : steps) {
    if (auto *co = dyn_cast<RewriteUnit>(step.unit)) {
      const auto &CO = *co;
      if (CO.subgraph == subgraph) {
        for (Value *root : CO.roots) {
          auto it = valueToNodeMap.find(root);
          assert(it != valueToNodeMap.end() && it->second &&
                 "a rewritten output has a node");
          stepNodes.insert(it->second.get());
        }
      }
    }
  }

  for (auto &[node, cost] : perOutputInitialAccCost) {
    if (!stepNodes.count(node)) {
      totalInitialAccCost += cost;
    }
  }

  for (auto &[node, cost] : candidates[candidateIndex].perOutputAccCost) {
    if (!stepNodes.count(node)) {
      totalCandidateAccCost += cost;
    }
  }

  double adjustedAccCostDelta = totalCandidateAccCost - totalInitialAccCost;

  accCostDeltaCache[key] = adjustedAccCostDelta;
  return adjustedAccCostDelta;
}

bool PrecisionUnit::pricedAgainst(const OptimizationUnit &earlier) const {
  const auto *co = dyn_cast<RewriteUnit>(&earlier);
  return co && co->subgraph == subgraph;
}

std::string PrecisionUnit::choiceText(size_t i) const {
  const PTCandidate &cand = candidates[i];
  return "CS: " + cand.desc + " (#" + std::to_string(cand.ordinal) + ")";
}

void PrecisionUnit::materialize(size_t i) {
  assert(i < candidates.size() && "the solver names one of the candidates");
  // An operation this change erases may be an input of another subgraph whose
  // rewrite a later step materializes: the rewrite reads the value the
  // operation's users now read, as the units were priced.
  candidates[i].apply(session(), *subgraph, nullptr, &graph());
}

void PrecisionUnit::price() {
  SmallVector<PTCandidate, 8> priced;
  std::string firstWhy;
  for (PTCandidate &c : candidates) {
    Expected<double> cost = priceCandidate(session(), *subgraph, c);
    if (cost) {
      c.CompCost = *cost;
      priced.push_back(std::move(c));
      continue;
    }
    std::string why =
        "its cost cannot be computed: " + toString(cost.takeError());
    if (firstWhy.empty())
      firstWhy = why;
    dropped.push_back({c.desc, std::move(why)});
  }
  if (!firstWhy.empty())
    llvm::errs() << "[poseidon] "
                 << subgraph->outputs[0]->getFunction()->getName() << ": "
                 << (candidates.size() - priced.size()) << " of "
                 << candidates.size()
                 << " precision candidate(s) dropped, first because "
                 << firstWhy << '\n';
  candidates = std::move(priced);
  setUnifiedAccuracyCost(session().Opts, *this, graph().valueToNodeMap);
}

void PrecisionUnit::scaleErrors(double factor) {
  initialAccCost *= factor;
  for (auto &[node, cost] : perOutputInitialAccCost)
    cost *= factor;
  for (auto &cand : candidates) {
    cand.accuracyCost *= factor;
    for (auto &[node, cost] : cand.perOutputAccCost)
      cost *= factor;
  }
}

void PrecisionUnit::printPriced(raw_ostream &OS) {
  OS << "\n################################\n";
  OS << "Initial AccuracyCost: " << initialAccCost << '\n';
  OS << "Initial ComputationCost: " << initialCompCost << '\n';
  OS << "Candidates:\n";
  OS << "Δ AccCost\t\tΔ CompCost\t\tDescription\n"
     << "---------------------------\n";
  for (size_t i = 0; i < candidates.size(); ++i) {
    auto &candidate = candidates[i];
    OS << getAccCostDelta(i) << "\t\t";
    if (candidate.CompCost == std::numeric_limits<double>::max())
      OS << "unpriced";
    else
      OS << getCompCostDelta(i);
    OS << "\t\t" << candidate.desc << '\n';
  }
  OS << "################################\n\n";
}

} // namespace poseidon
