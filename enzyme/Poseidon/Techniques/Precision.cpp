//=- Precision.cpp - precision candidates of an FP subgraph -----------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
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
#include "Targets/DeviceMath.h"
#include "Targets/Target.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <cmath>
#include <map>
#include <memory>
#include <optional>
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

  if (isa<UnaryOperator>(I) || isa<BinaryOperator>(I)) {
    SmallVector<Value *, 2> newOps;
    for (auto &operand : I->operands()) {
      Value *newOp = nullptr;
      if (oldToNew.count(operand)) {
        newOp = oldToNew[operand];
      } else if (operand->getType()->isIntegerTy()) {
        newOp = operand;
      } else {
        IRBuilder<> OpBuilder(I);
        OpBuilder.setFastMathFlags(I->getFastMathFlags());
        if (isa<Constant>(operand)) {
          newOp =
              OpBuilder.CreateFPCast(operand, newType, "poseidon.const.fpcast");
        } else if (isa<Argument>(operand) || isa<Instruction>(operand)) {
          newOp = OpBuilder.CreateFPCast(operand, newType, "poseidon.fpcast");
        } else {
          llvm_unreachable("Unsupported operand type");
        }
      }
      newOps.push_back(newOp);
    }
    newI = Builder.CreateNAryOp(I->getOpcode(), newOps);
  } else if (auto *CI = dyn_cast<CallInst>(I)) {
    SmallVector<Value *, 4> newArgs;
    for (auto &arg : CI->args()) {
      Value *newArg = nullptr;
      if (oldToNew.count(arg)) {
        newArg = oldToNew[arg];
      } else if (arg->getType()->isIntegerTy()) {
        newArg = arg;
      } else {
        IRBuilder<> ArgBuilder(I);
        ArgBuilder.setFastMathFlags(I->getFastMathFlags());
        if (isa<Constant>(arg)) {
          newArg =
              ArgBuilder.CreateFPCast(arg, newType, "poseidon.const.fpcast");
        } else if (isa<Argument>(arg) || isa<Instruction>(arg)) {
          newArg = ArgBuilder.CreateFPCast(arg, newType, "poseidon.fpcast");
        } else {
          llvm_unreachable("Unsupported argument type");
        }
      }
      newArgs.push_back(newArg);
    }
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
        StringRef base = libmBaseName(funcName);
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
                        FPGraph *graph) {
  SetVector<Instruction *> operations;
  DenseMap<Value *, Value *>
      clonedToOriginal; // Maps cloned outputs to old outputs
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

  // One limb record per candidate: the loop below may apply several expansion
  // changes, and the cost walk is rooted on every limb they produce.
  resetExpLimbs(S);

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
      if (nComp == 2)
        applyExpansion(S, instsToChangeSorted, allChangedSet, &restoredValues);
      else
        applyExpansion(S, nComp, instsToChangeSorted, allChangedSet,
                       &restoredValues);
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

void setUnifiedAccuracyCost(
    const Options &O, PrecisionUnit &CS,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap) {
  ArrayRef<SubgraphSamples::Point> sampledPoints =
      samplesOf(O, *CS.subgraph, valueToNodeMap).points();

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

  MapVector<FPNode *, SmallVector<double, 4>> goldVals;
  for (auto *output : CS.subgraph->outputs) {
    auto *node = valueToNodeMap[output].get();
    goldVals[node].resize(O.Model.NumSamples);
    CS.perOutputInitialAccCost[node] = 0.;
  }

  SmallVector<FPNode *, 4> outputs;
  for (auto *output : CS.subgraph->outputs)
    outputs.push_back(valueToNodeMap[output].get());

  struct RunningAccArith {
    double sum = 0.0;
    unsigned count = 0;
  };
  std::unordered_map<FPNode *, RunningAccArith> runAcc;
  for (auto *node : outputs)
    runAcc[node] = RunningAccArith();

  // Every point is evaluated on its own; the sums run in point order.
  SmallVector<SmallVector<double, 8>, 0> exact(sampledPoints.size()),
      asWritten(sampledPoints.size());
  forEachBlock(O.Model.Threads, sampledPoints.size(), 8,
               [&](size_t begin, size_t end) {
                 for (size_t i = begin; i < end; ++i) {
                   getMPFRValues(outputs, sampledPoints[i], exact[i], 53);
                   getFPValues(outputs, sampledPoints[i], asWritten[i]);
                 }
               });
  for (size_t i = 0; i < sampledPoints.size(); ++i) {
    for (const auto &[node, result] : zip(outputs, exact[i]))
      goldVals[node][i] = result;

    for (const auto &[node, result] : zip(outputs, asWritten[i])) {
      double goldVal = goldVals[node][i];
      double error = sampleError(goldVal, result);
      if (!std::isnan(error)) {
        runAcc[node].sum += error;
        ++runAcc[node].count;
      }
    }
  }
  if (llvm::any_of(outputs,
                   [&](FPNode *node) { return runAcc[node].count == 0; })) {
    unpriceable("every sample of the original expression scored NaN");
    return;
  }
  CS.initialAccCost = 0.0;
  for (auto *node : outputs) {
    auto &ra = runAcc[node];
    double red = ra.sum / ra.count;
    CS.perOutputInitialAccCost[node] = red * std::fabs(node->grad);
    CS.initialAccCost += CS.perOutputInitialAccCost[node];
  }
  assert(!std::isnan(CS.initialAccCost) &&
         "the subgraph as written has an accuracy cost");

  // Each candidate is scored on its own, concurrently.
  SmallVector<const SmallVector<double, 4> *, 4> goldOf;
  for (auto *node : outputs)
    goldOf.push_back(&goldVals.find(node)->second);
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
        double goldVal = (*goldOf[j])[i];
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
                            "it returns a non-finite value at a sampled input "
                            "where the exact result is finite"});
    }
  }
  CS.candidates = std::move(newCandidates);
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

  // The cost of the cone of outputs[i] for each i in `which`, in that order,
  // and of the expansion roots.
  double cost(Session &S, const Subgraph &subgraph, ArrayRef<size_t> which);

private:
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
  // limbs applyExpansion records.
  SmallVector<Value *, 32> expansionLimbs;
  bool ptHasExpansion = false;
  for (auto &ch : pt.changes)
    if (expansionComponents(ch.newType))
      ptHasExpansion = true;
  pt.apply(S, subgraph, &VMap);
  if (ptHasExpansion) {
    // A tier pair can apply an expansion change and a double-single change,
    // each recording its own roots.
    takeLastExpansionLimbs(S, expansionLimbs);
    takeLastExpLimbs(S, expansionLimbs);
  }
  // An Expansion2 change that converts nothing (unsupported instructions)
  // leaves no limbs, and the untouched FP64 outputs are still the right roots.
  const bool expansionMaterialized = !expansionLimbs.empty();

  // Run the materializer's cleanups, so that the price charges only the casts
  // the materializer emits. FClone has no call sites, so F, whose arguments are
  // FClone's, supplies the parameter-to-global mapping. A df64 candidate wants
  // the pair layout and any other the halved one, so one parameter arm runs.
  const bool wantsDS = [&] {
    for (auto &ch : pt.changes)
      if (ch.newType == PrecisionChangeType::Expansion2)
        return true;
    return false;
  }();
  const ParamArm paramArm = !O.Materialization.NarrowParamStaging
                                ? ParamArm::None
                            : wantsDS ? ParamArm::DS
                                      : ParamArm::FP32;
  // The df64 narrowing turns a staged load's split, some of whose halves are
  // limbs, into a pair load: the limbs follow it through handles.
  SmallVector<WeakTrackingVH, 32> limbHandles(expansionLimbs.begin(),
                                              expansionLimbs.end());
  applyStagingNarrowing(S, *FClone, /*announce=*/false, paramArm, F);
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

  simplifyFunction(*FClone, OptimizationLevel::O3);

  cloneDT.emplace(*FClone);
  cloneLI.emplace(*cloneDT);
}

double CandidateClone::cost(Session &S, const Subgraph &subgraph,
                            ArrayRef<size_t> which) {
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
  for (size_t i : roots) {
    Value *V = nullptr;
    if (trackedOutputs[i].pointsToAliveValue())
      V = trackedOutputs[i];
    else if (trackedOutputFallbacks[i].pointsToAliveValue())
      V = trackedOutputFallbacks[i];
    if (V)
      clonedOutputs.insert(V);
  }

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

  // Boundary casts the simplified clone hoists out of the cone's body loop, the
  // deepest loop of its other work, execute at most once per thread per
  // launch, so a cast at a shallower depth is scaled by outBoundaryFreqScale.
  unsigned bodyDepth = 0;
  for (Instruction *I : coneInsts)
    if (!isa<FPExtInst>(I) && !isa<FPTruncInst>(I))
      bodyDepth = std::max(bodyDepth, cloneLI->getLoopDepth(I->getParent()));

  const bool freqWeighted =
      O.Model.FreqWeightedPricing && subgraph.depthExec.size() > 1;
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
               d < bodyDepth) {
      c *= subgraph.outBoundaryFreqScale;
    }
    cost += c;
  }
  return cost;
}

double getCompCost(Session &S, Subgraph &subgraph, PTCandidate &pt) {
  CandidateClone clone(S, subgraph, subgraph.outputs.getArrayRef(), pt);
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

static bool precAllowed(const Options &O, PrecisionChangeType t, bool gpuMode,
                        bool expansionSafe,
                        const std::unordered_set<std::string> &hwScalar) {
  if (unsigned nComp = expansionComponents(t)) {
    if (!gpuMode || !expansionSafe)
      return false;
    if (!O.Candidates.EnableMultifloat)
      return false;
    if (nComp > O.Candidates.ExpansionComponents)
      return false;
  }
  if (t == PrecisionChangeType::FP16 && (!gpuMode || !hwScalar.count("half")))
    return false;
  if (t == PrecisionChangeType::BF16 && (!gpuMode || !hwScalar.count("bf16")))
    return false;
  return true;
}

// Whether every tier of a tier pair or triple is allowed.
static bool tiersAllowed(const Options &O, ArrayRef<PrecisionChangeType> tiers,
                         bool gpuMode, bool expansionSafe,
                         const std::unordered_set<std::string> &hwScalar) {
  return all_of(tiers, [&](PrecisionChangeType t) {
    return precAllowed(O, t, gpuMode, expansionSafe, hwScalar);
  });
}

static std::string fmtPrecPct(PrecisionChangeType t, int pct) {
  std::string s = getPrecisionChangeTypeString(t).str();
  s += "(";
  s += std::to_string(pct);
  s += "%)";
  return s;
}

// The tiers of a candidate with the percentage of the operations each takes,
// as its description lists them: "FP64(20%) + FP32(80%)".
static std::string
fmtTiers(ArrayRef<std::pair<PrecisionChangeType, int>> tierPcts) {
  std::string s;
  for (const auto &[t, pct] : tierPcts) {
    if (!s.empty())
      s += " + ";
    s += fmtPrecPct(t, pct);
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
  if (O.Model.FreqWeightedPricing) {
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
  static const std::unordered_set<std::string> kEmptyScalars;
  const std::unordered_set<std::string> &hwScalar =
      gpuMode ? getScalarTypes(S.CostModel, F) : kEmptyScalars;
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

      const std::string tiers = fmtTiers({{tp.hi, pct}, {tp.lo, 100 - pct}});
      if (O.Diagnostics.Print) {
        llvm::errs() << "Created " << label
                     << " two-tier PT candidate: " << tiers << " (N=" << N
                     << ")\n";
      }
      std::string desc = label.str();
      if (!desc.empty())
        desc += " ";
      desc += tiers;

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

        const std::string tiers = fmtTiers({{tr.hi, pctHi},
                                            {tr.mid, pctHiMid - pctHi},
                                            {tr.lo, 100 - pctHiMid}});
        if (O.Diagnostics.Print) {
          llvm::errs() << "Created " << label
                       << " three-tier PT candidate: " << tiers << " (N=" << N
                       << ")\n";
        }
        std::string desc = label.str();
        if (!desc.empty())
          desc += " ";
        desc += tiers;

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

bool PrecisionUnit::RepriceKey::operator==(const RepriceKey &other) const {
  return candidateIndex == other.candidateIndex &&
         RewriteUnits == other.RewriteUnits;
}

std::size_t
PrecisionUnit::RepriceKeyHash::operator()(const RepriceKey &key) const {
  std::size_t seed = std::hash<size_t>{}(key.candidateIndex);
  for (const auto *ao : key.RewriteUnits) {
    seed ^= std::hash<const RewriteUnit *>{}(ao) + 0x9e3779b9 + (seed << 6) +
            (seed >> 2);
  }
  return seed;
}

void PrecisionUnit::apply(Session &S, size_t candidateIndex) {
  if (candidateIndex >= candidates.size()) {
    llvm_unreachable("Invalid candidate index");
  }

  // An operation this change erases may be an input of another subgraph whose
  // rewrite a later step materializes: the rewrite reads the value the
  // operation's users now read, as the units were priced.
  candidates[candidateIndex].apply(S, *subgraph, nullptr, &graph());
}

// Lower is better
double PrecisionUnit::getCompCostDelta(size_t candidateIndex) {
  // TODO: adjust this based on erased instructions
  return (candidates[candidateIndex].CompCost - initialCompCost) *
         (double)executions;
}

// Lower is better
double PrecisionUnit::getAccCostDelta(size_t candidateIndex) {
  return candidates[candidateIndex].accuracyCost - initialAccCost;
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
  std::unordered_set<RepriceKey, RepriceKeyHash> requested;
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

double PrecisionUnit::getAdjustedCompCostDelta(Session &S,
                                               size_t candidateIndex,
                                               ArrayRef<SolutionStep> steps) {
  priceRequests(S, {{candidateIndex, steps}});
  return compCostDeltaCache.at(repriceKey(candidateIndex, steps));
}

void PrecisionUnit::priceAhead(ArrayRef<ArrayRef<SolutionStep>> priors) {
  SmallVector<std::pair<size_t, ArrayRef<SolutionStep>>, 0> requests;
  for (ArrayRef<SolutionStep> steps : priors)
    for (size_t i = 0; i < candidates.size(); ++i)
      requests.push_back({i, steps});
  priceRequests(session(), requests);
}

double PrecisionUnit::getAdjustedAccCostDelta(
    size_t candidateIndex, ArrayRef<SolutionStep> steps,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap) {
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

double PrecisionUnit::costDelta(size_t i, ArrayRef<SolutionStep> prior) {
  return getAdjustedCompCostDelta(session(), i, prior);
}

double PrecisionUnit::errorDelta(size_t i, ArrayRef<SolutionStep> prior) {
  return getAdjustedAccCostDelta(i, prior, graph().valueToNodeMap,
                                 graph().symbolToValueMap);
}

bool PrecisionUnit::pricedAgainst(const OptimizationUnit &earlier) const {
  const auto *co = dyn_cast<RewriteUnit>(&earlier);
  return co && co->subgraph == subgraph;
}

const SetVector<Instruction *> *PrecisionUnit::footprint() const {
  return subgraph ? &subgraph->operations : nullptr;
}

std::string PrecisionUnit::choiceText(size_t i) const {
  const PTCandidate &cand = candidates[i];
  return "CS: " + cand.desc + " (#" + std::to_string(cand.ordinal) + ")";
}

json::Object PrecisionUnit::stepJSON(size_t i) {
  auto &cand = candidates[i];
  json::Object stepObj;
  stepObj["type"] = "precision_change";
  stepObj["description"] = cand.desc;
  stepObj["candidate_index"] = static_cast<int64_t>(cand.ordinal);

  json::Array changes;
  for (const auto &change : cand.changes) {
    json::Object changeObj;
    changeObj["from"] = getPrecisionChangeTypeString(change.oldType);
    changeObj["to"] = getPrecisionChangeTypeString(change.newType);
    changeObj["num_operations"] = static_cast<int64_t>(change.nodes.size());

    SmallVector<Instruction *, 8> insts;
    json::Array nodeIR;
    for (auto *node : change.nodes) {
      if (auto *I = dyn_cast<Instruction>(node->value)) {
        insts.push_back(I);
        nodeIR.push_back(getValueStr(I));
      }
    }
    changeObj["source_locations"] = getSourceLocationsJSON(insts);
    changeObj["affected_instructions"] = std::move(nodeIR);
    changes.push_back(std::move(changeObj));
  }
  stepObj["changes"] = std::move(changes);
  return stepObj;
}

void PrecisionUnit::writeReportStep(raw_ostream &OS, size_t i, StringRef pad) {
  auto &cand = candidates[i];
  OS << pad << "[Precision] " << cand.desc << " (#" << cand.ordinal << ")\n";
  for (const auto &change : cand.changes) {
    OS << pad << "  " << getPrecisionChangeTypeString(change.oldType) << " -> "
       << getPrecisionChangeTypeString(change.newType) << " for "
       << change.nodes.size() << " operations\n";
    SmallSet<std::string, 4> seen;
    for (auto *node : change.nodes) {
      if (auto *I = dyn_cast<Instruction>(node->value)) {
        auto loc = getSourceLocationStr(I);
        if (!loc.empty() && seen.insert(loc).second)
          OS << pad << "    Source: " << loc << '\n';
      }
    }
  }
}

void PrecisionUnit::materialize(size_t i) { apply(session(), i); }

void PrecisionUnit::price() {
  for (PTCandidate &c : candidates)
    c.CompCost = getCompCost(session(), *subgraph, c);
  setUnifiedAccuracyCost(session().Opts, *this, graph().valueToNodeMap,
                         graph().symbolToValueMap);
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

// The candidates per unit a report lists, cheapest first.
static constexpr size_t kMaxCandRows = 40;

// The candidates' positions by cost change, cheapest first; equal changes keep
// their positions' order.
static SmallVector<size_t, 40> byCostDelta(PrecisionUnit &CS) {
  SmallVector<size_t, 40> order(CS.candidates.size());
  for (size_t ci = 0; ci < order.size(); ++ci)
    order[ci] = ci;
  llvm::stable_sort(order, [&](size_t a, size_t b) {
    return CS.getCompCostDelta(a) < CS.getCompCostDelta(b);
  });
  return order;
}

void PrecisionUnit::addReportSummary(json::Array &summary) {
  json::Object cs;
  cs["num_candidates"] = static_cast<int64_t>(candidates.size());
  cs["initial_accuracy_cost"] = jsonFloat(initialAccCost);
  cs["initial_computation_cost"] = initialCompCost;
  SmallVector<size_t, 40> order = byCostDelta(*this);
  json::Array cands;
  for (size_t k = 0; k < order.size() && k < kMaxCandRows; ++k) {
    size_t ci = order[k];
    json::Object c;
    c["id"] = candidateId(ci);
    c["description"] = candidates[ci].desc;
    c["computation_cost_delta"] = getCompCostDelta(ci);
    c["accuracy_cost_delta"] = jsonFloat(getAccCostDelta(ci));
    cands.push_back(std::move(c));
  }
  cs["candidates"] = std::move(cands);
  summary.push_back(std::move(cs));
}

void PrecisionUnit::writeReportSummary(raw_ostream &textOut) {
  textOut << "Subgraph #" << indexInKind() << ": " << candidates.size()
          << " candidates (Δcost, Δacc; sorted by Δcost)\n";
  SmallVector<size_t, 40> order = byCostDelta(*this);
  for (size_t k = 0; k < order.size() && k < kMaxCandRows; ++k) {
    size_t ci = order[k];
    textOut << "  [" << candidateId(ci) << "] " << candidates[ci].desc
            << ": Δcost=" << getCompCostDelta(ci)
            << " Δacc=" << format("%.6e", getAccCostDelta(ci)) << '\n';
  }
}

void PrecisionUnit::addCuratedRewrites(json::Array &rewrites) {
  for (size_t ci = 0; ci < candidates.size(); ++ci) {
    auto &pt = candidates[ci];
    double accDelta = getAccCostDelta(ci);
    double compDelta = getCompCostDelta(ci);

    if (std::isnan(accDelta))
      continue;

    // Emitted even though dominated: an all-dominated frontier must leave a
    // record of why each candidate lost.
    std::string category = "dominated";
    double efficiency = 0.0;
    if (std::optional<Tradeoff> tradeoff =
            classifyTradeoff(compDelta, accDelta)) {
      category = tradeoff->category.str();
      efficiency = tradeoff->efficiency;
    }

    json::Object rw;
    rw["id"] = candidateId(ci);
    rw["type"] = "precision_change";
    rw["description"] = pt.desc;
    rw["category"] = category;
    rw["efficiency"] = jsonFloat(efficiency);
    rw["computation_cost_delta"] = compDelta;
    rw["accuracy_cost_delta"] = jsonFloat(accDelta);

    SmallVector<Instruction *, 16> ptInsts;
    for (const auto &change : pt.changes) {
      for (auto *node : change.nodes) {
        if (auto *I = dyn_cast<Instruction>(node->value))
          ptInsts.push_back(I);
      }
    }
    rw["source_locations"] = getSourceLocationsJSON(ptInsts);
    json::Array affectedIR;
    for (auto *I : ptInsts)
      affectedIR.push_back(getValueStr(I));
    rw["affected_instructions"] = std::move(affectedIR);

    rewrites.push_back(std::move(rw));
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
