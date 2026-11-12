//=- Precision.cpp - Precision change utilities for Poseidon --------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities for handling precision changes in the Poseidon
// optimization pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"

#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "Expansion.h"
#include "Herbie.h"
#include "IR/Utils.h"
#include "Model/CostModel.h"
#include "Model/Evaluators.h"
#include "Model/Sampling.h"
#include "Pass/Session.h"
#include "Precision.h"
#include "Profile/ProfileRead.h"
#include "Sites/Canonicalize.h"
#include "Staging.h"
#include "Targets/DeviceMath.h"
#include "Targets/Target.h"

using namespace llvm;

namespace poseidon {

void changePrecision(Instruction *I, PrecisionChange &change,
                     MapVector<Value *, Value *> &oldToNew) {
  if (!isOptimizable(*I)) {
    llvm_unreachable("Trying to tune an instruction is not isOptimizable");
  }

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
        llvm::errs() << "PT: Unknown intrinsic: " << *CI << "\n";
        llvm_unreachable("changePrecision: Unknown intrinsic call to change");
      }
    } else {
      StringRef funcName = calledFunc->getName();

      // An F32-downcast sqrt becomes the target's approximate square root
      // where it has one, instead of a correctly-rounded library call: the
      // lossy F32 tier tolerates the approximation, and a per-evaluation
      // function call would dominate sqrt-heavy kernels.
      {
        const Target &T = Target::of(*CI->getModule());
        std::string base = funcName.str();
        if (!base.empty() && (base.back() == 'f' || base.back() == 'l'))
          base.pop_back();
        const std::string librarySqrt =
            T.mathFunction("sqrt", Type::getDoubleTy(CI->getContext()));
        bool isSqrt = (base == "sqrt" || base == librarySqrt);
        Intrinsic::ID approxSqrt = T.approxSqrtF32();
        if (isSqrt && newType->isFloatTy() &&
            approxSqrt != Intrinsic::not_intrinsic && newArgs.size() == 1) {
          Function *fastSqrt = Intrinsic::getOrInsertDeclaration(
              CI->getModule(), approxSqrt, {});
          newI = Builder.CreateCall(fastSqrt, {newArgs[0]});
          oldToNew[I] = newI;
          return;
        }
      }

      // fmax / fmin are not in libmFuncs(): getPTFuncs() prices every name in
      // that list, so adding them there would abort every build whose cost
      // model lacks the row. Retype them to the intrinsics clang lowers
      // __builtin_fmax / __builtin_fmin to instead, which have the same
      // IEEE-754 semantics, are already modelled, and need no declaration that
      // ptxas would then have to resolve (libdevice is linked LinkOnlyNeeded
      // before this pass runs, so a fresh __nv_fmaxf would reach ptxas
      // undefined).
      {
        std::string base = deviceMathName(funcName).str();
        if (base.empty()) {
          base = funcName.str();
          if (!base.empty() && (base.back() == 'f' || base.back() == 'l'))
            base.pop_back();
        }
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
                       << " libm function for: " << *CI << "\n";
          llvm_unreachable("changePrecision: Failed to get libm function");
        }
      } else {
        llvm::errs() << "PT: Unknown function call: " << *CI << "\n";
        llvm_unreachable("changePrecision: Unknown function call to change");
      }
    }

  } else {
    llvm::errs() << "Unexpectedly isOptimizable instruction: " << *I << "\n";
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
      assert(VMap->count(I));
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
  // changes (e.g. a Expansion4 tier over a Expansion3 one) and getCompCost
  // must see every root they produce.
  resetExpLimbs(S);

  for (auto &change : changes) {
    SmallPtrSet<Instruction *, 8> seen;
    SmallVector<Instruction *, 8> todo;
    MapVector<Value *, Value *> oldToNew;

    SetVector<Instruction *> instsToChange;
    for (auto node : change.nodes) {
      if (!node || !node->value) {
        continue;
      }
      assert(isa<Instruction>(node->value));
      auto *I = cast<Instruction>(node->value);
      if (VMap) {
        assert(VMap->count(I));
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
        // Already erased by `CO.apply()`.
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
        IRBuilder<> builder(cast<Instruction>(oldV)->getParent(),
                            ++BasicBlock::iterator(cast<Instruction>(oldV)));
        casted = builder.CreateFPCast(
            newV, getLLVMFPType(change.oldType, builder.getContext()));

        if (VMap) {
          assert(VMap->count(clonedToOriginal[oldV]));
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

  // Confirm the closure invariant holds for every surviving PHI before mutating
  // anything; bail cleanly rather than leave a half-rewritten function.
  for (PHINode *P : D)
    for (Value *IV : P->incoming_values()) {
      Constant *c = nullptr;
      if (!(inSet.count(IV) || floatExtSrc(IV) || floatReprConst(IV, c)))
        return false;
    }

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
    IRBuilder<> B(BB, BB->getFirstInsertionPt());
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

  CancellationPlan cancelPlan =
      buildCancellationPlan(O, *CS.subgraph, valueToNodeMap);
  SmallVector<MapVector<Value *, double>, 4> sampledPoints;
  getSampledPoints(O, CS.subgraph->inputs.getArrayRef(), valueToNodeMap,
                   symbolToValueMap, sampledPoints, &cancelPlan);

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

  for (const auto &pair : enumerate(sampledPoints)) {
    SmallVector<double, 8> results;
    getMPFRValues(outputs, pair.value(), results, true, 53);
    for (const auto &[node, result] : zip(outputs, results))
      goldVals[node][pair.index()] = result;

    getFPValues(outputs, pair.value(), results);
    for (const auto &[node, result] : zip(outputs, results)) {
      double goldVal = goldVals[node][pair.index()];
      double error = sampleError(goldVal, result);
      if (!std::isnan(error)) {
        runAcc[node].sum += error;
        ++runAcc[node].count;
      }
    }
  }
  CS.initialAccCost = 0.0;
  for (auto *node : outputs) {
    auto &ra = runAcc[node];
    assert(ra.count != 0 && "No valid sample found for original subgraph");
    double red = ra.sum / ra.count;
    CS.perOutputInitialAccCost[node] = red * std::fabs(node->grad);
    CS.initialAccCost += CS.perOutputInitialAccCost[node];
  }
  assert(!std::isnan(CS.initialAccCost));

  SmallVector<PTCandidate, 4> newCandidates;
  for (auto &candidate : CS.candidates) {
    bool discardCandidate = false;
    struct RunningAccArith {
      double sum = 0.0;
      unsigned count = 0;
    };
    std::unordered_map<FPNode *, RunningAccArith> candAcc;
    for (auto *node : outputs)
      candAcc[node] = RunningAccArith();

    for (const auto &pair : enumerate(sampledPoints)) {
      SmallVector<double, 8> results;
      getFPValues(outputs, pair.value(), results, candidate.changes);
      for (const auto &[node, result] : zip(outputs, results)) {
        double goldVal = goldVals[node][pair.index()];
        if (std::isfinite(goldVal) && !std::isfinite(result)) {
          discardCandidate = true;
          break;
        }
        double error = sampleError(goldVal, result);
        if (!std::isnan(error)) {
          candAcc[node].sum += error;
          ++candAcc[node].count;
        }
      }
      if (discardCandidate)
        break;
    }
    if (!discardCandidate) {
      candidate.accuracyCost = 0.0;
      for (auto *node : outputs) {
        auto &ra = candAcc[node];
        assert(ra.count != 0 && "No valid sample found for candidate subgraph");
        double red = ra.sum / ra.count;
        candidate.perOutputAccCost[node] = red * std::fabs(node->grad);
        candidate.accuracyCost += candidate.perOutputAccCost[node];
      }
      assert(!std::isnan(candidate.accuracyCost));
      newCandidates.push_back(std::move(candidate));
    } else {
      CS.dropped.push_back({candidate.desc,
                            "it returns a non-finite value at a sampled input "
                            "where the exact result is finite"});
    }
  }
  CS.candidates = std::move(newCandidates);
}

double getCompCost(Session &S, Subgraph &subgraph, PTCandidate &pt) {
  const Options &O = S.Opts;
  assert(!subgraph.outputs.empty());

  double cost = 0.0;

  Function *F = cast<Instruction>(subgraph.outputs[0])->getFunction();

  ValueToValueMapTy VMap;
  Function *FClone = CloneFunction(F, VMap);
  FClone->setName(F->getName() + "_clone");

  // Priced with the materializer's own carried-PHI elimination: the emitted
  // loop carries the {hi,lo} pair across the back edge and collapses it once at
  // the exit, so the df64 restore must not be billed per iteration.
  // applyExpansion records the limbs it produced and the walk is re-rooted on
  // them below.
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

  // Mirror the real materialization's cleanups so the price charges only the
  // casts the materializer emits. demoteFPCastPHIs RAUWs each demoted PHI with
  // fpext(new PHI) before erasing it, and the WeakTrackingVH handles follow.
  // FClone has no call sites, so F supplies the parameter-to-global mapping
  // (the arguments are positionally identical). A df64 candidate wants the pair
  // layout and anything else the halved one, so exactly one parameter arm runs.
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
  // Cancel the join/split roundtrip the df64 conversion leaves on an expansion
  // restore, as applyExpansion does in the emitted code; no-op without the
  // poseidon.ds.join tag.
  SmallVector<Value *, 8> dsFoldRoots;
  foldDSPairRoundtrip(*FClone, &dsFoldRoots, O.Diagnostics.Print);

  SmallVector<WeakTrackingVH, 8> trackedInputs;
  for (auto &input : subgraph.inputs) {
    if (VMap.count(input))
      trackedInputs.emplace_back(VMap[input]);
  }
  SmallVector<WeakTrackingVH, 8> trackedOutputs;
  for (auto &output : subgraph.outputs) {
    if (VMap.count(output))
      trackedOutputs.emplace_back(VMap[output]);
  }
  // Re-root on the df64 limbs from applyExpansion and from foldDSPairRoundtrip:
  // they are the unit's outputs in the emitted code and the F64 handles above
  // are dead, so without this the candidate walks an empty cone and prices as
  // free.
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
  SmallVector<WeakTrackingVH, 8> trackedOutputFallbacks(trackedOutputs.size());
  for (size_t i = 0; i < trackedOutputs.size(); ++i)
    if (trackedOutputs[i].pointsToAliveValue())
      if (auto *FE = dyn_cast_or_null<FPExtInst>((Value *)trackedOutputs[i]))
        trackedOutputFallbacks[i] = FE->getOperand(0);

  simplifyFunction(*FClone, OptimizationLevel::O3);

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
  SmallSetVector<Value *, 8> clonedOutputs;
  for (size_t i = 0; i < trackedOutputs.size(); ++i) {
    Value *V = nullptr;
    if (trackedOutputs[i].pointsToAliveValue())
      V = trackedOutputs[i];
    else if (i < trackedOutputFallbacks.size() &&
             trackedOutputFallbacks[i].pointsToAliveValue())
      V = trackedOutputFallbacks[i];
    if (V)
      clonedOutputs.insert(V);
  }

  // Boundary casts the O3'd clone hoists out of the cone's body loop execute at
  // most once per thread per launch, so casts at a shallower loop depth than
  // the body get outBoundaryFreqScale; casts inside the body keep full
  // frequency.

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

  // Body-loop depth = the deepest loop hosting the cone's non-cast FP work.
  DominatorTree cloneDT(*FClone);
  LoopInfo cloneLI(cloneDT);
  unsigned bodyDepth = 0;
  for (Instruction *I : coneInsts)
    if (!isa<FPExtInst>(I) && !isa<FPTruncInst>(I))
      bodyDepth = std::max(bodyDepth, cloneLI.getLoopDepth(I->getParent()));

  const bool freqWeighted =
      O.Model.FreqWeightedPricing && subgraph.depthExec.size() > 1;
  for (Instruction *I : coneInsts) {
    double c = getInstructionCompCost(S.CostModel, I);
    unsigned d = cloneLI.getLoopDepth(I->getParent());
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

  FClone->eraseFromParent();

  return cost;
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

static bool tierPairAllowed(const Options &O, TierPair tp, bool gpuMode,
                            bool expansionSafe,
                            const std::unordered_set<std::string> &hwScalar) {
  return precAllowed(O, tp.hi, gpuMode, expansionSafe, hwScalar) &&
         precAllowed(O, tp.lo, gpuMode, expansionSafe, hwScalar);
}

static bool tierTripleAllowed(const Options &O, TierTriple tr, bool gpuMode,
                              bool expansionSafe,
                              const std::unordered_set<std::string> &hwScalar) {
  return precAllowed(O, tr.hi, gpuMode, expansionSafe, hwScalar) &&
         precAllowed(O, tr.mid, gpuMode, expansionSafe, hwScalar) &&
         precAllowed(O, tr.lo, gpuMode, expansionSafe, hwScalar);
}

static std::string fmtPrecPct(PrecisionChangeType t, int pct) {
  std::string s = getPrecisionChangeTypeString(t).str();
  s += "(";
  s += std::to_string(pct);
  s += "%)";
  return s;
}

} // namespace

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
        llvm::errs() << " " << kv.first << ":" << kv.second;
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
      llvm::errs() << "[poseidon] PT Function identified: " << *I << "\n";
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

      if (O.Diagnostics.Print) {
        llvm::errs() << "Created " << label
                     << " two-tier PT candidate: " << fmtPrecPct(tp.hi, pct)
                     << " + " << fmtPrecPct(tp.lo, 100 - pct) << " (N=" << N
                     << ")\n";
      }
      std::string desc = label.str();
      if (!desc.empty())
        desc += " ";
      desc += fmtPrecPct(tp.hi, pct);
      desc += " + ";
      desc += fmtPrecPct(tp.lo, 100 - pct);

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

        if (O.Diagnostics.Print) {
          llvm::errs() << "Created " << label << " three-tier PT candidate: "
                       << fmtPrecPct(tr.hi, pctHi) << " + "
                       << fmtPrecPct(tr.mid, pctHiMid - pctHi) << " + "
                       << fmtPrecPct(tr.lo, 100 - pctHiMid) << " (N=" << N
                       << ")\n";
        }
        std::string desc = label.str();
        if (!desc.empty())
          desc += " ";
        desc += fmtPrecPct(tr.hi, pctHi);
        desc += " + ";
        desc += fmtPrecPct(tr.mid, pctHiMid - pctHi);
        desc += " + ";
        desc += fmtPrecPct(tr.lo, 100 - pctHiMid);

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
    if (!tierPairAllowed(O, tp, gpuMode, expansionSafe, hwScalar))
      continue;
    sweepTwoTier(tp, sortedAllOps, "All");
    if (!sortedFuncs.empty())
      sweepTwoTier(tp, sortedFuncs, "Funcs");
  }

  if (O.Candidates.EnableThreeTier) {
    for (TierTriple tr : kCanonicalTriples) {
      if (!tierTripleAllowed(O, tr, gpuMode, expansionSafe, hwScalar))
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
  // TODO: adjust this based on erasured instructions
  return (candidates[candidateIndex].CompCost - initialCompCost) *
         (double)executions;
}

// Lower is better
double PrecisionUnit::getAccCostDelta(size_t candidateIndex) {
  return candidates[candidateIndex].accuracyCost - initialAccCost;
}

double PrecisionUnit::getAdjustedCompCostDelta(Session &S,
                                               size_t candidateIndex,
                                               ArrayRef<SolutionStep> steps) {
  RewriteUnitSet RewriteUnits;
  for (const auto &step : steps) {
    if (auto *co = dyn_cast<RewriteUnit>(step.unit)) {
      if (co->subgraph == subgraph) {
        RewriteUnits.insert(co);
      }
    }
  }

  RepriceKey key{candidateIndex, RewriteUnits};

  auto cacheIt = compCostDeltaCache.find(key);
  if (cacheIt != compCostDeltaCache.end()) {
    return cacheIt->second;
  }

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
    return 0;
  }

  double initialCompCost = getCompCost(
      S.CostModel, {newSubgraph.outputs.begin(), newSubgraph.outputs.end()},
      newSubgraph.inputs, &newSubgraph.opExec, newSubgraph.execNormalizer);

  double candidateCompCost =
      getCompCost(S, newSubgraph, candidates[candidateIndex]);

  double adjustedCostDelta =
      (candidateCompCost - initialCompCost) * (double)executions;
  // Per-candidate pricing audit: with a one-point Pareto table these are the
  // ONLY record of why each candidate lost (all-dominated frontiers must be
  // auditable, not silent).
  if (S.Opts.Diagnostics.Print)
    llvm::errs() << "CS candidate '" << candidates[candidateIndex].desc
                 << "': initial=" << initialCompCost
                 << " cand=" << candidateCompCost << " exec=" << executions
                 << " Δcost=" << adjustedCostDelta
                 << " Δacc=" << getAccCostDelta(candidateIndex) << "\n";

  compCostDeltaCache[key] = adjustedCostDelta;
  return adjustedCostDelta;
}

double PrecisionUnit::getAdjustedAccCostDelta(
    size_t candidateIndex, ArrayRef<SolutionStep> steps,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap) {
  RewriteUnitSet RewriteUnits;
  for (const auto &step : steps) {
    if (auto *co = dyn_cast<RewriteUnit>(step.unit)) {
      if (co->subgraph == subgraph) {
        RewriteUnits.insert(co);
      }
    }
  }

  RepriceKey key{candidateIndex, RewriteUnits};

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
          assert(it != valueToNodeMap.end() && it->second);
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
          OS << pad << "    Source: " << loc << "\n";
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

// Top-N per-candidate deltas keep an all-dominated frontier auditable (the
// table then collapses to the no-op point).
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
            << " Δacc=" << format("%.6e", getAccCostDelta(ci)) << "\n";
  }
}

void PrecisionUnit::addCuratedRewrites(json::Array &rewrites) {
  for (size_t ci = 0; ci < candidates.size(); ++ci) {
    auto &pt = candidates[ci];
    double accDelta = getAccCostDelta(ci);
    auto compDelta = getCompCostDelta(ci);
    double compDeltaVal = compDelta;

    if (std::isnan(accDelta))
      continue;

    // Emitted even though dominated: an all-dominated frontier must leave a
    // record of why each candidate lost.
    std::string category = "dominated";
    double efficiency = 0.0;
    if (std::optional<Tradeoff> tradeoff =
            classifyTradeoff(compDeltaVal, accDelta)) {
      category = tradeoff->category.str();
      efficiency = tradeoff->efficiency;
    }

    json::Object rw;
    rw["id"] = candidateId(ci);
    rw["type"] = "precision_change";
    rw["description"] = pt.desc;
    rw["category"] = category;
    rw["efficiency"] = jsonFloat(efficiency);
    rw["computation_cost_delta"] = compDeltaVal;
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
  OS << "Initial AccuracyCost: " << initialAccCost << "\n";
  OS << "Initial ComputationCost: " << initialCompCost << "\n";
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
    OS << "\t\t" << candidate.desc << "\n";
  }
  OS << "################################\n\n";
}

} // namespace poseidon
