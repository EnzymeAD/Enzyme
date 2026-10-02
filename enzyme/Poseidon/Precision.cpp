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
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"

#include <cmath>
#include <string>
#include <unordered_map>

#include "CostModel.h"
#include "Evaluators.h"
#include "Expansion.h"
#include "Flags.h"
#include "Optimize.h"
#include "Precision.h"
#include "Sampling.h"
#include "Staging.h"
#include "Types.h"
#include "Utils.h"

using namespace llvm;

namespace poseidon {

const char *fpKindName(FPKind k) {
  switch (k) {
  case FPKind::F16:
    return "f16";
  case FPKind::BF16:
    return "bf16";
  case FPKind::TF32:
    return "tf32";
  case FPKind::F32:
    return "f32";
  case FPKind::F64:
    return "f64";
  case FPKind::S8:
    return "s8";
  case FPKind::S32:
    return "s32";
  case FPKind::Invalid:
    return "invalid";
  }
  llvm_unreachable("unknown FPKind value");
}

FPKind fpKindFromType(Type *T) {
  if (T->isHalfTy())
    return FPKind::F16;
  if (T->isBFloatTy())
    return FPKind::BF16;
  if (T->isFloatTy())
    return FPKind::F32;
  if (T->isDoubleTy())
    return FPKind::F64;
  return FPKind::Invalid;
}

// nullptr for kinds with no FP rounding semantics (F64, Invalid, S8/S32).
static const fltSemantics *semanticsForKind(FPKind k) {
  switch (k) {
  case FPKind::F16:
    return &APFloat::IEEEhalf();
  case FPKind::BF16:
    return &APFloat::BFloat();
  case FPKind::TF32:
    return &APFloat::FloatTF32();
  case FPKind::F32:
    return &APFloat::IEEEsingle();
  case FPKind::F64:
  case FPKind::Invalid:
  case FPKind::S8:
  case FPKind::S32:
    return nullptr;
  }
  return nullptr;
}

double roundToPrec(double x, FPKind k) {
  if (k == FPKind::S8 || k == FPKind::S32)
    report_fatal_error(
        "roundToPrec: integer FPKind (S8/S32) has no FP rounding semantics");
  const fltSemantics *sem = semanticsForKind(k);
  if (!sem)
    return x;
  bool losesInfo;
  APFloat ap(x);
  ap.convert(*sem, APFloat::rmNearestTiesToEven, &losesInfo);
  ap.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven, &losesInfo);
  return ap.convertToDouble();
}

double minSubnormalForKind(FPKind k) {
  const fltSemantics *sem = semanticsForKind(k);
  if (!sem)
    return 0.0;
  APFloat sm = APFloat::getSmallest(*sem);
  bool li;
  sm.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven, &li);
  return sm.convertToDouble();
}

unsigned getMPFRPrec(PrecisionChangeType type) {
  switch (type) {
  case PrecisionChangeType::BF16:
    return 8;
  case PrecisionChangeType::FP16:
    return 11;
  case PrecisionChangeType::FP32:
    return 24;
  case PrecisionChangeType::FP64:
    return 53;
  case PrecisionChangeType::FP80:
    return 64;
  case PrecisionChangeType::FP128:
    return 113;
  default:
    llvm_unreachable("Unsupported FP precision");
  }
}

Type *getLLVMFPType(PrecisionChangeType type, LLVMContext &context) {
  switch (type) {
  case PrecisionChangeType::BF16:
    return Type::getBFloatTy(context);
  case PrecisionChangeType::FP16:
    return Type::getHalfTy(context);
  case PrecisionChangeType::FP32:
    return Type::getFloatTy(context);
  case PrecisionChangeType::FP64:
    return Type::getDoubleTy(context);
  case PrecisionChangeType::FP80:
    return Type::getX86_FP80Ty(context);
  case PrecisionChangeType::FP128:
    return Type::getFP128Ty(context);
  default:
    llvm_unreachable("Unsupported FP precision");
  }
}

PrecisionChangeType getPrecisionChangeType(Type *type) {
  if (type->isHalfTy()) {
    return PrecisionChangeType::BF16;
  } else if (type->isHalfTy()) {
    return PrecisionChangeType::FP16;
  } else if (type->isFloatTy()) {
    return PrecisionChangeType::FP32;
  } else if (type->isDoubleTy()) {
    return PrecisionChangeType::FP64;
  } else if (type->isX86_FP80Ty()) {
    return PrecisionChangeType::FP80;
  } else if (type->isFP128Ty()) {
    return PrecisionChangeType::FP128;
  } else {
    llvm_unreachable("Unsupported FP precision");
  }
}

StringRef getPrecisionChangeTypeString(PrecisionChangeType type) {
  switch (type) {
  case PrecisionChangeType::BF16:
    return "BF16";
  case PrecisionChangeType::FP16:
    return "FP16";
  case PrecisionChangeType::FP32:
    return "FP32";
  case PrecisionChangeType::FP64:
    return "FP64";
  case PrecisionChangeType::FP80:
    return "FP80";
  case PrecisionChangeType::FP128:
    return "FP128";
  case PrecisionChangeType::Expansion2:
    return "Expansion2";
  case PrecisionChangeType::Expansion3:
    return "Expansion3";
  case PrecisionChangeType::Expansion4:
    return "Expansion4";
  default:
    return "Unknown PT type";
  }
}

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

      // Lower F32-downcast sqrt on NVPTX to nvvm.sqrt.approx.f (MUFU,
      // inlined) instead of a correctly-rounded __nv_sqrtf call: the lossy
      // F32 tier tolerates the ~2^-22 approx error, and a per-evaluation
      // function call would dominate sqrt-heavy kernels.
      {
        std::string base = funcName.str();
        if (!base.empty() && (base.back() == 'f' || base.back() == 'l'))
          base.pop_back();
        bool isSqrt = (base == "sqrt" || base == "__nv_sqrt");
        if (isSqrt && newType->isFloatTy() &&
            llvm::Triple(CI->getModule()->getTargetTriple()).isNVPTX() &&
            newArgs.size() == 1) {
          Function *fastSqrt = Intrinsic::getOrInsertDeclaration(
              CI->getModule(), Intrinsic::nvvm_sqrt_approx_f, {});
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
        std::string base = funcName.str();
        if (base.size() > 5 && base.compare(0, 5, "__nv_") == 0)
          base = base.substr(5);
        if (!base.empty() && (base.back() == 'f' || base.back() == 'l'))
          base.pop_back();
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
          Value *callResult = Builder.CreateCall(newFunc, callArgs);
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
// values and change outputs in VMap to new casted outputs.
void PTCandidate::apply(Subgraph &subgraph, ValueToValueMapTy *VMap) {
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
  resetExpLimbs();

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
        applyExpansion(instsToChangeSorted, allChangedSet, &restoredValues);
      else
        applyExpansion(nComp, instsToChangeSorted, allChangedSet,
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

void setUnifiedAccuracyCost(
    CandidateSubgraph &CS,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap) {

  CancellationPlan cancelPlan =
      buildCancellationPlan(*CS.subgraph, valueToNodeMap);
  SmallVector<MapVector<Value *, double>, 4> sampledPoints;
  getSampledPoints(CS.subgraph->inputs.getArrayRef(), valueToNodeMap,
                   symbolToValueMap, sampledPoints, &cancelPlan);

  MapVector<FPNode *, SmallVector<double, 4>> goldVals;
  for (auto *output : CS.subgraph->outputs) {
    auto *node = valueToNodeMap[output].get();
    goldVals[node].resize(flags::NumSamples);
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
      getFPValues(outputs, pair.value(), results, &candidate);
      for (const auto &[node, result] : zip(outputs, results)) {
        double goldVal = goldVals[node][pair.index()];
        if (flags::StrictMode && !std::isnan(goldVal) &&
            !std::isfinite(result)) {
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
    }
  }
  CS.candidates = std::move(newCandidates);
}

double getCompCost(Subgraph &subgraph, PTCandidate &pt) {
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
  pt.apply(subgraph, &VMap);
  if (ptHasExpansion) {
    // A tier pair can apply an expansion change and a double-single change,
    // each recording its own roots.
    takeLastExpansionLimbs(expansionLimbs);
    takeLastExpLimbs(expansionLimbs);
  }
  // An Expansion2 change that converts nothing (unsupported instructions)
  // leaves no limbs, and the untouched FP64 outputs are still the right roots.
  const bool expansionMaterialized = !expansionLimbs.empty();

  // Mirror the real materialization's cleanups so the price charges only the
  // casts the materializer emits. demoteFPCastPHIs RAUWs each demoted PHI with
  // fpext(new PHI) before erasing it, and the WeakTrackingVH handles follow.
  applyStagingNarrowing(*FClone, /*announce=*/false);
  // Cancel the join/split roundtrip the df64 conversion leaves on an expansion
  // restore, as applyExpansion does in the emitted code; no-op without the
  // poseidon.ds.join tag.
  SmallVector<Value *, 8> dsFoldRoots;
  foldDSPairRoundtrip(*FClone, &dsFoldRoots);

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
  if (flags::Print && ptHasExpansion && !expansionMaterialized)
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

  SmallPtrSet<Value *, 8> clonedOutputs;
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

  for (Instruction *I : coneInsts) {
    double c = getInstructionCompCost(I);
    unsigned d = cloneLI.getLoopDepth(I->getParent());
    if ((isa<FPExtInst>(I) || isa<FPTruncInst>(I)) && bodyDepth > 0 &&
        d < bodyDepth) {
      c *= subgraph.outBoundaryFreqScale;
    }
    cost += c;
  }

  FClone->eraseFromParent();

  return cost;
}

} // namespace poseidon
