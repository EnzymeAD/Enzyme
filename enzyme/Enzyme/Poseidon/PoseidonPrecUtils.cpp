//=- PoseidonPrecUtils.cpp - Precision change utilities for Poseidon ------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities for handling precision changes in the Poseidon
// optimization pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <cmath>
#include <string>
#include <unordered_map>

#include "../Utils.h"
#include "Poseidon.h"
#include "PoseidonEvaluators.h"
#include "PoseidonPrecUtils.h"
#include "PoseidonTypes.h"
#include "PoseidonUtils.h"

using namespace llvm;

extern "C" {
cl::opt<bool> FPOptShowPTDetails(
    "fpopt-show-pt-details", cl::init(false), cl::Hidden,
    cl::desc("Print details of precision tuning candidates along with the DP "
             "table (highly verbose for large applications)"));
cl::opt<unsigned>
    FPOptMaxMPFRPrec("fpopt-max-mpfr-prec", cl::init(1024), cl::Hidden,
                     cl::desc("Max precision for MPFR gold value computation"));
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
  default:
    return "Unknown PT type";
  }
}

void changePrecision(Instruction *I, PrecisionChange &change,
                     MapVector<Value *, Value *> &oldToNew) {
  if (!Poseidonable(*I)) {
    llvm_unreachable("Trying to tune an instruction is not Poseidonable");
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
        oldToNew[operand] = newOp;
      } else {
        if (Instruction *opInst = dyn_cast<Instruction>(operand)) {
          IRBuilder<> OpBuilder(opInst->getParent(),
                                ++BasicBlock::iterator(opInst));
          OpBuilder.setFastMathFlags(I->getFastMathFlags());
          newOp = OpBuilder.CreateFPCast(operand, newType, "fpopt.fpcast");
        } else if (Argument *argOp = dyn_cast<Argument>(operand)) {
          BasicBlock &entry = argOp->getParent()->getEntryBlock();
          IRBuilder<> OpBuilder(&*entry.getFirstInsertionPt());
          OpBuilder.setFastMathFlags(I->getFastMathFlags());
          newOp = OpBuilder.CreateFPCast(operand, newType, "fpopt.fpcast");
        } else if (Constant *constOp = dyn_cast<Constant>(operand)) {
          IRBuilder<> ConstBuilder(I);
          ConstBuilder.setFastMathFlags(I->getFastMathFlags());
          newOp = ConstBuilder.CreateFPCast(constOp, newType, "fpopt.const.fpcast");
        } else {
          llvm_unreachable("Unsupported operand type");
        }
        oldToNew[operand] = newOp;
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
        oldToNew[arg] = newArg;
      } else {
        if (Instruction *argInst = dyn_cast<Instruction>(arg)) {
          IRBuilder<> ArgBuilder(argInst->getParent(),
                                 ++BasicBlock::iterator(argInst));
          ArgBuilder.setFastMathFlags(I->getFastMathFlags());
          newArg = ArgBuilder.CreateFPCast(arg, newType, "fpopt.fpcast");
        } else if (Argument *argArg = dyn_cast<Argument>(arg)) {
          BasicBlock &entry = argArg->getParent()->getEntryBlock();
          IRBuilder<> ArgBuilder(&*entry.getFirstInsertionPt());
          ArgBuilder.setFastMathFlags(I->getFastMathFlags());
          newArg = ArgBuilder.CreateFPCast(arg, newType, "fpopt.fpcast");
        } else if (Constant *constArg = dyn_cast<Constant>(arg)) {
          auto srcBits =
              constArg->getType()->getPrimitiveSizeInBits().getFixedValue();
          auto dstBits = newType->getPrimitiveSizeInBits().getFixedValue();

          Instruction::CastOps opcode = (srcBits < dstBits) ? Instruction::FPExt
                                        : (srcBits > dstBits)
                                            ? Instruction::FPTrunc
                                            : Instruction::BitCast;

          newArg = ConstantExpr::getCast(opcode, constArg, newType);
        } else {
          llvm_unreachable("Unsupported argument type");
        }
        oldToNew[arg] = newArg;
      }
      newArgs.push_back(newArg);
    }
    auto *calledFunc = CI->getCalledFunction();
    if (calledFunc && calledFunc->isIntrinsic()) {
      Intrinsic::ID intrinsicID = calledFunc->getIntrinsicID();
      if (intrinsicID != Intrinsic::not_intrinsic) {
        // Special cases for intrinsics with mixed types
        if (intrinsicID == Intrinsic::powi) {
          // powi
          SmallVector<Type *, 2> overloadedTypes;
          overloadedTypes.push_back(newType);
          overloadedTypes.push_back(CI->getArgOperand(1)->getType());
#if LLVM_VERSION_MAJOR >= 21
          Function *newFunc = Intrinsic::getOrInsertDeclaration(
#else
          Function *newFunc = getIntrinsicDeclaration(
#endif
              CI->getModule(), intrinsicID, overloadedTypes);
          newI = Builder.CreateCall(newFunc, newArgs);
        } else {
#if LLVM_VERSION_MAJOR >= 21
          Function *newFunc = Intrinsic::getOrInsertDeclaration(
#else
          Function *newFunc = getIntrinsicDeclaration(
#endif
              CI->getModule(), intrinsicID, newType);
          newI = Builder.CreateCall(newFunc, newArgs);
        }
      } else {
        llvm::errs() << "PT: Unknown intrinsic: " << *CI << "\n";
        llvm_unreachable("changePrecision: Unknown intrinsic call to change");
      }
    } else {
      StringRef funcName = calledFunc->getName();
      std::string newFuncName = getLibmFunctionForPrecision(funcName, newType);

      if (!newFuncName.empty()) {
        Module *M = CI->getModule();
        SmallVector<Type *, 4> newArgTypes(newArgs.size(), newType);

        FunctionCallee newFuncCallee = M->getOrInsertFunction(
            newFuncName, FunctionType::get(newType, newArgTypes, false));

        if (Function *newFunc = dyn_cast<Function>(newFuncCallee.getCallee())) {
          newI = Builder.CreateCall(newFunc, newArgs);
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
    llvm_unreachable("Unexpectedly Poseidonable instruction");
  }

  oldToNew[I] = newI;
}

// If `VMap` is passed, map `llvm::Value`s in `subgraph` to their cloned
// values and change outputs in VMap to new casted outputs.
void PTCandidate::apply(Subgraph &subgraph, ValueToValueMapTy *VMap) {
  SetVector<Instruction *> operations;
  ValueToValueMapTy clonedToOriginal; // Maps cloned outputs to old outputs
  if (VMap) {
    for (auto *I : subgraph.operations) {
      assert(VMap->count(I));
      operations.insert(cast<Instruction>(VMap->lookup(I)));

      clonedToOriginal[VMap->lookup(I)] = I;
      // llvm::errs() << "Mapping back: " << *VMap->lookup(I) << " (in "
      //              << cast<Instruction>(VMap->lookup(I))
      //                     ->getParent()
      //                     ->getParent()
      //                     ->getName()
      //              << ") --> " << *I << " (in "
      //              << I->getParent()->getParent()->getName() << ")\n";
    }
  } else {
    operations = subgraph.operations;
  }

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
        I = cast<Instruction>(VMap->lookup(I));
      }
      if (!operations.contains(I)) {
        // Already erased by `CO.apply()`.
        continue;
      }
      instsToChange.insert(I);
    }

    SmallVector<Instruction *, 8> instsToChangeSorted;
    topoSort(instsToChange, instsToChangeSorted);

    for (auto *I : instsToChangeSorted) {
      changePrecision(I, change, oldToNew);
    }

    // Restore the precisions of the last level of instructions to be changed.
    // Clean up old instructions.
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

      // Assumes no external uses of the old value since all corresponding new
      // values are already restored to original precision and used to replace
      // uses of their old value. This is also advantageous to the solvers.
      for (auto *user : oldV->users()) {
        assert(instsToChange.contains(cast<Instruction>(user)) &&
               "PT: Unexpected external user of a changed instruction");
      }

      if (!oldV->use_empty()) {
        oldV->replaceAllUsesWith(UndefValue::get(oldV->getType()));
      }

      cast<Instruction>(oldV)->eraseFromParent();

      // The change is being materialized to the original subgraph
      if (!VMap)
        subgraph.operations.remove(cast<Instruction>(oldV));
    }
  }
}

void setUnifiedAccuracyCost(
    CandidateSubgraph &CS,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap) {

  SmallVector<MapVector<Value *, double>, 4> sampledPoints;
  getSampledPoints(CS.subgraph->inputs.getArrayRef(), valueToNodeMap,
                   symbolToValueMap, sampledPoints);

  MapVector<FPNode *, SmallVector<double, 4>> goldVals;
  for (auto *output : CS.subgraph->outputs) {
    auto *node = valueToNodeMap[output].get();
    goldVals[node].resize(FPOptNumSamples);
    CS.perOutputInitialAccCost[node] = 0.;
  }

  SmallVector<FPNode *, 4> outputs;
  for (auto *output : CS.subgraph->outputs)
    outputs.push_back(valueToNodeMap[output].get());

  if (FPOptReductionEval == "geomean") {
    struct RunningAcc {
      double sumLog = 0.0;
      unsigned count = 0;
    };
    std::unordered_map<FPNode *, RunningAcc> runAcc;
    for (auto *node : outputs)
      runAcc[node] = RunningAcc();

    for (const auto &pair : enumerate(sampledPoints)) {
      SmallVector<double, 8> results;
      getMPFRValues(outputs, pair.value(), results, true, 53);
      for (const auto &[node, result] : zip(outputs, results))
        goldVals[node][pair.index()] = result;

      getFPValues(outputs, pair.value(), results);
      for (const auto &[node, result] : zip(outputs, results)) {
        double goldVal = goldVals[node][pair.index()];
        double error = std::fabs(goldVal - result);
        if (!std::isnan(error)) {
          if (error == 0.0) {
            if (FPOptGeoMeanEps == 0.0)
              error = getOneULP(goldVal);
            else
              error += FPOptGeoMeanEps;
          } else if (FPOptGeoMeanEps != 0.0) {
            error += FPOptGeoMeanEps;
          }
          runAcc[node].sumLog += std::log(error);
          ++runAcc[node].count;
        }
      }
    }
    CS.initialAccCost = 0.0;
    for (auto *node : outputs) {
      RunningAcc &ra = runAcc[node];
      assert(ra.count != 0 && "No valid sample found for original subgraph");
      double red = std::exp(ra.sumLog / ra.count);
      CS.perOutputInitialAccCost[node] = red * std::fabs(node->grad);
      CS.initialAccCost += CS.perOutputInitialAccCost[node];
    }
  } else if (FPOptReductionEval == "arithmean") {
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
        double error = std::fabs(goldVal - result);
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
  } else if (FPOptReductionEval == "maxabs") {
    std::unordered_map<FPNode *, double> runAcc;
    for (auto *node : outputs)
      runAcc[node] = 0.0;

    for (const auto &pair : enumerate(sampledPoints)) {
      SmallVector<double, 8> results;
      getMPFRValues(outputs, pair.value(), results, true, 53);
      for (const auto &[node, result] : zip(outputs, results))
        goldVals[node][pair.index()] = result;

      getFPValues(outputs, pair.value(), results);
      for (const auto &[node, result] : zip(outputs, results)) {
        double goldVal = goldVals[node][pair.index()];
        double error = std::fabs(goldVal - result);
        if (!std::isnan(error))
          runAcc[node] = std::max(runAcc[node], error);
      }
    }
    CS.initialAccCost = 0.0;
    for (auto *node : outputs) {
      double red = runAcc[node];
      CS.perOutputInitialAccCost[node] = red * std::fabs(node->grad);
      CS.initialAccCost += CS.perOutputInitialAccCost[node];
    }
  } else {
    llvm_unreachable("Unknown fpopt-reduction strategy");
  }
  assert(!std::isnan(CS.initialAccCost));

  SmallVector<PTCandidate, 4> newCandidates;
  for (auto &candidate : CS.candidates) {
    bool discardCandidate = false;
    if (FPOptReductionEval == "geomean") {
      struct RunningAcc {
        double sumLog = 0.0;
        unsigned count = 0;
      };
      std::unordered_map<FPNode *, RunningAcc> candAcc;
      for (auto *node : outputs)
        candAcc[node] = RunningAcc();

      for (const auto &pair : enumerate(sampledPoints)) {
        SmallVector<double, 8> results;
        getFPValues(outputs, pair.value(), results, &candidate);
        for (const auto &[node, result] : zip(outputs, results)) {
          double goldVal = goldVals[node][pair.index()];
          if (FPOptStrictMode && !std::isnan(goldVal) && std::isnan(result)) {
            discardCandidate = true;
            break;
          }
          double error = std::fabs(goldVal - result);
          if (!std::isnan(error)) {
            if (error == 0.0) {
              if (FPOptGeoMeanEps == 0.0)
                error = getOneULP(goldVal);
              else
                error += FPOptGeoMeanEps;
            } else if (FPOptGeoMeanEps != 0.0) {
              error += FPOptGeoMeanEps;
            }
            candAcc[node].sumLog += std::log(error);
            ++candAcc[node].count;
          }
        }
        if (discardCandidate)
          break;
      }
      if (!discardCandidate) {
        candidate.accuracyCost = 0.0;
        for (auto *node : outputs) {
          RunningAcc &ra = candAcc[node];
          assert(ra.count != 0 &&
                 "No valid sample found for candidate subgraph");
          double red = std::exp(ra.sumLog / ra.count);
          candidate.perOutputAccCost[node] = red * std::fabs(node->grad);
          candidate.accuracyCost += candidate.perOutputAccCost[node];
        }
        assert(!std::isnan(candidate.accuracyCost));
        newCandidates.push_back(std::move(candidate));
      }
    } else if (FPOptReductionEval == "arithmean") {
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
          if (FPOptStrictMode && !std::isnan(goldVal) && std::isnan(result)) {
            discardCandidate = true;
            break;
          }
          double error = std::fabs(goldVal - result);
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
          assert(ra.count != 0 &&
                 "No valid sample found for candidate subgraph");
          double red = ra.sum / ra.count;
          candidate.perOutputAccCost[node] = red * std::fabs(node->grad);
          candidate.accuracyCost += candidate.perOutputAccCost[node];
        }
        assert(!std::isnan(candidate.accuracyCost));
        newCandidates.push_back(std::move(candidate));
      }
    } else if (FPOptReductionEval == "maxabs") {
      std::unordered_map<FPNode *, double> candAcc;
      for (auto *node : outputs)
        candAcc[node] = 0.0;

      for (const auto &pair : enumerate(sampledPoints)) {
        SmallVector<double, 8> results;
        getFPValues(outputs, pair.value(), results, &candidate);
        for (const auto &[node, result] : zip(outputs, results)) {
          double goldVal = goldVals[node][pair.index()];
          if (FPOptStrictMode && !std::isnan(goldVal) && std::isnan(result)) {
            discardCandidate = true;
            break;
          }
          double error = std::fabs(goldVal - result);
          if (!std::isnan(error))
            candAcc[node] = std::max(candAcc[node], error);
        }
        if (discardCandidate)
          break;
      }
      if (!discardCandidate) {
        candidate.accuracyCost = 0.0;
        for (auto *node : outputs) {
          double red = candAcc[node];
          candidate.perOutputAccCost[node] = red * std::fabs(node->grad);
          candidate.accuracyCost += candidate.perOutputAccCost[node];
        }
        assert(!std::isnan(candidate.accuracyCost));
        newCandidates.push_back(std::move(candidate));
      }
    } else {
      llvm_unreachable("Unknown fpopt-reduction strategy");
    }
  }
  CS.candidates = std::move(newCandidates);
}

InstructionCost getCompCost(Subgraph &subgraph, const TargetTransformInfo &TTI,
                            PTCandidate &pt) {
  assert(!subgraph.outputs.empty());

  InstructionCost cost = 0;

  Function *F = cast<Instruction>(subgraph.outputs[0])->getFunction();

  ValueToValueMapTy VMap;
  Function *FClone = CloneFunction(F, VMap);
  FClone->setName(F->getName() + "_clone");

  pt.apply(subgraph, &VMap);
  // output values in VMap are changed to the new casted values
  // llvm::errs() << "\nDEBUG: " << pt.desc << "\n";
  // FClone->print(llvm::errs());

  SmallPtrSet<Value *, 8> clonedInputs;
  for (auto &input : subgraph.inputs) {
    clonedInputs.insert(VMap[input]);
  }

  SmallPtrSet<Value *, 8> clonedOutputs;
  for (auto &output : subgraph.outputs) {
    clonedOutputs.insert(VMap[output]);
  }

  SmallPtrSet<Value *, 8> seen;
  SmallVector<Value *, 8> todo;

  todo.insert(todo.end(), clonedOutputs.begin(), clonedOutputs.end());
  while (!todo.empty()) {
    auto cur = todo.pop_back_val();
    if (!seen.insert(cur).second)
      continue;

    if (clonedInputs.contains(cur))
      continue;

    if (auto *I = dyn_cast<Instruction>(cur)) {
      auto instCost = getInstructionCompCost(I, TTI);
      // llvm::errs() << "Cost of " << *I << " is: " << instCost << "\n";

      cost += instCost;

      auto operands =
          isa<CallInst>(I) ? cast<CallInst>(I)->args() : I->operands();
      for (auto &operand : operands) {
        todo.push_back(operand);
      }
    }
  }

  FClone->eraseFromParent();

  return cost;
}