//=- PoseidonUtils.h - Utility functions for Poseidon optimization pass ----=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares utility functions for the Poseidon floating-point
// optimization pass.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_POSEIDON_UTILS_H
#define ENZYME_POSEIDON_UTILS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/InstructionCost.h"

#include <map>
#include <string>
#include <unordered_set>

namespace llvm {
class Function;
class Instruction;
class TargetTransformInfo;
class Value;
} // namespace llvm

using namespace llvm;

// Forward declarations
class FPNode;
struct FPCC;
enum class PrecisionChangeType;

// Utility function declarations
double getOneULP(double value);
unsigned getMPFRPrec(PrecisionChangeType type);
Type *getLLVMFPType(PrecisionChangeType type, LLVMContext &context);
PrecisionChangeType getPrecisionChangeType(Type *type);
StringRef getPrecisionChangeTypeString(PrecisionChangeType type);
std::string getLibmFunctionForPrecision(StringRef funcName, Type *newType);
double stringToDouble(const std::string &str);
void topoSort(const SetVector<Instruction *> &insts,
              SmallVectorImpl<Instruction *> &instsSorted);

void getUniqueArgs(const std::string &expr, SmallSet<std::string, 8> &args);

const std::map<std::pair<std::string, std::string>, InstructionCost> &
getCostModel();
InstructionCost queryCostModel(const std::string &OpcodeName,
                               const std::string &TypeName);
InstructionCost getInstructionCompCost(const Instruction *I,
                                       const TargetTransformInfo &TTI);

const std::unordered_set<std::string> &getPTFuncs();

InstructionCost computeMaxCost(
    BasicBlock *BB, std::unordered_map<BasicBlock *, InstructionCost> &MaxCost,
    std::unordered_set<BasicBlock *> &Visited, const TargetTransformInfo &TTI);

// // Various overloads of getCompCost
// InstructionCost getCompCost(Function *F, const TargetTransformInfo &TTI);
// InstructionCost getCompCost(const SmallVector<Value *> &outputs,
//                             const SetVector<Value *> &inputs,
//                             const TargetTransformInfo &TTI);
// InstructionCost getCompCost(const std::string &expr,
//                             const SetVector<Value *> &inputs,
//                             const TargetTransformInfo &TTI,
//                             std::shared_ptr<FPNode> *rootOut = nullptr);
// InstructionCost getCompCost(FPCC &component, const TargetTransformInfo &TTI,
//                             bool groundTruth = false);

// // FPCC utilities
// void splitFPCC(FPCC &CC, SmallVector<FPCC, 1> &newCCs);
// void collectExprInsts(Value *V, const SetVector<Value *> &inputs,
//                       SetVector<Instruction *> &exprInsts);

// // Herbie-related utilities
// std::string getHerbieOperator(const Instruction &I);
// bool extractValueFromLog(const std::string &logPath,
//                          const std::string &functionName,
//                          const std::string &logType,
//                          std::unordered_map<FPNode *, SmallVector<double, 4>>
//                          &map, const std::string &regex = "");
// bool extractGradFromLog(const std::string &logPath,
//                         const std::string &functionName,
//                         const std::string &logType,
//                         std::unordered_map<Value *, GradInfo> &gradMap);
// bool isLogged(const std::string &logPath, const std::string &functionName);
// std::string getPrecondition(const SetVector<Value *> &inputs,
//                             std::unordered_map<Value *,
//                             std::shared_ptr<FPNode>> &valueToNodeMap);

#endif // ENZYME_POSEIDON_UTILS_H