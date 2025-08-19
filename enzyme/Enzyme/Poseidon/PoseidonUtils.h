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

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/InstructionCost.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace llvm;

extern "C" {
extern llvm::cl::opt<std::string> FPOptCostModelPath;
extern llvm::cl::opt<unsigned> FPOptNumSamples;
extern llvm::cl::opt<unsigned> FPOptRandomSeed;
extern llvm::cl::opt<unsigned> FPOptMinOpsForSplit;
extern llvm::cl::opt<unsigned> FPOptMinUsesForSplit;
}

struct Subgraph;
class FPNode;

// Utility function declarations
double getOneULP(double value);
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

InstructionCost getCompCost(Function *F, const TargetTransformInfo &TTI);

InstructionCost getCompCost(const SmallVector<Value *> &outputs,
                            const SetVector<Value *> &inputs,
                            const TargetTransformInfo &TTI);

void collectExprInsts(Value *V, const SetVector<Value *> &inputs,
                      SmallPtrSetImpl<Instruction *> &exprInsts,
                      SmallPtrSetImpl<Value *> &visited);

#endif // ENZYME_POSEIDON_UTILS_H
