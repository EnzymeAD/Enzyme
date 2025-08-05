//=- PoseidonHerbieUtils.h - Herbie integration utilities -----------------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares utilities for integrating with the Herbie tool for
// floating-point expression optimization.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_POSEIDON_HERBIE_UTILS_H
#define ENZYME_POSEIDON_HERBIE_UTILS_H

#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/CommandLine.h"

#include <memory>
#include <string>
#include <unordered_map>

#include "PoseidonTypes.h"

using namespace llvm;

extern llvm::cl::opt<bool> EnzymePrintHerbie;
extern llvm::cl::opt<int> HerbieNumThreads;
extern llvm::cl::opt<int> HerbieTimeout;
extern llvm::cl::opt<int> HerbieNumPoints;
extern llvm::cl::opt<int> HerbieNumIters;
extern llvm::cl::opt<bool> HerbieDisableNumerics;
extern llvm::cl::opt<bool> HerbieDisableArithmetic;
extern llvm::cl::opt<bool> HerbieDisableFractions;
extern llvm::cl::opt<bool> HerbieDisableTaylor;
extern llvm::cl::opt<bool> HerbieDisableSetupSimplify;
extern llvm::cl::opt<bool> HerbieDisableGenSimplify;
extern llvm::cl::opt<bool> HerbieDisableRegime;
extern llvm::cl::opt<bool> HerbieDisableBranchExpr;
extern llvm::cl::opt<bool> HerbieDisableAvgError;

std::shared_ptr<FPNode> parseHerbieExpr(
    const std::string &expr,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap);

bool improveViaHerbie(
    const std::vector<std::string> &inputExprs,
    std::vector<ApplicableOutput> &AOs, Module *M,
    const TargetTransformInfo &TTI,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap,
    int componentIndex);

std::string getHerbieOperator(const Instruction &I);

std::string getPrecondition(
    const SmallSet<std::string, 8> &args,
    const std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    const std::unordered_map<std::string, Value *> &symbolToValueMap);

void setUnifiedAccuracyCost(
    ApplicableOutput &AO,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap);

InstructionCost getCompCost(
    const std::string &expr, Module *M, const TargetTransformInfo &TTI,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap,
    const FastMathFlags &FMF);

#endif // ENZYME_POSEIDON_HERBIE_UTILS_H