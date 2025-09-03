//=- PoseidonSolvers.h - Solver utilities for Poseidon --------------------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares solver-related utilities for the Poseidon optimization
// pass.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_POSEIDON_SOLVERS_H
#define ENZYME_POSEIDON_SOLVERS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/CommandLine.h"

#include <memory>
#include <unordered_map>

#include "PoseidonTypes.h"

using namespace llvm;

extern "C" {
extern llvm::cl::opt<std::string> FPOptSolverType;
extern llvm::cl::opt<int64_t> FPOptComputationCostBudget;
extern llvm::cl::opt<bool> FPOptShowTable;
extern llvm::cl::list<int64_t> FPOptShowTableCosts;
extern llvm::cl::opt<bool> FPOptEarlyPrune;
extern llvm::cl::opt<double> FPOptCostDominanceThreshold;
extern llvm::cl::opt<double> FPOptAccuracyDominanceThreshold;
}

bool accuracyGreedySolver(
    SmallVector<CandidateOutput, 4> &COs,
    SmallVector<CandidateSubgraph, 4> &CSs,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap);

bool accuracyDPSolver(
    Function &F, const TargetTransformInfo &TTI,
    SmallVector<CandidateOutput, 4> &COs,
    SmallVector<CandidateSubgraph, 4> &CSs,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap);

#endif // ENZYME_POSEIDON_SOLVERS_H