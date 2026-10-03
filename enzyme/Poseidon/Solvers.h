//=- Solvers.h - Solver utilities for Poseidon ----------------------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares solver-related utilities for the Poseidon optimization
// pass.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_SOLVERS_H
#define POSEIDON_SOLVERS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/CommandLine.h"

#include <memory>
#include <optional>
#include <unordered_map>

#include "Flags.h"
#include "Matmul.h"
#include "Types.h"

namespace poseidon {

// The DP cache holds one table per function, keyed by function identity as
// well as the cost-model fingerprint: with several annotated sites in one
// compilation unit an earlier site's table says nothing about a later one.
std::string dpCachePath();
bool dpCacheHasFunction(llvm::StringRef fnName);

// `accScale` is what a site tolerance is measured against: the site's total
// profiled sensitivity, times whatever factor the accuracy costs were already
// put on the application's scale by. Dividing an absolute accuracy cost by it
// gives the per-operation relative rounding level that would produce the same
// effect on the site's outputs, which is the unit `errorTol` is in.
llvm::SmallVector<SolutionStep> accuracyDPSolver(
    llvm::Function &F, llvm::SmallVector<CandidateOutput, 4> &COs,
    llvm::SmallVector<CandidateSubgraph, 4> &CSs,
    llvm::SmallVector<CandidateMatmul, 4> &CMs,
    std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, llvm::Value *> &symbolToValueMap,
    double errorTol = 0.0, double accScale = 0.0);

// Per matmul, pick the cheapest candidate whose estimated relative domain error
// is <= `budget` and that beats the F64 baseline; scalar subgraphs are left at
// baseline in this mode. `confidence` is the level those errors were computed
// at; the selector only reports it, since it is already in the numbers.
llvm::SmallVector<SolutionStep>
errorBudgetSelector(llvm::SmallVector<CandidateMatmul, 4> &CMs, double budget,
                    double confidence);

// True when an elementwise step edits instructions a chosen matmul step raises.
// The two selectors run separately under a site tolerance, so the footprint
// rule the DP applies internally has to be applied between their results as
// well.
bool stepConflictsWithMatmulSteps(const SolutionStep &elem,
                                  llvm::ArrayRef<SolutionStep> matmulSteps);

struct FunctionFPState {
  llvm::Function *F = nullptr;
  double errTol = 0.0;
  // Fraction of the sampled inputs a matrix-product candidate's domainError
  // bounds; elementwise candidates are scored by their mean and ignore it.
  double confidence = kDefaultConfidence;
  unsigned sampleLogBits = 0;
  // Sum of the profile's per-slot sumSens, times the kappa factor the accuracy
  // costs carry; the denominator that turns an accuracy cost into a
  // per-operation relative rounding level. See accuracyDPSolver.
  double accScale = 0.0;
  std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>> valueToNodeMap;
  std::unordered_map<std::string, llvm::Value *> symbolToValueMap;
  llvm::SmallVector<Subgraph, 1> subgraphs;
  // Backing storage for the matmuls; CandidateMatmul::matmul points into this,
  // so it must outlive materialization (it does NOT live in a collect-local).
  llvm::SmallVector<AbstractMatmul, 4> abstractMatmuls;
  llvm::SmallVector<CandidateOutput, 4> COs;
  llvm::SmallVector<CandidateSubgraph, 4> CSs;
  llvm::SmallVector<CandidateMatmul, 4> CMs;

  // WMMA-raise analyses. findScalarLoopMatmuls stashes a `ScalarEvolution *`
  // (and SCEVs it owns) into each ScalarLoopHandle for the materializer's
  // SCEVExpander, so they MUST outlive materialization and live here rather
  // than in a collect-local. Declared SE-last so it is destroyed first (it
  // references the others).
  std::optional<llvm::DominatorTree> raiseDT;
  std::optional<llvm::LoopInfo> raiseLI;
  std::optional<llvm::AssumptionCache> raiseAC;
  std::optional<llvm::TargetLibraryInfoImpl> raiseTLII;
  std::optional<llvm::TargetLibraryInfo> raiseTLI;
  std::optional<llvm::ScalarEvolution> raiseSE;
};

bool collectFPCandidates(llvm::Function &F, double errorTol, double confidence,
                         unsigned sampleLogBits, FunctionFPState &st);

bool materializeFPSolution(llvm::Function &F, FunctionFPState &st,
                           llvm::ArrayRef<SolutionStep> steps);

llvm::SmallVector<SolutionStep>
jointAccuracyDPSolver(llvm::ArrayRef<FunctionFPState *> states,
                      double errorTol);

llvm::SmallVector<SolutionStep>
parseManualRewrites(llvm::StringRef spec,
                    llvm::SmallVector<CandidateOutput, 4> &COs,
                    llvm::SmallVector<CandidateSubgraph, 4> &CSs,
                    llvm::SmallVector<CandidateMatmul, 4> &CMs);

bool applySolution(
    llvm::ArrayRef<SolutionStep> steps,
    std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, llvm::Value *> &symbolToValueMap);

} // namespace poseidon
#endif // POSEIDON_SOLVERS_H
