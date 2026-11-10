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

#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "Core/CacheKey.h"
#include "Techniques/Matmul/Matmul.h"
#include "Types.h"

namespace poseidon {

struct Options;
class Session;

enum class UnitKind { R, PT, M };

// One item of -poseidon-apply-rewrites: [<site>:]<unit>_<ordinal> or
// [<site>:]<unit>=<name>.
struct ApplyItem {
  std::string text;
  std::string site;
  UnitKind kind = UnitKind::R;
  size_t unit = 0;
  std::optional<size_t> ordinal;
  std::string name;
  // The keys of the module's sites `site` named. Sites are solved as the pass
  // reaches them, so that it named none is only known at the end of the
  // module.
  std::set<std::string> named;
};

struct PinState {
  std::string spec;
  std::vector<ApplyItem> items;
  std::vector<std::string> sites;
  // Two sites cut from one body share a key; printing their pin twice would
  // give a value that applies each item twice.
  std::set<std::string> printed;
};

struct SolveRecord {
  std::string selector;
  bool fromCache = false;
  std::map<llvm::InstructionCost, double> table;
  std::map<llvm::InstructionCost, llvm::SmallVector<SolutionStep>> solutions;
  double baseline = std::numeric_limits<double>::quiet_NaN();
  double accScale = 0.0;
  double tau = 0.0;
  bool siteTarget = false;
  std::optional<int64_t> selected;
};

// A site's DP table in the cache: the key of what it is solved from and, when
// the cache holds a table under that key, the table bound to this compile's
// candidates.
struct SiteTable {
  CacheKey key;
  bool hit = false;
  std::map<llvm::InstructionCost, double> costToAccuracyMap;
  std::map<llvm::InstructionCost, llvm::SmallVector<SolutionStep>>
      costToSolutionMap;
  std::optional<double> baseline;
};

// `accScale` is what a site tolerance is measured against: the site's total
// profiled sensitivity, times whatever factor the accuracy costs were already
// put on the application's scale by. Dividing an absolute accuracy cost by it
// gives the per-operation relative rounding level that would produce the same
// effect on the site's outputs, which is the unit `errorTol` is in.
llvm::SmallVector<SolutionStep> accuracyDPSolver(
    Session &S, llvm::Function &F, llvm::SmallVector<CandidateOutput, 4> &COs,
    llvm::SmallVector<CandidateSubgraph, 4> &CSs,
    llvm::SmallVector<CandidateMatmul, 4> &CMs,
    std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, llvm::Value *> &symbolToValueMap,
    double errorTol = 0.0, double accScale = 0.0, SolveRecord *record = nullptr,
    SiteTable *table = nullptr);

// Per matmul, pick the cheapest candidate whose estimated relative domain error
// is <= `budget` and that beats the F64 baseline; scalar subgraphs are left at
// baseline in this mode. `confidence` is the level those errors were computed
// at; the selector only reports it, since it is already in the numbers.
llvm::SmallVector<SolutionStep>
errorBudgetSelector(const Options &O,
                    llvm::SmallVector<CandidateMatmul, 4> &CMs, double budget,
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
  // Set when the site is solved against the DP table cache.
  std::unique_ptr<SiteTable> table;

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

bool collectFPCandidates(Session &S, llvm::Function &F, double errorTol,
                         unsigned sampleLogBits, FunctionFPState &st);

// Prices the collected candidates, unless the site's DP table is read back
// from the cache: the table already holds every price its solve used.
void priceUnlessCached(Session &S, FunctionFPState &st,
                       llvm::StringRef profilePath);

bool materializeFPSolution(Session &S, llvm::Function &F, FunctionFPState &st,
                           llvm::ArrayRef<SolutionStep> steps);

llvm::SmallVector<SolutionStep>
jointAccuracyDPSolver(Session &S, llvm::ArrayRef<FunctionFPState *> states,
                      double errorTol, SolveRecord *record = nullptr);

// The steps the items of -poseidon-apply-rewrites name at the site of `st`.
llvm::SmallVector<SolutionStep> parseManualRewrites(Session &S,
                                                    FunctionFPState &st);

// A site of the module, for the items of -poseidon-apply-rewrites that name
// one; `body` is what the site's clone is made from. An item whose site two
// sites answer to is an error.
void noteSiteForPins(Session &S, const llvm::Function &body);

// Prints the value of -poseidon-apply-rewrites that applies `applied` at the
// site of `st` again.
void printPin(Session &S, const FunctionFPState &st,
              llvm::ArrayRef<SolutionStep> applied);

// At the end of a module: an item that names a site has to have named a site
// of the module, if the module has any.
void checkPinnedSites(Session &S);

// `applied` receives the steps materialized, in order.
bool applySolution(
    Session &S, llvm::ArrayRef<SolutionStep> steps,
    std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, llvm::Value *> &symbolToValueMap,
    llvm::SmallVectorImpl<SolutionStep> &applied);

} // namespace poseidon
#endif // POSEIDON_SOLVERS_H
