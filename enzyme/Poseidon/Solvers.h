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
#include "Techniques/Herbie.h"
#include "Techniques/Matmul/Matmul.h"
#include "Techniques/Precision.h"

namespace poseidon {

struct Options;
class Session;

struct SolveRecord {
  std::string selector;
  bool fromCache = false;
  CostMap table;
  SolutionMap solutions;
  double baseline = std::numeric_limits<double>::quiet_NaN();
  double accScale = 0.0;
  // What the profiled run spent in the solved sites: the DP's cost axis is in
  // shares of it.
  double profiledCost = 0.0;
  double tau = 0.0;
  bool siteTarget = false;
  std::optional<int64_t> selected;
};

// `accScale` is what a site tolerance is measured against: the site's total
// profiled sensitivity, times whatever factor the accuracy costs were already
// put on the application's scale by. Dividing an absolute accuracy cost by it
// gives the per-operation relative rounding level that would produce the same
// effect on the site's outputs, which is the unit `errorTol` is in.
struct FunctionFPState;
struct SiteTable;

double baselineAccCost(llvm::ArrayRef<RewriteUnit> COs,
                       llvm::ArrayRef<PrecisionUnit> CSs,
                       unsigned *unpricedOut = nullptr);

// Solves `units`, units of the site `st`, in one DP.
llvm::SmallVector<SolutionStep>
accuracyDPSolver(Session &S, llvm::Function &F, FunctionFPState &st,
                 llvm::ArrayRef<SiteUnit *> units, double errorTol = 0.0,
                 double accScale = 0.0, SolveRecord *record = nullptr,
                 SiteTable *table = nullptr);

llvm::SmallVector<SolutionStep>
jointAccuracyDPSolver(Session &S, llvm::ArrayRef<FunctionFPState *> states,
                      double errorTol, SolveRecord *record = nullptr);

} // namespace poseidon
#endif // POSEIDON_SOLVERS_H
