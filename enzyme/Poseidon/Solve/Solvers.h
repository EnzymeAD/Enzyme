//===- Solvers.h - the DP over a solve's units ----------------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The DP over the units of one site or of several sites, and the selection
// of a frontier point by budget or accuracy target.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_SOLVERS_H
#define POSEIDON_SOLVERS_H

#include "Core/Solver.h"
#include "Techniques/Herbie.h"
#include "Techniques/Precision.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"

#include <limits>
#include <optional>
#include <string>

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

struct FunctionFPState;
struct SiteTable;

double baselineAccCost(llvm::ArrayRef<RewriteUnit> COs,
                       llvm::ArrayRef<PrecisionUnit> CSs,
                       unsigned *unpricedOut = nullptr);

// Solves `units`, units of the site `st`, in one DP. `accScale` is what
// `errorTol` is measured against: the site's total profiled sensitivity times
// the factor that put the accuracy costs on the application's scale, so that
// an accuracy cost divided by it is a per-operation relative rounding level.
llvm::SmallVector<SolutionStep>
accuracyDPSolver(Session &S, llvm::Function &F, FunctionFPState &st,
                 llvm::ArrayRef<SiteUnit *> units, double errorTol = 0.0,
                 double accScale = 0.0, SolveRecord *record = nullptr,
                 SiteTable *table = nullptr);

llvm::SmallVector<SolutionStep>
jointAccuracyDPSolver(Session &S, llvm::ArrayRef<FunctionFPState *> states,
                      double errorTol, SolveRecord *record = nullptr);

// Whether a budget or an accuracy target can select a frontier point of this
// accuracy cost. A point whose cost is not finite or carries the exponent
// penalty stays on the frontier when nothing cheaper beats it, and no
// selection takes it.
bool selectable(const Options &O, double accCost);

} // namespace poseidon
#endif // POSEIDON_SOLVERS_H
