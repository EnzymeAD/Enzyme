//===- Solvers.h - the DP over a solve's units ----------------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The DP over the units of one site or of several sites, the selection of a
// frontier point by budget or accuracy target, and the cache of a site's
// solved table: a later compile with the same key binds the table to its own
// candidates instead of pricing and solving again.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_SOLVERS_H
#define POSEIDON_SOLVERS_H

#include "Core/CacheKey.h"
#include "Core/Solver.h"
#include "Techniques/Unit.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace poseidon {

struct FunctionFPState;
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
  // The sites of a joint solve.
  std::vector<std::string> sites;
  std::optional<int64_t> selected;
};

// A site's DP table in the cache: the key of what it is solved from, the file
// the entry under that key is kept in and, when the cache holds it, the table
// bound to this compile's candidates.
struct SiteTable {
  std::string path;
  CacheKey key;
  bool hit = false;
  CostMap costToAccuracyMap;
  SolutionMap costToSolutionMap;
  std::optional<double> baseline;
};

// The cache entry of the DP table of the site `st` over `units`, read from
// the cache when it holds one; an entry that is unreadable or was solved by
// another compiler is reported and missed.
std::unique_ptr<SiteTable> lookupSiteTable(Session &S,
                                           const FunctionFPState &st,
                                           llvm::ArrayRef<SiteUnit *> units,
                                           llvm::StringRef profilePath);

// The order the DP takes `units[i]`, units of the i-th site, in, which is the
// order of the steps of every solution: exactOrder, or prunedOrder under
// -poseidon-early-prune.
llvm::SmallVector<OptimizationUnit *>
solveOrder(const Options &O, llvm::ArrayRef<llvm::ArrayRef<SiteUnit *>> units);

// Solves `units[i]`, units of the site `sites[i]`, in one DP and selects a
// frontier point: under `tau` > 0 the one the target picks, else the most
// accurate one within -poseidon-cost-budget. `tau` is measured against the
// sites' accScale, so that an accuracy cost divided by it is a per-operation
// relative rounding level. A single site's table is read from or written to
// the cache when the site has one. `joint` marks a solve of the sites of
// -poseidon-joint-dp, however many.
llvm::SmallVector<SolutionStep>
solveSites(Session &S, llvm::ArrayRef<FunctionFPState *> sites,
           llvm::ArrayRef<llvm::ArrayRef<SiteUnit *>> units, double tau,
           bool joint, SolveRecord *record);

// Whether a budget or an accuracy target can select a frontier point of this
// accuracy cost. A point whose cost is not finite or carries the exponent
// penalty stays on the frontier when nothing cheaper beats it, and no
// selection takes it.
bool selectable(const Options &O, double accCost);

} // namespace poseidon
#endif // POSEIDON_SOLVERS_H
