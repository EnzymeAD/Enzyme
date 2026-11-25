//===- SolveSites.h - the DP over a solve's units -------------------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
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

#ifndef POSEIDON_SOLVE_SOLVE_SITES_H
#define POSEIDON_SOLVE_SOLVE_SITES_H

#include "Solve/DP.h"
#include "Support/CacheKey.h"
#include "Techniques/SiteUnit.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace poseidon {

struct SiteState;
struct Options;
class Session;

/// How a site's solve went, for its explanation: the frontier it searched and
/// the point it chose.
struct SolveRecord {
  /// What chose the applied point: "dp", "joint", "pinned" or "error-budget".
  std::string selector;
  /// Whether the frontier came from the DP table cache.
  bool fromCache = false;
  /// The frontier: per cost, the least accuracy cost of a strategy of that
  /// cost.
  CostMap table;
  /// Per cost on the frontier, its strategy.
  SolutionMap solutions;
  /// The accuracy cost of the site as written; NaN when it was not computed.
  double baseline = std::numeric_limits<double>::quiet_NaN();
  /// The sites' summed accuracy scale (SiteState::accScale).
  double accScale = 0.0;
  /// What the profiled run spent in the solved sites: the DP's cost axis is in
  /// shares of it.
  double profiledCost = 0.0;
  /// The relative error target the solve was held to; 0 when it solved for a
  /// cost budget.
  double tau = 0.0;
  /// Whether `tau` was written at the site rather than given by -poseidon-tau.
  bool siteTarget = false;
  /// The sites of a joint solve.
  std::vector<std::string> sites;
  /// The cost of the point applied; none when nothing was applied.
  std::optional<DPCost> selected;
};

/// A site's DP table in the cache: the key of what it is solved from, the file
/// the entry under that key is kept in and, when the cache holds it, the table
/// bound to this compile's candidates.
struct DPTable {
  /// The cache file the entry is kept in.
  std::string path;
  /// What the table is solved from.
  CacheKey key;
  /// Whether the cache held an entry under `key`.
  bool hit = false;
  /// The cached frontier, when `hit`.
  CostMap costToAccuracyMap;
  /// The strategies of the cached frontier, bound to this compile's candidates,
  /// when `hit`.
  SolutionMap costToSolutionMap;
  /// The accuracy cost of the site as written, as cached; none when the entry
  /// carries none.
  std::optional<double> baseline;
};

/// The cache entry of the DP table of the site `st` over `units`, read from
/// the cache when it holds one; an entry that is unreadable or was solved by
/// another compiler is reported and missed.
llvm::Expected<std::unique_ptr<DPTable>>
lookupDPTable(Session &S, const SiteState &st, llvm::ArrayRef<SiteUnit *> units,
              llvm::StringRef profilePath);

/// The order the DP takes `units[i]`, units of the i-th site, in, which is the
/// order of the steps of every solution: exactOrder, or prunedOrder under
/// -poseidon-early-prune.
llvm::SmallVector<OptimizationUnit *>
solveOrder(const Options &O, llvm::ArrayRef<llvm::ArrayRef<SiteUnit *>> units);

/// Solves `units[i]`, units of the site `sites[i]`, in one DP and selects a
/// frontier point: under `tau` > 0 the one the target picks, else the most
/// accurate one within -poseidon-cost-budget. `tau` is measured against the
/// sites' accScale, so that an accuracy cost divided by it is a per-operation
/// relative rounding level. A single site's table is read from or written to
/// the cache when the site has one. `joint` marks a solve of the sites of
/// -poseidon-solve-jointly, however many.
llvm::Expected<llvm::SmallVector<SolutionStep>>
solveSites(Session &S, llvm::ArrayRef<SiteState *> sites,
           llvm::ArrayRef<llvm::ArrayRef<SiteUnit *>> units, double tau,
           bool joint, SolveRecord *record);

/// Whether a budget or an accuracy target can select a frontier point of this
/// accuracy cost. A point whose cost is not finite or carries the exponent
/// penalty stays on the frontier when nothing cheaper beats it, and no
/// selection takes it.
bool selectable(const Options &O, double accCost);

} // namespace poseidon
#endif // POSEIDON_SOLVE_SOLVE_SITES_H
