//===- DP.h - the solver over optimization units --------------------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The DP over optimization units (Core/Unit.h). A strategy takes at most one
// candidate per unit; the DP keeps, per cost, the strategy of least error.
// The solver asks units about each other and never sees their instructions.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_SOLVE_DP_H
#define POSEIDON_SOLVE_DP_H

#include "Core/Unit.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>

namespace poseidon {

/// A cost on the DP's axis. The axis is integral so that a strategy reached
/// in either order, or with a step displaced and charged back, has one key.
using DPCost = int64_t;

/// The DP's axis of one solve: cost-model units over the profiled run, scaled
/// by the power of two that puts what the profiled run spent in the solve's
/// sites between 2^32 and 2^33 units of the axis. Relative to the solve, so
/// that neither the unit the cost model is measured in nor how often the
/// profiled run executed a site decides which changes it can tell apart; a
/// power of two, so that changes that tie in cost-model units tie on the axis.
class CostScale {
public:
  /// `profiledCost`: what the profiled run spent in the solve's sites, in
  /// cost-model units.
  explicit CostScale(double profiledCost);

  /// A cost change in cost-model units over the profiled run, on the axis; none
  /// when the axis cannot hold it.
  std::optional<DPCost> toDPCost(double delta) const;
  /// The share of the profiled cost `cost` stands for, and the cost a share
  /// stands for: fromShare(share(c)) == c. A share is a -poseidon-cost-budget,
  /// which the flag check bounds by 2^30, or the share of a cost on the axis.
  double share(DPCost cost) const;
  /// The cost the share `share` of the profiled cost stands for, rounded to the
  /// axis: fromShare(share(c)) == c.
  DPCost fromShare(double share) const;

private:
  /// What the profiled run spent in the solve's sites, in cost-model units.
  double ProfiledCost;
  /// The power of two one cost-model unit is scaled by on the axis.
  int Exponent = 0;
};

/// Whether the footprints of `a` and `b` keep them out of one strategy: they
/// overlap and one of the two consumes its footprint.
bool footprintsConflict(const OptimizationUnit &a, const OptimizationUnit &b);

/// Why the cost `cost`, introduced by `what` ("a cost of"), does not fit the
/// DP's axis.
std::string offAxisMessage(llvm::StringRef what, double cost);

/// The frontier: per cost, the least error of a strategy of that cost.
using CostMap = std::map<DPCost, double>;
/// Per cost, that strategy.
using SolutionMap = std::map<DPCost, llvm::SmallVector<SolutionStep>>;

// In both orders below, `sites` lists each site's units in site order.

/// The exact DP's order. Per site: each unit priced against earlier choices,
/// right after the units it reads that are not yet placed; then the site's
/// other units that do not consume their footprint. Then every site's units
/// that consume their footprint, so a footprint conflict is always the
/// consuming unit's refusal.
llvm::SmallVector<OptimizationUnit *>
exactOrder(llvm::ArrayRef<llvm::ArrayRef<OptimizationUnit *>> sites);

/// The pruned DP's order: units priced alone that do not consume their
/// footprint, then units that consume it, then units priced against earlier
/// choices; within each class, sites in order. The frontier the pruned DP
/// returns depends on this order.
llvm::SmallVector<OptimizationUnit *>
prunedOrder(llvm::ArrayRef<llvm::ArrayRef<OptimizationUnit *>> sites);

// For both solvers below, `cost` and `sol` hold the empty strategy on entry, at
// cost 0 and error 0.

// Both return an error when a cost change the search reprices, or a sum of
// changes, leaves the DP's axis: the search keeps it until it ends, and drops
// the strategies it concerns.

/// The frontier of every footprint-compatible strategy over `units`, taken in
/// that order; the default solve. With `progress`, a line after each unit and
/// a closing summary.
llvm::Error solveExact(llvm::ArrayRef<OptimizationUnit *> units,
                       const CostScale &scale, CostMap &cost, SolutionMap &sol,
                       bool progress);

/// The frontier pruned after each unit of `units`, taken in that order: a
/// strategy is dropped for one that is cheaper, no less accurate and neither
/// refuses nor reprices a later unit that it leaves free. An approximation of
/// solveExact's frontier under -poseidon-early-prune, faster on sites with many
/// units. With `progress`, a line after each unit counts the units of its kind
/// done.
llvm::Error solvePruned(llvm::ArrayRef<OptimizationUnit *> units,
                        const CostScale &scale, CostMap &cost, SolutionMap &sol,
                        bool progress);

} // namespace poseidon
#endif // POSEIDON_SOLVE_DP_H
