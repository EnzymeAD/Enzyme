//===- DP.h - optimization units and the solver over them -----------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Optimization units and the DP over them. A unit offers candidates, each with
// a cost and an error change against the code as written, and has a
// footprint, the instructions its candidates edit or consume. A strategy takes
// at most one candidate per unit; the DP keeps, per cost, the strategy of
// least error. The solver asks units about each other and never sees their
// instructions.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_SOLVE_DP_H
#define POSEIDON_SOLVE_DP_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>

namespace poseidon {

class OptimizationUnit;

/// A candidate taken: candidate `candidateIndex` of `unit`.
struct SolutionStep {
  /// The unit; owned by the site it was proposed for.
  OptimizationUnit *unit;
  /// The candidate's index among the unit's candidates.
  size_t candidateIndex;

  /// Candidate `candidateIndex` of `unit`.
  SolutionStep(OptimizationUnit *unit, size_t candidateIndex)
      : unit(unit), candidateIndex(candidateIndex) {}
};

/// An error met where it cannot be returned, kept until it can be; the first
/// of several is kept.
class DeferredError {
public:
  DeferredError() = default;
  DeferredError(DeferredError &&) = default;
  DeferredError &operator=(DeferredError &&) = default;
  ~DeferredError() {
    if (E)
      llvm::consumeError(std::move(*E));
  }
  /// Keeps `New` unless an error is kept already, and then consumes it.
  void set(llvm::Error New) {
    if (E)
      llvm::consumeError(std::move(New));
    else
      E = std::move(New);
  }
  /// The kept error, or success when there is none; nothing is kept after.
  llvm::Error take() {
    if (!E)
      return llvm::Error::success();
    llvm::Error Out = std::move(*E);
    E.reset();
    return Out;
  }

private:
  std::optional<llvm::Error> E;
};

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

/// The frontier: per cost, the least error of a strategy of that cost.
using CostMap = std::map<DPCost, double>;
/// Per cost, that strategy.
using SolutionMap = std::map<DPCost, llvm::SmallVector<SolutionStep>>;

/// A place in the code the solver chooses for: the candidates it offers, their
/// prices, and how its steps combine with other units'.
class OptimizationUnit {
public:
  virtual ~OptimizationUnit();

  /// The subclass, for isa<> and dyn_cast<>.
  unsigned kind() const { return kindId; }
  /// What the solver's progress lines call the unit's kind.
  virtual llvm::StringRef kindName() const = 0;

  /// How many candidates the unit offers.
  virtual size_t numCandidates() const = 0;

  /// Candidate i's cost change against the code as written, in cost-model units
  /// over the profiled run, given the steps already in the strategy.
  virtual double costDelta(size_t i, llvm::ArrayRef<SolutionStep> prior) = 0;
  /// Candidate i's error change, on the application's scale, given the steps
  /// already in the strategy.
  virtual double errorDelta(size_t i, llvm::ArrayRef<SolutionStep> prior) = 0;
  /// The cost change, in cost-model units over the profiled run, by which
  /// candidate i's price can be off: a price measured as a wall-clock ratio
  /// repeats only to within it. 0 for a price exact on the model's terms.
  virtual double costTieBand(size_t i) const { return 0.0; }
  /// True when costDelta and errorDelta do not read `prior`. Only such a unit's
  /// step can be displaced, its deltas charged back exactly.
  virtual bool pricedAlone() const { return true; }
  /// The pricings the unit has memoized so far.
  virtual size_t memoizedPricings() const { return 0; }
  /// Called before the solver asks for the deltas of every candidate against
  /// each of `priors`, for a unit that prices several of them at once.
  virtual void priceAhead(llvm::ArrayRef<llvm::ArrayRef<SolutionStep>> priors) {
  }

  /// True when taking a candidate consumes the footprint (a raise replaces the
  /// loop that computes a product); false when it edits the footprint in place
  /// and composes with the other such units on it.
  virtual bool consumesFootprint() const = 0;
  /// Whether the two footprints share an instruction. A footprint does not
  /// overlap itself: a product is not in conflict with itself.
  virtual bool footprintOverlaps(const OptimizationUnit &other) const = 0;
  /// Whether the two units rewrite one output.
  virtual bool rewritesSameOutputAs(const OptimizationUnit &other) const = 0;
  /// Whether this unit's deltas depend on whether `earlier` was taken.
  virtual bool pricedAgainst(const OptimizationUnit &earlier) const {
    return false;
  }
  /// Whether every later unit admits and prices a step of this unit exactly
  /// as a step of `earlier`.
  virtual bool sameRewriteTargetAs(const OptimizationUnit &earlier) const {
    return false;
  }

protected:
  /// A unit of the subclass `kindId`.
  explicit OptimizationUnit(unsigned kindId) : kindId(kindId) {}
  OptimizationUnit(const OptimizationUnit &) = default;
  OptimizationUnit(OptimizationUnit &&) = default;
  OptimizationUnit &operator=(const OptimizationUnit &) = default;
  OptimizationUnit &operator=(OptimizationUnit &&) = default;

private:
  /// The subclass, as kind() returns it.
  unsigned kindId;
};

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
/// that order. With `progress`, a line after each unit and a closing summary.
llvm::Error solveExact(llvm::ArrayRef<OptimizationUnit *> units,
                       const CostScale &scale, CostMap &cost, SolutionMap &sol,
                       bool progress);

/// The frontier pruned after each unit of `units`, taken in that order. With
/// `progress`, a line after each unit counts the units of its kind done.
llvm::Error solvePruned(llvm::ArrayRef<OptimizationUnit *> units,
                        const CostScale &scale, CostMap &cost, SolutionMap &sol,
                        bool progress);

} // namespace poseidon
#endif // POSEIDON_SOLVE_DP_H
