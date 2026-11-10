//=- Solver.h - optimization units and the solver over them ---------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Per optimization unit, a technique proposes candidates. A candidate has a
// cost (cost model) and an error (accuracy model), both as changes against
// the code as written; the unit has a footprint, the instructions its
// candidates edit or consume. A strategy takes at most one candidate per
// unit. Two units that rewrite one output, or whose footprints overlap while
// one of them consumes its footprint, are not taken together. The DP keeps,
// per cost, the strategy of least error.
//
// One FP subgraph contributes one unit per algebraic search and one precision
// unit; they compose on the subgraph, the precision unit priced over what the
// algebraic units leave. A matrix product is one unit.
//
// The solver asks units about each other; it never sees their instructions.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_CORE_SOLVER_H
#define POSEIDON_CORE_SOLVER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <cstdint>
#include <map>

namespace poseidon {

class OptimizationUnit;

// A candidate taken: candidate `candidateIndex` of `unit`.
struct SolutionStep {
  OptimizationUnit *unit;
  size_t candidateIndex;

  SolutionStep(OptimizationUnit *unit, size_t candidateIndex)
      : unit(unit), candidateIndex(candidateIndex) {}
};

// A cost on the DP's axis. The axis is integral so that a strategy reached
// in either order, or with a step displaced and charged back, has one key.
using DPCost = int64_t;

// a + b on the DP's axis; aborts when the sum leaves it.
DPCost addDPCost(DPCost a, DPCost b);

// The DP's axis of one solve: cost-model units over the profiled run, scaled
// by the power of two that puts what the profiled run spent in the solve's
// sites between 2^32 and 2^33 units of the axis. Relative to the solve, so
// that neither the unit the cost model is measured in nor how often the
// profiled run executed a site decides which changes it can tell apart; a
// power of two, so that changes that tie in cost-model units tie on the axis.
class CostScale {
public:
  // `profiledCost`: what the profiled run spent in the solve's sites, in
  // cost-model units.
  explicit CostScale(double profiledCost);

  // A cost change in cost-model units over the profiled run, on the axis;
  // aborts when the axis cannot hold it.
  DPCost toDPCost(double delta) const;
  // The share of the profiled cost `cost` stands for, and the cost a share
  // stands for: fromShare(share(c)) == c.
  double share(DPCost cost) const;
  DPCost fromShare(double share) const;
  double profiledCost() const { return ProfiledCost; }

private:
  double ProfiledCost;
  int Exponent = 0;
};

// The frontier: per cost, the least error of a strategy of that cost.
using CostMap = std::map<DPCost, double>;
// Per cost, that strategy.
using SolutionMap = std::map<DPCost, llvm::SmallVector<SolutionStep>>;

class OptimizationUnit {
public:
  virtual ~OptimizationUnit();

  // The subclass, for isa<> and dyn_cast<>.
  unsigned kind() const { return kindId; }
  // What logs and the DP table call the unit's kind.
  virtual llvm::StringRef kindName() const = 0;

  // How many candidates the unit offers.
  virtual size_t numCandidates() const = 0;

  // Candidate i's cost change against the code as written, in cost-model units
  // over the profiled run, given the steps already in the strategy.
  virtual double costDelta(size_t i, llvm::ArrayRef<SolutionStep> prior) = 0;
  // Candidate i's error change, on the application's scale, given the steps
  // already in the strategy.
  virtual double errorDelta(size_t i, llvm::ArrayRef<SolutionStep> prior) = 0;
  // True when costDelta and errorDelta do not read `prior`. Only such a unit's
  // step can be displaced, its deltas charged back exactly.
  virtual bool pricedAlone() const { return true; }
  // The pricings the unit has memoized so far.
  virtual size_t memoizedPricings() const { return 0; }

  // True when taking a candidate consumes the footprint (a raise replaces the
  // loop that computes a product); false when it edits the footprint in place
  // and composes with the other such units on it.
  virtual bool consumesFootprint() const = 0;
  // Whether the two footprints share an instruction. A footprint does not
  // overlap itself: a product is not in conflict with itself.
  virtual bool footprintOverlaps(const OptimizationUnit &other) const = 0;
  // Whether the two units rewrite one output.
  virtual bool rewritesSameOutputAs(const OptimizationUnit &other) const = 0;
  // Whether this unit's deltas depend on whether `earlier` was taken.
  virtual bool pricedAgainst(const OptimizationUnit &earlier) const {
    return false;
  }
  // Whether every later unit admits and prices a step of this unit exactly
  // as a step of `earlier`.
  virtual bool sameRewriteTargetAs(const OptimizationUnit &earlier) const {
    return false;
  }

protected:
  explicit OptimizationUnit(unsigned kindId) : kindId(kindId) {}
  OptimizationUnit(const OptimizationUnit &) = default;
  OptimizationUnit(OptimizationUnit &&) = default;
  OptimizationUnit &operator=(const OptimizationUnit &) = default;
  OptimizationUnit &operator=(OptimizationUnit &&) = default;

private:
  unsigned kindId;
};

// ---- The rules ------------------------------------------------------------

// Whether `a` and `b` cannot be in one strategy: they rewrite one output, or
// their footprints overlap and one of them consumes its footprint.
bool excludes(const OptimizationUnit &a, const OptimizationUnit &b);

// Whether taking `later` is admitted or priced differently depending on
// whether `earlier` was taken.
bool reads(const OptimizationUnit &later, const OptimizationUnit &earlier);

// Whether `step`, which does not consume its footprint, overlaps a step of
// `chosen` that does.
bool overlapsConsumed(const SolutionStep &step,
                      llvm::ArrayRef<SolutionStep> chosen);

// The pruned DP's admission of `unit` into a strategy holding `prior`:
// refused if a prior step rewrites one of its outputs; a unit that consumes
// its footprint is refused by any prior step it overlaps; any other unit
// displaces the prior steps that consume a footprint it overlaps, their
// deltas taken off `cost` and `error`. `kept` receives the prior steps that
// stay.
bool admit(llvm::ArrayRef<SolutionStep> prior, const OptimizationUnit &unit,
           const CostScale &scale, llvm::SmallVectorImpl<SolutionStep> &kept,
           DPCost &cost, double &error);

// ---- The order units enter the DP -----------------------------------------
//
// `sites` lists each site's units in site order.

// The exact DP's order. Per site: each unit priced against earlier choices,
// right after the units it reads that are not yet placed; then the site's
// other units that do not consume their footprint. Then every site's units
// that consume their footprint, so a footprint conflict is always the
// consuming unit's refusal.
llvm::SmallVector<OptimizationUnit *>
exactOrder(llvm::ArrayRef<llvm::ArrayRef<OptimizationUnit *>> sites);

// The pruned DP's order: units priced alone that do not consume their
// footprint, then units that consume it, then units priced against earlier
// choices; within each class, sites in order. The frontier the pruned DP
// returns depends on this order.
llvm::SmallVector<OptimizationUnit *>
prunedOrder(llvm::ArrayRef<llvm::ArrayRef<OptimizationUnit *>> sites);

// ---- The frontier -----------------------------------------------------------
//
// On entry `cost` and `sol` hold the empty strategy, at cost 0 and error 0.

// The frontier of every footprint-compatible strategy over `units`, taken in
// that order.
void solveExact(llvm::ArrayRef<OptimizationUnit *> units,
                const CostScale &scale, CostMap &cost, SolutionMap &sol);

// The frontier pruned after each unit of `units`, taken in that order. With
// `progress`, a line after each unit counts the units of its kind done.
void solvePruned(llvm::ArrayRef<OptimizationUnit *> units,
                 const CostScale &scale, CostMap &cost, SolutionMap &sol,
                 bool progress);

} // namespace poseidon
#endif // POSEIDON_CORE_SOLVER_H
