//===- DP.cpp - optimization units and the solver over them ---------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DP.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;

namespace poseidon {

OptimizationUnit::~OptimizationUnit() = default;

// `value` in cost-model units on the axis 2^exponent scales them to; none
// when the axis cannot hold it.
static std::optional<DPCost> onAxis(double value, int exponent) {
  double q = std::ldexp(value, exponent);
  // 2^63 is the first double past the axis; every smaller magnitude rounds
  // into it.
  if (!(std::fabs(q) < 0x1p63))
    return std::nullopt;
  return std::llround(q);
}

// The profiled cost lies between 2^kProfiledCostLog2 and twice that on the
// axis.
static constexpr int kProfiledCostLog2 = 32;

CostScale::CostScale(double profiledCost) : ProfiledCost(profiledCost) {
  assert(profiledCost >= 0.0 && std::isfinite(profiledCost) &&
         "a profiled cost is a sum of checked prices times counts");
  if (profiledCost > 0.0)
    Exponent = kProfiledCostLog2 - std::ilogb(profiledCost);
}

std::optional<DPCost> CostScale::toDPCost(double delta) const {
  // A solve the profile never reached costs nothing and saves nothing.
  if (delta == 0.0)
    return 0;
  assert(ProfiledCost != 0.0 &&
         "a cost change in a solve whose profiled run spent nothing");
  return onAxis(delta, Exponent);
}

double CostScale::share(DPCost cost) const {
  if (cost == 0)
    return 0.0;
  return std::ldexp((double)cost, -Exponent) / ProfiledCost;
}

DPCost CostScale::fromShare(double share) const {
  // Nothing ran, so no change saves anything: a budget keeps only its sign.
  if (ProfiledCost == 0.0)
    return (share > 0.0) - (share < 0.0);
  assert(std::isfinite(share) && "-poseidon-cost-budget is checked finite");
  // The profiled cost lies below 2^33 on the axis and a share is at most 2^30,
  // so only rounding at the axis's end leaves it; a budget there holds every
  // cost the axis does.
  return onAxis(share * ProfiledCost, Exponent)
      .value_or(share > 0.0 ? std::numeric_limits<DPCost>::max()
                            : std::numeric_limits<DPCost>::min());
}

static Error offAxis(double delta) {
  std::string Text;
  raw_string_ostream(Text) << delta;
  return make_error<StringError>(
      "a cost change of " + Text +
          " does not fit the DP's cost axis; a cost-model row prices a "
          "candidate beyond 2^30 times the profiled cost",
      inconvertibleErrorCode());
}

// a + b on the DP's axis; none, with the error kept in `deferred`, when the
// sum leaves it.
static std::optional<DPCost> addDPCost(DPCost a, DPCost b,
                                       DeferredError &deferred) {
  if (std::optional<DPCost> sum = checkedAdd(a, b))
    return sum;
  deferred.set(make_error<StringError>(
      "the cost " + Twine(a) + " + " + Twine(b) +
          " leaves the DP's integer cost axis; a cost-model row prices a "
          "candidate beyond 2^30 times the profiled cost",
      inconvertibleErrorCode()));
  return std::nullopt;
}

// Candidate i's cost change on the axis, with `prior` before it; none, with
// the error kept in `deferred`, when the axis cannot hold it.
static std::optional<DPCost> costOnAxis(const CostScale &scale,
                                        OptimizationUnit &unit, size_t i,
                                        ArrayRef<SolutionStep> prior,
                                        DeferredError &deferred) {
  const double delta = unit.costDelta(i, prior);
  std::optional<DPCost> cost = scale.toDPCost(delta);
  if (!cost)
    deferred.set(offAxis(delta));
  return cost;
}

// Whether `a` and `b` cannot be in one strategy: they rewrite one output, or
// their footprints overlap and one of them consumes its footprint.
static bool excludes(const OptimizationUnit &a, const OptimizationUnit &b) {
  if (a.rewritesSameOutputAs(b))
    return true;
  return (a.consumesFootprint() || b.consumesFootprint()) &&
         a.footprintOverlaps(b);
}

// Whether taking `later` is admitted or priced differently depending on
// whether `earlier` was taken. Neither test reads which candidate `earlier`
// took.
static bool reads(const OptimizationUnit &later,
                  const OptimizationUnit &earlier) {
  return excludes(later, earlier) || later.pricedAgainst(earlier);
}

// The pruned DP's admission of `unit` into a strategy holding `prior`:
// refused if a prior step rewrites one of its outputs; a unit that consumes
// its footprint is refused by any prior step it overlaps; any other unit
// displaces the prior steps that consume a footprint it overlaps, their
// deltas taken off `cost` and `error`. `kept` receives the prior steps that
// stay.
//
// Charging a displaced step back is exact because a step that consumes its
// footprint is priced alone. A unit that consumes its footprint is refused
// instead, since a prior step may be priced against the steps before it.
// Without displacement the frontier keyed by cost evicts equal-cost prefixes
// that later units would have extended.
static bool admit(ArrayRef<SolutionStep> prior, const OptimizationUnit &unit,
                  const CostScale &scale, SmallVectorImpl<SolutionStep> &kept,
                  DPCost &cost, double &error, DeferredError &deferred) {
  // Only one rewrite of an output can be materialized.
  for (const SolutionStep &p : prior)
    if (p.unit->rewritesSameOutputAs(unit))
      return false;
  if (unit.consumesFootprint()) {
    for (const SolutionStep &p : prior)
      if (p.unit->footprintOverlaps(unit))
        return false;
    kept.assign(prior.begin(), prior.end());
    return true;
  }
  kept.clear();
  for (const SolutionStep &p : prior) {
    if (p.unit->consumesFootprint() && p.unit->footprintOverlaps(unit)) {
      assert(p.unit->pricedAlone() && "a displaced step is charged back alone");
      std::optional<DPCost> back =
          costOnAxis(scale, *p.unit, p.candidateIndex, {}, deferred);
      std::optional<DPCost> charged =
          back ? addDPCost(cost, -*back, deferred) : std::nullopt;
      if (!charged)
        return false;
      cost = *charged;
      error -= p.unit->errorDelta(p.candidateIndex, {});
      continue;
    }
    kept.push_back(p);
  }
  return true;
}

static void paretoIntegrate(const CostScale &scale, CostMap &cost,
                            SolutionMap &sol, OptimizationUnit &unit,
                            DeferredError &deferred) {
  CostMap newCost = cost;
  SolutionMap newSol = sol;
  for (const auto &p : cost) {
    DPCost curr = p.first;
    double currAcc = p.second;
    const SmallVector<SolutionStep> &chosen = sol[p.first];
    SmallVector<SolutionStep> priorSteps;
    if (!admit(chosen, unit, scale, priorSteps, curr, currAcc, deferred))
      continue;
    for (size_t i = 0; i < unit.numCandidates(); ++i) {
      std::optional<DPCost> cd =
          costOnAxis(scale, unit, i, priorSteps, deferred);
      std::optional<DPCost> sum =
          cd ? addDPCost(curr, *cd, deferred) : std::nullopt;
      if (!sum)
        continue;
      double ad = unit.errorDelta(i, priorSteps);
      // Compared against the post-displacement point, so a displacement that
      // gives back more than the candidate saves is dropped.
      DPCost nc = *sum;
      if (nc >= p.first && currAcc + ad >= p.second)
        continue;
      double na = currAcc + ad;
      auto it = newCost.find(nc);
      if (it == newCost.end() || it->second > na) {
        newCost[nc] = na;
        newSol[nc] = priorSteps;
        newSol[nc].emplace_back(&unit, i);
      }
    }
  }
  cost = std::move(newCost);
  sol = std::move(newSol);
}

// Bit j is set when a strategy holding `steps` constrains later[j]: admit
// refuses it (a step rewrites one of its outputs, or it consumes its footprint
// and a step overlaps that footprint), or it is priced against a step.
static BitVector constrainedLater(ArrayRef<SolutionStep> steps,
                                  ArrayRef<OptimizationUnit *> later) {
  BitVector refused(later.size());
  for (size_t j = 0; j < later.size(); ++j) {
    const OptimizationUnit &unit = *later[j];
    refused[j] = any_of(steps, [&](const SolutionStep &s) {
      return s.unit->rewritesSameOutputAs(unit) ||
             (unit.consumesFootprint() && s.unit->footprintOverlaps(unit)) ||
             unit.pricedAgainst(*s.unit);
    });
  }
  return refused;
}

// Drops a point for another that is cheaper, no less accurate, and constrains
// none of the units still to come, `later`, that the dropped point leaves
// free: a point some later unit can extend, or prices as written, where the
// cheaper one it refuses or reprices, is kept.
static void paretoPrune(CostMap &cost, SolutionMap &sol,
                        ArrayRef<OptimizationUnit *> later) {
  DenseMap<DPCost, BitVector> constrained;
  for (const auto &[c, steps] : sol)
    constrained[c] = constrainedLater(steps, later);
  CostMap pruned;
  SolutionMap prunedSol;
  for (const auto &[lc, la] : cost) {
    const BitVector &lConstrained = constrained[lc];
    bool dominated = false;
    for (const auto &[rc, ra] : cost) {
      if (rc >= lc)
        break;
      if (la >= ra && !constrained[rc].test(lConstrained)) {
        dominated = true;
        break;
      }
    }
    if (!dominated) {
      pruned[lc] = la;
      prunedSol[lc] = sol[lc];
    }
  }
  cost = std::move(pruned);
  sol = std::move(prunedSol);
}

SmallVector<OptimizationUnit *>
exactOrder(ArrayRef<ArrayRef<OptimizationUnit *>> sites) {
  SmallVector<OptimizationUnit *> order;
  for (ArrayRef<OptimizationUnit *> site : sites) {
    SmallPtrSet<const OptimizationUnit *, 16> placed;
    for (OptimizationUnit *u : site) {
      if (u->pricedAlone() || u->consumesFootprint())
        continue;
      for (OptimizationUnit *w : site)
        if (w != u && !placed.contains(w) && u->pricedAgainst(*w)) {
          order.push_back(w);
          placed.insert(w);
        }
      order.push_back(u);
      placed.insert(u);
    }
    for (OptimizationUnit *u : site)
      if (!placed.contains(u) && !u->consumesFootprint())
        order.push_back(u);
  }
  for (ArrayRef<OptimizationUnit *> site : sites)
    for (OptimizationUnit *u : site)
      if (u->consumesFootprint())
        order.push_back(u);
  return order;
}

SmallVector<OptimizationUnit *>
prunedOrder(ArrayRef<ArrayRef<OptimizationUnit *>> sites) {
  auto rank = [](const OptimizationUnit *u) {
    if (u->consumesFootprint())
      return 1;
    return u->pricedAlone() ? 0 : 2;
  };
  SmallVector<OptimizationUnit *> order;
  for (ArrayRef<OptimizationUnit *> site : sites)
    order.append(site.begin(), site.end());
  std::stable_sort(order.begin(), order.end(),
                   [&](const OptimizationUnit *a, const OptimizationUnit *b) {
                     return rank(a) < rank(b);
                   });
  return order;
}

namespace {
// The exact DP over units in order. A partial solution is keyed by its cost
// and by its signature: the taken units some later unit still reads. Two with
// one signature are admitted and priced alike by every later unit, so one that
// is cheaper and no less accurate than the other stays so under every
// completion, and only such a one is dropped. After the last unit every
// signature is empty and the one remaining bucket is the frontier of all
// footprint-compatible strategies.
class ExactDP {
public:
  // The canonical units, as indices into `units`, sorted.
  using Signature = std::vector<int>;
  struct Entry {
    double acc;
    SmallVector<SolutionStep> steps;
  };
  using Bucket = std::map<DPCost, Entry>;
  using Table = std::map<Signature, Bucket>;

  ExactDP(ArrayRef<OptimizationUnit *> units, const CostScale &scale,
          DeferredError &deferred);
  // The table after unit `k`, from the table before it.
  Table step(int k, const Table &table) const;

private:
  // `sig` without the units no unit after `k` reads.
  Signature stillRead(const Signature &sig, int k) const;
  // For each bucket of `table`, the steps unit `k` is priced against, those
  // of the bucket's first entry over canonical units; none when the unit has
  // no candidates or one of those steps excludes it.
  SmallVector<std::optional<SmallVector<SolutionStep>>, 0>
  priors(int k, const Table &table) const;

  ArrayRef<OptimizationUnit *> units;
  const CostScale &scale;
  DeferredError &deferred;
  // The last unit reading a canonical unit, -1 for none; the canonical unit of
  // each unit, the first with the same rewrite target.
  SmallVector<int> lastRead, canon;
  DenseMap<const OptimizationUnit *, int> unitIndex;
};
} // namespace

// Keeps the more accurate of the entries at cost `c`, the earlier on a tie.
static void insertEntry(ExactDP::Bucket &b, DPCost c, double a,
                        const SmallVectorImpl<SolutionStep> &steps) {
  auto [it, fresh] = b.try_emplace(c);
  if (!fresh && !(it->second.acc > a))
    return;
  it->second.acc = a;
  it->second.steps.assign(steps.begin(), steps.end());
}

// An entry goes when a cheaper one is no less accurate.
static void pruneBucket(ExactDP::Bucket &b) {
  std::optional<double> best;
  for (auto it = b.begin(); it != b.end();) {
    const double a = it->second.acc;
    if (best && a >= *best) {
      it = b.erase(it);
      continue;
    }
    if (!std::isnan(a))
      best = best ? std::min(*best, a) : a;
    ++it;
  }
}

ExactDP::ExactDP(ArrayRef<OptimizationUnit *> units, const CostScale &scale,
                 DeferredError &deferred)
    : units(units), scale(scale), deferred(deferred),
      lastRead(units.size(), -1), canon(units.size()) {
  const int n = units.size();
  for (int u = 0; u < n; ++u) {
    unitIndex[units[u]] = u;
    canon[u] = u;
    for (int w = 0; w < u; ++w)
      if (units[u]->sameRewriteTargetAs(*units[w])) {
        canon[u] = canon[w];
        break;
      }
    for (int v = u + 1; v < n; ++v)
      if (reads(*units[v], *units[u]))
        lastRead[canon[u]] = std::max(lastRead[canon[u]], v);
  }
}

ExactDP::Signature ExactDP::stillRead(const Signature &sig, int k) const {
  Signature kept;
  for (int u : sig)
    if (lastRead[u] > k)
      kept.push_back(u);
  return kept;
}

SmallVector<std::optional<SmallVector<SolutionStep>>, 0>
ExactDP::priors(int k, const Table &table) const {
  OptimizationUnit &unit = *units[k];
  SmallVector<std::optional<SmallVector<SolutionStep>>, 0> pricedFor;
  for (const auto &[sig, bucket] : table) {
    std::optional<SmallVector<SolutionStep>> &priced = pricedFor.emplace_back();
    if (unit.numCandidates() == 0)
      continue;
    const SmallVector<SolutionStep> &rep = bucket.begin()->second.steps;
    if (llvm::any_of(rep, [&](const SolutionStep &s) {
          return excludes(unit, *s.unit);
        }))
      continue;
    priced.emplace();
    for (const SolutionStep &s : rep) {
      int u = canon[unitIndex.find(s.unit)->second];
      priced->push_back(SolutionStep(units[u], s.candidateIndex));
    }
  }
  return pricedFor;
}

ExactDP::Table ExactDP::step(int k, const Table &table) const {
  OptimizationUnit &unit = *units[k];
  // Every entry carries over with unit k left as written.
  Table next;
  for (const auto &[sig, bucket] : table) {
    Bucket &into = next[stillRead(sig, k)];
    for (const auto &[c, e] : bucket)
      insertEntry(into, c, e.acc, e.steps);
  }
  // Everything a unit reads is in the signature, so one entry stands for the
  // bucket; the unit extends a bucket it is not excluded from.
  SmallVector<std::optional<SmallVector<SolutionStep>>, 0> pricedFor =
      priors(k, table);
  SmallVector<ArrayRef<SolutionStep>, 0> pricedSteps;
  for (const std::optional<SmallVector<SolutionStep>> &priced : pricedFor)
    if (priced)
      pricedSteps.push_back(*priced);
  if (!pricedSteps.empty())
    unit.priceAhead(pricedSteps);
  for (const auto &[entry, priced] : llvm::zip(table, pricedFor)) {
    if (!priced)
      continue;
    const auto &[sig, bucket] = entry;
    SmallVector<std::optional<DPCost>> dc;
    SmallVector<double> da;
    for (size_t i = 0; i < unit.numCandidates(); ++i) {
      dc.push_back(costOnAxis(scale, unit, i, *priced, deferred));
      da.push_back(unit.errorDelta(i, *priced));
    }
    Signature taken = stillRead(sig, k);
    if (lastRead[canon[k]] > k) {
      taken.push_back(canon[k]);
      llvm::sort(taken);
    }
    Bucket &into = next[taken];
    SmallVector<SolutionStep> steps;
    for (const auto &[c, e] : bucket) {
      steps.assign(e.steps.begin(), e.steps.end());
      steps.push_back(SolutionStep(&unit, 0));
      for (size_t i = 0; i < unit.numCandidates(); ++i) {
        std::optional<DPCost> sum =
            dc[i] ? addDPCost(c, *dc[i], deferred) : std::nullopt;
        if (!sum)
          continue;
        steps.back().candidateIndex = i;
        insertEntry(into, *sum, e.acc + da[i], steps);
      }
    }
  }
  for (auto &[sig, bucket] : next)
    pruneBucket(bucket);
  return next;
}

Error solveExact(ArrayRef<OptimizationUnit *> units, const CostScale &scale,
                 CostMap &cost, SolutionMap &sol, bool progress) {
  const int n = units.size();
  DeferredError deferred;
  ExactDP dp(units, scale, deferred);
  ExactDP::Table table;
  insertEntry(table[ExactDP::Signature()], cost.begin()->first,
              cost.begin()->second, sol.begin()->second);
  size_t peak = 1, peakSignatures = 1, pricings = 0;
  const auto start = std::chrono::steady_clock::now();
  for (int k = 0; k < n; ++k) {
    const size_t memoBefore = units[k]->memoizedPricings();
    table = dp.step(k, table);
    pricings += units[k]->memoizedPricings() - memoBefore;
    size_t entries = 0;
    for (const auto &[sig, bucket] : table)
      entries += bucket.size();
    if (entries > peak) {
      peak = entries;
      peakSignatures = table.size();
    }
    if (progress)
      llvm::errs() << "[poseidon] Exact DP: finished unit " << k + 1 << " of "
                   << n << ": " << entries << " entries in " << table.size()
                   << " signatures\n";
  }
  assert(table.size() == 1 && table.begin()->first.empty() &&
         "exact DP: a signature outlived the last unit");

  cost.clear();
  sol.clear();
  for (auto &[c, e] : table.begin()->second) {
    cost[c] = e.acc;
    sol[c] = std::move(e.steps);
  }
  const double secs =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  if (progress)
    llvm::errs() << "[poseidon] Exact DP: " << cost.size()
                 << " frontier points; peak " << peak << " entries in "
                 << peakSignatures << " signatures; " << pricings
                 << " precision changes priced; " << secs << " s\n";
  return deferred.take();
}

Error solvePruned(ArrayRef<OptimizationUnit *> units, const CostScale &scale,
                  CostMap &cost, SolutionMap &sol, bool progress) {
  DeferredError deferred;
  StringMap<std::pair<size_t, size_t>> counts;
  for (const OptimizationUnit *unit : units)
    ++counts[unit->kindName()].second;
  for (size_t k = 0; k < units.size(); ++k) {
    OptimizationUnit *unit = units[k];
    paretoIntegrate(scale, cost, sol, *unit, deferred);
    paretoPrune(cost, sol, units.drop_front(k + 1));
    if (!progress)
      continue;
    auto &[done, total] = counts[unit->kindName()];
    llvm::errs() << "[poseidon] Finished processing " << ++done << " of "
                 << total << ' ' << unit->kindName()
                 << " unit(s); the DP table has " << cost.size()
                 << " point(s)\n";
  }
  return deferred.take();
}

} // namespace poseidon
