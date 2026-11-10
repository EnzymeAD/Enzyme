//=- Solver.cpp - optimization units and the solver over them -------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Solver.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;

namespace poseidon {

OptimizationUnit::~OptimizationUnit() = default;

// `value` in cost-model units on the axis 2^exponent scales them to.
static DPCost onAxis(double value, int exponent, StringRef what) {
  double q = std::ldexp(value, exponent);
  // 2^63 is the first double past the axis; every smaller magnitude rounds
  // into it.
  if (!(std::fabs(q) < 0x1p63)) {
    std::string Text;
    raw_string_ostream(Text) << value;
    report_fatal_error("Poseidon: " + Twine(what) + " " + Twine(Text) +
                       " does not fit the DP's cost axis");
  }
  return std::llround(q);
}

CostScale::CostScale(double profiledCost) : ProfiledCost(profiledCost) {
  if (!(profiledCost >= 0.0) || !std::isfinite(profiledCost)) {
    std::string Text;
    raw_string_ostream(Text) << profiledCost;
    report_fatal_error("Poseidon: the profiled cost " + Twine(Text) +
                       " cannot scale the DP's cost axis");
  }
  if (profiledCost > 0.0)
    Exponent = 32 - std::ilogb(profiledCost);
}

DPCost CostScale::toDPCost(double delta) const {
  // A solve the profile never reached costs nothing and saves nothing.
  if (delta == 0.0)
    return 0;
  if (ProfiledCost == 0.0) {
    std::string Text;
    raw_string_ostream(Text) << delta;
    report_fatal_error("Poseidon: a cost change of " + Twine(Text) +
                       " in a solve whose profiled run spent nothing");
  }
  return onAxis(delta, Exponent, "a cost change of");
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
  if (!std::isfinite(share)) {
    std::string Text;
    raw_string_ostream(Text) << share;
    report_fatal_error("Poseidon: a budget of " + Twine(Text) +
                       " of the profiled cost does not fit the DP's cost axis");
  }
  return onAxis(share * ProfiledCost, Exponent, "a budget of");
}

DPCost addDPCost(DPCost a, DPCost b) {
  if (std::optional<DPCost> sum = checkedAdd(a, b))
    return *sum;
  report_fatal_error("Poseidon: the cost " + Twine(a) + " + " + Twine(b) +
                     " leaves the DP's integer cost axis");
}

// Whether `a` and `b` cannot be in one strategy: they rewrite one output, or
// their footprints overlap and one of them consumes its footprint. Units that
// edit one subgraph in place compose: the precision unit is priced over what
// the algebraic units leave.
bool excludes(const OptimizationUnit &a, const OptimizationUnit &b) {
  if (a.rewritesSameOutputAs(b))
    return true;
  return (a.consumesFootprint() || b.consumesFootprint()) &&
         a.footprintOverlaps(b);
}

// Whether taking `later` is admitted or priced differently depending on
// whether `earlier` was taken. Neither reads which candidate `earlier` took.
bool reads(const OptimizationUnit &later, const OptimizationUnit &earlier) {
  return excludes(later, earlier) || later.pricedAgainst(earlier);
}

bool overlapsConsumed(const SolutionStep &step, ArrayRef<SolutionStep> chosen) {
  if (step.unit->consumesFootprint())
    return false;
  return llvm::any_of(chosen, [&](const SolutionStep &c) {
    return c.unit->consumesFootprint() && c.unit->footprintOverlaps(*step.unit);
  });
}

// A unit that does not consume its footprint displaces the prior steps that
// consume a footprint it overlaps and charges their deltas back: exact,
// because such a step is priced alone. A unit that consumes its footprint is
// refused by any prior step it overlaps instead, since a prior step may be
// priced against the steps before it. Without displacement the frontier keyed
// by cost evicts equal-cost prefixes that later units would have extended.
bool admit(ArrayRef<SolutionStep> prior, const OptimizationUnit &unit,
           const CostScale &scale, SmallVectorImpl<SolutionStep> &kept,
           DPCost &cost, double &error) {
  // The binary64 and binary32 searches of one output are separate units, but
  // only one rewrite of that output can be materialized.
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
      cost = addDPCost(
          cost, -scale.toDPCost(p.unit->costDelta(p.candidateIndex, {})));
      error -= p.unit->errorDelta(p.candidateIndex, {});
      continue;
    }
    kept.push_back(p);
  }
  return true;
}

static void paretoIntegrate(const CostScale &scale, CostMap &cost,
                            SolutionMap &sol, OptimizationUnit &unit) {
  CostMap newCost = cost;
  SolutionMap newSol = sol;
  for (const auto &p : cost) {
    DPCost curr = p.first;
    double currAcc = p.second;
    const SmallVector<SolutionStep> &chosen = sol[p.first];
    SmallVector<SolutionStep> priorSteps;
    if (!admit(chosen, unit, scale, priorSteps, curr, currAcc))
      continue;
    for (size_t i = 0; i < unit.numCandidates(); ++i) {
      DPCost cd = scale.toDPCost(unit.costDelta(i, priorSteps));
      auto ad = unit.errorDelta(i, priorSteps);
      // Compared against the post-displacement point, so a displacement that
      // gives back more than the candidate saves is dropped.
      DPCost nc = addDPCost(curr, cd);
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

static void paretoPrune(CostMap &cost, SolutionMap &sol) {
  CostMap pruned;
  SolutionMap prunedSol;
  for (const auto &l : cost) {
    DPCost lc = l.first;
    double la = l.second;
    bool dominated = false;
    for (const auto &r : cost) {
      DPCost rc = r.first;
      double ra = r.second;
      if (lc > rc && la >= ra) {
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

// -poseidon-early-prune=false. A partial solution is keyed by its cost and by
// its signature: the taken units some later unit still reads. Two with one
// signature are admitted and priced alike by every later unit, so one that is
// cheaper and no less accurate than the other stays so under every completion,
// and only such a one is dropped. After the last unit every signature is
// empty and the one remaining bucket is the frontier of all
// footprint-compatible strategies.
void solveExact(ArrayRef<OptimizationUnit *> units, const CostScale &scale,
                CostMap &cost, SolutionMap &sol) {
  const int n = units.size();
  SmallVector<int> lastRead(n, -1), canon(n);
  DenseMap<const OptimizationUnit *, int> unitIndex;
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

  using Signature = std::vector<int>;
  struct Entry {
    double acc;
    SmallVector<SolutionStep> steps;
  };
  using Bucket = std::map<DPCost, Entry>;
  auto insert = [](Bucket &b, DPCost c, double a,
                   const SmallVectorImpl<SolutionStep> &steps) {
    auto [it, fresh] = b.try_emplace(c);
    if (!fresh && !(it->second.acc > a))
      return;
    it->second.acc = a;
    it->second.steps.assign(steps.begin(), steps.end());
  };
  // paretoPrune's rule: an entry goes when a cheaper one is no less accurate.
  auto prune = [](Bucket &b) {
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
  };

  std::map<Signature, Bucket> table;
  insert(table[Signature()], cost.begin()->first, cost.begin()->second,
         sol.begin()->second);
  size_t peak = 1, peakSignatures = 1, pricings = 0;
  const auto start = std::chrono::steady_clock::now();
  for (int k = 0; k < n; ++k) {
    OptimizationUnit &unit = *units[k];
    const SolutionStep step(&unit, 0);
    auto stillRead = [&](const Signature &sig) {
      Signature kept;
      for (int u : sig)
        if (lastRead[u] > k)
          kept.push_back(u);
      return kept;
    };
    const size_t memoBefore = unit.memoizedPricings();

    std::map<Signature, Bucket> next;
    for (const auto &[sig, bucket] : table) {
      Bucket &into = next[stillRead(sig)];
      for (const auto &[c, e] : bucket)
        insert(into, c, e.acc, e.steps);
    }
    for (const auto &[sig, bucket] : table) {
      if (unit.numCandidates() == 0)
        break;
      // Everything a unit reads is in the signature, so one entry stands for
      // the bucket.
      const SmallVector<SolutionStep> &rep = bucket.begin()->second.steps;
      if (llvm::any_of(rep, [&](const SolutionStep &s) {
            return excludes(unit, *s.unit);
          }))
        continue;
      SmallVector<SolutionStep> priced;
      for (const SolutionStep &s : rep) {
        int u = canon[unitIndex.find(s.unit)->second];
        priced.push_back(SolutionStep(units[u], s.candidateIndex));
      }
      SmallVector<DPCost> dc;
      SmallVector<double> da;
      for (size_t i = 0; i < unit.numCandidates(); ++i) {
        dc.push_back(scale.toDPCost(unit.costDelta(i, priced)));
        da.push_back(unit.errorDelta(i, priced));
      }
      Signature taken = stillRead(sig);
      if (lastRead[canon[k]] > k) {
        taken.push_back(canon[k]);
        llvm::sort(taken);
      }
      Bucket &into = next[taken];
      SmallVector<SolutionStep> steps;
      for (const auto &[c, e] : bucket) {
        steps.assign(e.steps.begin(), e.steps.end());
        steps.push_back(step);
        for (size_t i = 0; i < unit.numCandidates(); ++i) {
          steps.back().candidateIndex = i;
          insert(into, addDPCost(c, dc[i]), e.acc + da[i], steps);
        }
      }
    }
    size_t entries = 0;
    for (auto &[sig, bucket] : next) {
      prune(bucket);
      entries += bucket.size();
    }
    table = std::move(next);
    pricings += unit.memoizedPricings() - memoBefore;
    if (entries > peak) {
      peak = entries;
      peakSignatures = table.size();
    }
    llvm::errs() << "##### Exact DP: finished unit " << k + 1 << " of " << n
                 << ": " << entries << " entries in " << table.size()
                 << " signatures #####\n";
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
  llvm::errs() << "Exact DP: " << cost.size() << " frontier points; peak "
               << peak << " entries in " << peakSignatures << " signatures; "
               << pricings << " precision changes priced; " << secs << " s\n";
}

void solvePruned(ArrayRef<OptimizationUnit *> units, const CostScale &scale,
                 CostMap &cost, SolutionMap &sol, bool progress) {
  StringMap<std::pair<size_t, size_t>> counts;
  for (const OptimizationUnit *unit : units)
    ++counts[unit->kindName()].second;
  for (OptimizationUnit *unit : units) {
    paretoIntegrate(scale, cost, sol, *unit);
    paretoPrune(cost, sol);
    if (!progress)
      continue;
    auto &[done, total] = counts[unit->kindName()];
    llvm::errs() << "##### Finished processing " << ++done << " of " << total
                 << " " << unit->kindName() << "s #####\n";
    llvm::errs() << "Current DP table sizes: " << cost.size() << "\n";
  }
}

} // namespace poseidon
