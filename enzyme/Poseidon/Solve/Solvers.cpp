//===- Solvers.cpp - the DP over a solve's units --------------------------===//
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

#include "Solvers.h"
#include "IR/Utils.h"
#include "Optimize.h"
#include "Pass/Session.h"
#include "Report.h"
#include "SiteTable.h"
#include "Techniques/Herbie.h"
#include "Techniques/Matmul/Matmul.h"
#include "Techniques/Precision.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace poseidon {

// A site tolerance is a per-operation relative rounding level, so everything
// the ratio is built from has to be measured. `at` is the site the error is
// reported at.
static void checkToleranceInputs(const Function &at, StringRef siteName,
                                 double accScale, double baseline) {
  if (!(accScale > 0.0) || !std::isfinite(accScale)) {
    std::string scaleStr;
    llvm::raw_string_ostream(scaleStr) << accScale;
    reportUserError(at, at.getSubprogram(),
                    Twine("site ") + siteName +
                        " was given an accuracy tolerance but its profile "
                        "carries a total sensitivity of " +
                        scaleStr +
                        ", so no rewrite's modelled error can be expressed "
                        "as a per-operation relative level; re-run profile "
                        "generation for this site.");
  }

  if (!std::isfinite(baseline))
    reportUserError(at, at.getSubprogram(),
                    Twine("site ") + siteName +
                        " was given an accuracy tolerance but its own "
                        "baseline accuracy cost is not finite, so no "
                        "candidate's modelled error can be placed on an "
                        "absolute scale; give the site a cost budget "
                        "instead of a tolerance, or re-run profile "
                        "generation if its profile is not of this program.");
}

// The modelled error of the code as written. The DP table is relative to it
// (costToAccuracyMap[0] == 0) while a tolerance is an absolute level. Each
// subgraph counts once: a PrecisionUnit's initial cost covers every output of
// its subgraph, so RewriteUnit baselines count only on subgraphs without one.
// A MatmulUnit's baseline is zero. `unpricedOut` receives the number of
// RewriteUnits, and of outputs of PrecisionUnits, left out for want of a
// price.
double baselineAccCost(ArrayRef<RewriteUnit> COs, ArrayRef<PrecisionUnit> CSs,
                       unsigned *unpricedOut) {
  SmallPtrSet<const Subgraph *, 4> subgraphsWithCS;
  double baseline = 0.0;
  unsigned unpriced = 0;
  for (const auto &CS : CSs) {
    baseline += CS.initialAccCost;
    subgraphsWithCS.insert(CS.subgraph);
    if (!CS.unpricedReason.empty())
      unpriced += CS.subgraph->outputs.size();
  }
  SmallPtrSet<const Value *, 8> arrayPriced;
  for (const auto &CO : COs) {
    if (subgraphsWithCS.count(CO.subgraph))
      continue;
    // A RewriteUnit with no candidates was never priced, so its share of the
    // baseline is unknown; it cannot appear in a solution either.
    if (CO.candidates.empty()) {
      ++unpriced;
      continue;
    }
    // An array core prices all of its outputs at once; each output's baseline
    // is counted once however many cores cover it.
    if (llvm::any_of(CO.roots,
                     [&](const Value *V) { return arrayPriced.contains(V); }))
      continue;
    if (CO.isArray())
      arrayPriced.insert(CO.roots.begin(), CO.roots.end());
    baseline += CO.initialAccCost;
  }
  if (unpricedOut)
    *unpricedOut = unpriced;
  return baseline;
}

bool selectable(const Options &O, double accCost) {
  return std::isfinite(accCost) && accCost < O.Model.ExponentPenalty;
}

// The most accurate selectable point within the budget. The frontier's
// accuracy cost falls as its cost rises, so each selectable point is the
// one its own share selects.
static std::optional<DPCost> selectWithinBudget(const Options &O,
                                                const CostScale &scale,
                                                const CostMap &frontier) {
  const DPCost budget = scale.fromShare(O.Solver.CostBudget);
  std::optional<DPCost> best;
  for (const auto &[cost, acc] : frontier)
    if (cost <= budget && selectable(O, acc) &&
        (!best || acc < frontier.at(*best)))
      best = cost;
  return best;
}

// Writes the shares of the selectable frontier points, comma-separated, to
// <cache>/budgets.txt on every solve and replay. The replace is atomic, so
// parallel builds that reuse a cached table never read a partial file.
static void writeBudgetsFile(const Options &O, const CostScale &scale,
                             const CostMap &costToAccuracyMap) {
  if (O.Herbie.Cache.empty())
    return;
  std::string budgetsFile = O.Herbie.Cache + "/budgets.txt";
  std::string budgetsStr;
  for (const auto &pair : costToAccuracyMap)
    if (selectable(O, pair.second))
      budgetsStr += formatv("{0},", exactDouble(scale.share(pair.first)));
  if (!budgetsStr.empty())
    budgetsStr.pop_back();
  if (Error E = writeFileAtomically(budgetsFile, budgetsStr))
    llvm::errs() << "Error writing " << budgetsFile << ": "
                 << toString(std::move(E)) << '\n';
}

// A frontier of one point either is the code as written, which no
// combination of candidates is cheaper or more accurate than, or is one
// solution that is no costlier and no less accurate than every other
// combination, the code as written included.
static void noteSinglePointFrontier(const SolutionMap &frontier,
                                    size_t numCandidates) {
  if (numCandidates == 0)
    return;
  const SmallVector<SolutionStep> &only = frontier.begin()->second;
  if (only.empty())
    llvm::errs() << "[poseidon] NOTE: none of the " << numCandidates
                 << " candidate(s), alone or combined, is cheaper or more "
                    "accurate than the code as written, which is the only "
                    "frontier point.\n";
  else
    llvm::errs() << "[poseidon] NOTE: the only frontier point is a solution "
                    "of "
                 << only.size() << " step(s) from the " << numCandidates
                 << " candidate(s); it is no costlier and no less accurate "
                    "than the code as written and every other "
                    "combination.\n";
}

SmallVector<SolutionStep>
accuracyDPSolver(Session &S, Function &F, FunctionFPState &st,
                 ArrayRef<SiteUnit *> units, double errorTol, double accScale,
                 SolveRecord *record, SiteTable *table) {
  const Options &O = S.Opts;
  auto &COs = st.COs;
  auto &CSs = st.CSs;
  llvm::errs() << "Starting accuracy DP solver with cost budget: "
               << O.Solver.CostBudget << " of the site's profiled cost\n";
  if (errorTol > 0.0) {
    llvm::errs() << "Per-operation relative error tolerance: " << errorTol
                 << '\n';
  }

  const CostScale scale(st.profiledCost);
  CostMap costToAccuracyMap;
  SolutionMap costToSolutionMap;

  const bool loadedFromCache = table && table->hit;
  std::optional<double> cachedBaseline;

  if (loadedFromCache) {
    costToAccuracyMap = table->costToAccuracyMap;
    costToSolutionMap = table->costToSolutionMap;
    cachedBaseline = table->baseline;
    llvm::errs() << "Cache entry for " << F.getName()
                 << " found. Loaded DP tables from cache.\n";
  } else {
    llvm::errs() << "Solving the DP for " << F.getName() << ".\n";

    costToAccuracyMap[0] = 0;
    costToSolutionMap[0] = {};

    SmallVector<OptimizationUnit *> site(units.begin(), units.end());
    if (!O.Solver.EarlyPrune) {
      solveExact(exactOrder({site}), scale, costToAccuracyMap,
                 costToSolutionMap);
    } else {
      // Each prune also compares partial solutions a later unit treats
      // differently (a CS is priced against the COs chosen on its subgraph),
      // so the frontier can miss points the exact solve finds.
      solvePruned(prunedOrder({site}), scale, costToAccuracyMap,
                  costToSolutionMap,
                  /*progress=*/true);
    }

    if (table)
      writeSiteTable(O, F.getName(), table->key, costToAccuracyMap,
                     costToSolutionMap, baselineAccCost(COs, CSs),
                     st.profiledCost);
  }

  if (O.Diagnostics.Print && O.Diagnostics.ShowTable)
    printDPTable(scale, costToAccuracyMap, costToSolutionMap);

  writeBudgetsFile(O, scale, costToAccuracyMap);

  if (!loadedFromCache)
    emitReport(S, F.getName(), scale, costToAccuracyMap, costToSolutionMap,
               units);

  if (record) {
    record->selector = "dp";
    record->fromCache = loadedFromCache;
    record->table = costToAccuracyMap;
    record->solutions = costToSolutionMap;
    record->accScale = accScale;
    record->profiledCost = st.profiledCost;
    record->tau = errorTol;
    record->baseline =
        loadedFromCache
            ? cachedBaseline.value_or(std::numeric_limits<double>::quiet_NaN())
            : baselineAccCost(COs, CSs);
  }

  llvm::errs() << "Critical computation cost range: ["
               << scale.share(costToAccuracyMap.begin()->first) << ", "
               << scale.share(costToAccuracyMap.rbegin()->first) << "]\n";

  llvm::errs() << "DP table contains " << costToAccuracyMap.size()
               << " entries.\n";

  if (costToAccuracyMap.size() == 1) {
    size_t totalCands = 0;
    for (const SiteUnit *unit : units)
      totalCands += unit->numCandidates();
    noteSinglePointFrontier(costToSolutionMap, totalCands);
  }

  double totalCandidateCompositions = 1.0;
  for (const SiteUnit *unit : units)
    if (!unit->consumesFootprint())
      // +1 for the "do nothing" possibility
      totalCandidateCompositions *= unit->numCandidates() + 1;
  llvm::errs() << "Total candidate compositions: " << totalCandidateCompositions
               << '\n';

  if (costToSolutionMap.find(0) != costToSolutionMap.end()) {
    if (costToSolutionMap[0].empty()) {
      llvm::errs() << "WARNING: No-op solution (utilized cost budget = 0) is "
                      "considered Pareto-optimal.\n";
    }
  }

  double minAccCost = std::numeric_limits<double>::infinity();
  DPCost bestCompCost = 0;

  if (errorTol > 0.0) {
    unsigned unpriced = 0;
    double baseline =
        loadedFromCache
            ? cachedBaseline.value_or(std::numeric_limits<double>::quiet_NaN())
            : baselineAccCost(COs, CSs, &unpriced);
    if (unpriced)
      llvm::errs() << "[poseidon] " << F.getName() << ": " << unpriced
                   << " subgraph output(s) carry no priced candidate, so the "
                      "baseline below covers only the priced ones and the "
                      "reported relative error is a lower bound.\n";
    checkToleranceInputs(F, F.getName(), accScale, baseline);

    bool foundSolution = false;
    double bestRel = 0.0;

    for (const auto &pair : costToAccuracyMap) {
      DPCost compCost = pair.first;
      double accCost = pair.second;
      double rel = (baseline + accCost) / accScale;

      if (selectable(O, accCost) && rel <= errorTol) {
        const bool better =
            O.Solver.TauCheapest
                ? (compCost < bestCompCost ||
                   (compCost == bestCompCost && accCost < minAccCost))
                : (accCost < minAccCost);
        if (!foundSolution || better) {
          minAccCost = accCost;
          bestCompCost = compCost;
          bestRel = rel;
          foundSolution = true;
        }
      }
    }

    if (!foundSolution) {
      double bestAchievable = std::numeric_limits<double>::infinity();
      for (const auto &pair : costToAccuracyMap)
        bestAchievable = std::min(bestAchievable, pair.second);
      llvm::errs() << "No solution found that meets accuracy tolerance "
                   << errorTol << "!\n";
      llvm::errs() << "Best achievable relative accuracy in DP table: "
                   << (baseline + bestAchievable) / accScale << '\n';
      return {};
    }

    llvm::errs() << "[poseidon] " << F.getName() << ": tau=" << errorTol
                 << " S=" << accScale << " A0=" << baseline
                 << " selected cost=" << scale.share(bestCompCost)
                 << " accCost=" << minAccCost << " rel=" << bestRel << '\n';
  } else {
    std::optional<DPCost> best =
        selectWithinBudget(O, scale, costToAccuracyMap);

    if (!best) {
      llvm::errs() << "No solution found within the computation cost budget!\n";
      return {};
    }
    bestCompCost = *best;
    minAccCost = costToAccuracyMap.at(*best);

    llvm::errs() << "Minimum accuracy cost within budget: " << minAccCost
                 << '\n';
    llvm::errs() << "Computation cost budget used: "
                 << scale.share(bestCompCost) << '\n';
  }

  assert(costToSolutionMap.find(bestCompCost) != costToSolutionMap.end() &&
         "[poseidon] DP solver: expected a solution!");

  if (record)
    record->selected = bestCompCost;
  return SmallVector<SolutionStep>(costToSolutionMap[bestCompCost].begin(),
                                   costToSolutionMap[bestCompCost].end());
}

SmallVector<SolutionStep>
jointAccuracyDPSolver(Session &S, ArrayRef<FunctionFPState *> states,
                      double errorTol, SolveRecord *record) {
  const Options &O = S.Opts;
  llvm::errs() << "Starting JOINT accuracy DP solver over " << states.size()
               << " function(s) with shared cost budget: "
               << O.Solver.CostBudget << " of their profiled cost\n";
  double profiledCost = 0.0;
  for (auto *st : states)
    profiledCost += st->profiledCost;
  const CostScale scale(profiledCost);

  CostMap costToAccuracyMap;
  SolutionMap costToSolutionMap;
  costToAccuracyMap[0] = 0;
  costToSolutionMap[0] = {};

  // The joint table is not cached: table.json keys one site's candidate set.
  SmallVector<SmallVector<OptimizationUnit *>, 4> siteUnits;
  for (auto *st : states)
    siteUnits.emplace_back(st->units.begin(), st->units.end());
  SmallVector<ArrayRef<OptimizationUnit *>, 4> sites(siteUnits.begin(),
                                                     siteUnits.end());
  if (!O.Solver.EarlyPrune) {
    solveExact(exactOrder(sites), scale, costToAccuracyMap, costToSolutionMap);
  } else {
    solvePruned(prunedOrder(sites), scale, costToAccuracyMap, costToSolutionMap,
                /*progress=*/false);
  }

  llvm::errs() << "JOINT DP table contains " << costToAccuracyMap.size()
               << " entries; cost range ["
               << scale.share(costToAccuracyMap.begin()->first) << ", "
               << scale.share(costToAccuracyMap.rbegin()->first) << "]\n";

  writeBudgetsFile(O, scale, costToAccuracyMap);

  if (record) {
    record->selector = "joint";
    record->table = costToAccuracyMap;
    record->solutions = costToSolutionMap;
    record->tau = errorTol;
    record->profiledCost = profiledCost;
    record->baseline = 0.0;
    record->accScale = 0.0;
    for (auto *st : states) {
      record->baseline += baselineAccCost(st->COs, st->CSs);
      record->accScale += st->accScale;
    }
  }

  if (costToSolutionMap.count(0) && costToSolutionMap[0].empty())
    llvm::errs() << "WARNING: No-op solution (budget = 0) is Pareto-optimal.\n";

  if (costToAccuracyMap.size() == 1) {
    size_t totalCands = 0;
    for (auto *st : states)
      for (const SiteUnit *unit : st->units)
        totalCands += unit->numCandidates();
    noteSinglePointFrontier(costToSolutionMap, totalCands);
  }

  double minAccCost = std::numeric_limits<double>::infinity();
  DPCost bestCompCost = 0;
  if (errorTol > 0.0) {
    // The joint table is never cached, so both the baseline and the scale are
    // summed fresh over the sites it spans.
    double baseline = 0.0;
    double accScale = 0.0;
    std::string names;
    for (auto *st : states) {
      baseline += baselineAccCost(st->COs, st->CSs);
      accScale += st->accScale;
      if (!names.empty())
        names += ",";
      names += st->F ? st->F->getName().str() : std::string("<unnamed>");
    }
    checkToleranceInputs(*states.front()->F, names, accScale, baseline);

    bool found = false;
    double bestRel = 0.0;
    for (const auto &pair : costToAccuracyMap) {
      double rel = (baseline + pair.second) / accScale;
      if (!selectable(O, pair.second) || rel > errorTol)
        continue;
      const bool better =
          O.Solver.TauCheapest
              ? (pair.first < bestCompCost ||
                 (pair.first == bestCompCost && pair.second < minAccCost))
              : (pair.second < minAccCost);
      if (!found || better) {
        minAccCost = pair.second;
        bestCompCost = pair.first;
        bestRel = rel;
        found = true;
      }
    }
    if (!found) {
      double bestAchievable = std::numeric_limits<double>::infinity();
      for (const auto &pair : costToAccuracyMap)
        bestAchievable = std::min(bestAchievable, pair.second);
      llvm::errs() << "JOINT: no solution meets accuracy tolerance " << errorTol
                   << "; best achievable relative accuracy "
                   << (baseline + bestAchievable) / accScale << '\n';
      return {};
    }
    llvm::errs() << "[poseidon] " << names << ": tau=" << errorTol
                 << " S=" << accScale << " A0=" << baseline
                 << " selected cost=" << scale.share(bestCompCost)
                 << " accCost=" << minAccCost << " rel=" << bestRel << '\n';
  } else {
    std::optional<DPCost> best =
        selectWithinBudget(O, scale, costToAccuracyMap);
    if (!best) {
      llvm::errs()
          << "JOINT: no solution within the computation cost budget!\n";
      return {};
    }
    bestCompCost = *best;
    minAccCost = costToAccuracyMap.at(*best);
  }
  llvm::errs() << "JOINT: minimum accuracy cost within budget: " << minAccCost
               << "; computation cost used: " << scale.share(bestCompCost)
               << '\n';

  assert(costToSolutionMap.find(bestCompCost) != costToSolutionMap.end() &&
         "joint DP solver: expected a solution!");
  if (record)
    record->selected = bestCompCost;
  return SmallVector<SolutionStep>(costToSolutionMap[bestCompCost].begin(),
                                   costToSolutionMap[bestCompCost].end());
}

} // namespace poseidon
