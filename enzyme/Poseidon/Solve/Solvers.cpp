//=- Solvers.cpp - Solver utilities for Poseidon --------------------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements solver-related utilities for the Poseidon optimization
// pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/xxhash.h"

#include "Model/CostModel.h"
#include "Optimize.h"
#include "Pass/Session.h"
#include "Report.h"
#include "SiteTable.h"
#include "Sites/SiteIdentity.h"
#include "Solvers.h"

#include "IR/Utils.h"
#include "Techniques/Herbie.h"
#include "Techniques/Matmul/Matmul.h"
#include "Techniques/Precision.h"

#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/StructuralHash.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <chrono>
#include <set>

using namespace llvm;

namespace poseidon {

// A site tolerance is a per-operation relative rounding level, so everything
// the ratio is built from has to be real and measured; a missing piece is an
// upstream defect, never something to substitute a value for. `at` is the
// site the error is reported at.
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

// The DP table is anchored at the ORIGINAL program: costToAccuracyMap[0] == 0
// and every entry is a sum of deltas. A tolerance is an ABSOLUTE level, so the
// site's own modelled error has to be added back. Each subgraph contributes it
// once: a PrecisionUnit's perOutputInitialAccCost already covers all of its
// subgraph's outputs (getAdjustedAccCostDelta excludes the ones a
// RewriteUnit claims), so the per-output RewriteUnit baselines are only
// counted for subgraphs that have no PrecisionUnit at all. MatmulUnit
// baselines are zero by construction.
double baselineAccCost(ArrayRef<RewriteUnit> COs, ArrayRef<PrecisionUnit> CSs,
                       unsigned *unpricedOut) {
  SmallPtrSet<const Subgraph *, 4> subgraphsWithCS;
  double baseline = 0.0;
  for (const auto &CS : CSs) {
    baseline += CS.initialAccCost;
    subgraphsWithCS.insert(CS.subgraph);
  }
  unsigned unpriced = 0;
  SmallPtrSet<const Value *, 8> arrayPriced;
  for (const auto &CO : COs) {
    if (subgraphsWithCS.count(CO.subgraph))
      continue;
    // A RewriteUnit with no candidates was never priced (Herbie produced
    // nothing for it), so its share of the baseline was never measured. It
    // cannot appear in any solution either; the site's baseline is reported as
    // the part that WAS measured, and the caller says so.
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

// Write the shares of the selectable frontier points to <cache>/budgets.txt
// for the Pareto-variant workflow, on every solve and replay. Parallel variant
// builds reusing a cached table write the same shares, and the replace is
// atomic, so none of them reads a partial file.
static void writeBudgetsFile(const Options &O, const CostScale &scale,
                             const CostMap &costToAccuracyMap) {
  if (O.Herbie.Cache.empty())
    return;
  std::string budgetsFile = O.Herbie.Cache + "/budgets.txt";
  std::string budgetsStr;
  for (const auto &pair : costToAccuracyMap)
    if (selectable(O, pair.second))
      budgetsStr += formatv("{0},", format("%.17g", scale.share(pair.first)));
  if (!budgetsStr.empty())
    budgetsStr.pop_back();
  if (Error E = writeFileAtomically(budgetsFile, budgetsStr))
    llvm::errs() << "Error writing " << budgetsFile << ": "
                 << toString(std::move(E)) << "\n";
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
                 << "\n";
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
    if (totalCands > 0)
      llvm::errs() << "[poseidon] NOTE: " << totalCands
                   << " candidate(s) synthesized but NONE lands on the Pareto "
                      "frontier (all dominated by the baseline under this "
                      "device's cost model); the no-op solution is genuinely "
                      "optimal here.\n";
  }

  double totalCandidateCompositions = 1.0;
  for (const SiteUnit *unit : units)
    if (!unit->consumesFootprint())
      // +1 for the "do nothing" possibility
      totalCandidateCompositions *= unit->numCandidates() + 1;
  llvm::errs() << "Total candidate compositions: " << totalCandidateCompositions
               << "\n";

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
                   << (baseline + bestAchievable) / accScale << "\n";
      return {};
    }

    llvm::errs() << "[poseidon] " << F.getName() << ": tau=" << errorTol
                 << " S=" << accScale << " A0=" << baseline
                 << " selected cost=" << scale.share(bestCompCost)
                 << " accCost=" << minAccCost << " rel=" << bestRel << "\n";
  } else {
    std::optional<DPCost> best =
        selectWithinBudget(O, scale, costToAccuracyMap);

    // A kept whole-kernel candidate has positive cost and worse model accuracy
    // than the baseline, so min-accuracy selection can never choose it; an
    // exact positive-budget match dispatches it directly.

    if (!best) {
      llvm::errs() << "No solution found within the computation cost budget!\n";
      return {};
    }
    bestCompCost = *best;
    minAccCost = costToAccuracyMap.at(*best);

    llvm::errs() << "Minimum accuracy cost within budget: " << minAccCost
                 << "\n";
    llvm::errs() << "Computation cost budget used: "
                 << scale.share(bestCompCost) << "\n";
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

  // Same integration order as accuracyDPSolver. The joint table is not cached
  // (table.json is per candidate set); budgets.txt is still written.
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
    if (totalCands > 0)
      llvm::errs() << "[poseidon] NOTE: " << totalCands
                   << " candidate(s) synthesized but NONE lands on the Pareto "
                      "frontier (all dominated by the baseline under this "
                      "device's cost model); the no-op solution is genuinely "
                      "optimal here.\n";
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
                   << (baseline + bestAchievable) / accScale << "\n";
      return {};
    }
    llvm::errs() << "[poseidon] " << names << ": tau=" << errorTol
                 << " S=" << accScale << " A0=" << baseline
                 << " selected cost=" << scale.share(bestCompCost)
                 << " accCost=" << minAccCost << " rel=" << bestRel << "\n";
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
               << "\n";

  assert(costToSolutionMap.find(bestCompCost) != costToSolutionMap.end() &&
         "joint DP solver: expected a solution!");
  if (record)
    record->selected = bestCompCost;
  return SmallVector<SolutionStep>(costToSolutionMap[bestCompCost].begin(),
                                   costToSolutionMap[bestCompCost].end());
}

} // namespace poseidon
