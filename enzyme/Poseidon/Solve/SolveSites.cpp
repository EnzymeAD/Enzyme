//===- SolveSites.cpp - the DP over a solve's units -----------------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BuildIdentity.h"
#include "Core/Session.h"
#include "Core/Technique.h"
#include "Model/CostModel.h"
#include "Solve.h"
#include "Support/Diagnostics.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/StructuralHash.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <vector>

using namespace llvm;

namespace poseidon {

// Pricing is also where candidates are dropped, and a table hit skips pricing,
// so the hit sees every candidate the table's positions were taken without; a
// cached step names its candidate as well as its position.
static std::string candidateKey(StringRef text) { return contentDigest(text); }

static std::optional<size_t> findCandidate(const SiteUnit &unit, size_t hint,
                                           StringRef key) {
  if (hint < unit.numCandidates() &&
      candidateKey(unit.candidateName(hint)) == key)
    return hint;
  std::optional<size_t> found;
  for (size_t i = 0; i < unit.numCandidates(); ++i) {
    if (candidateKey(unit.candidateName(i)) != key)
      continue;
    if (found)
      return std::nullopt;
    found = i;
  }
  return found;
}

namespace {
// A step of a cached frontier point: the unit by kind and position, and the
// candidate by position and, for a kind whose pricing drops candidates, by
// name.
struct StoredStep {
  std::string unit, candidateKey;
  size_t index = 0, candidate = 0;
};
bool fromJSON(const json::Value &V, StoredStep &S, json::Path P) {
  json::ObjectMapper O(V, P);
  return O && O.map("unit", S.unit) && O.map("index", S.index) &&
         O.map("candidate", S.candidate) &&
         O.mapOptional("candidate_key", S.candidateKey);
}
struct StoredPoint {
  DPCost cost = 0;
  double accuracyCost = 0.0;
  std::vector<StoredStep> steps;
};
bool fromJSON(const json::Value &V, StoredPoint &P, json::Path Path) {
  json::ObjectMapper O(V, Path);
  return O && O.map("cost", P.cost) && O.map("accuracy_cost", P.accuracyCost) &&
         O.map("steps", P.steps);
}
} // namespace

// The key holds everything the table is computed from, the code that prices
// and solves it included. The candidates are part of it although the inputs
// before them decide them: a Herbie search can end differently under the same
// inputs (a timeout). The entry is filed under the digest of the key without
// the compiler, so that a rebuilt compiler replaces the entries of the one
// before it.
static Error keySiteTable(Session &S, const SiteState &st,
                          ArrayRef<SiteUnit *> units, StringRef profilePath,
                          DPTable &table) {
  const Options &O = S.Opts;
  const Function &F = *st.F;
  CacheKey &key = table.key;
  std::string body;
  raw_string_ostream bodyOS(body);
  bodyOS << StructuralHash(F, /*DetailedHash=*/true);
  // StructuralHash does not see fast-math flags, and they change what a
  // rewrite costs.
  for (const Instruction &I : instructions(F))
    if (isa<FPMathOperator>(I)) {
      bodyOS << '|';
      I.getFastMathFlags().print(bodyOS);
    }
  key.addContent("function body", body);
  key.add("device", getCostModelNativeArch(S.get<CostModelState>()));
  for (const Technique *T : techniques())
    T->addToTableKey(key);
  key.addContent("cost model", costModelRows(S.get<CostModelState>()));
  key.add("cost axis", "cost-model units times 2^(32 - log2 profiled cost)");

  SmallVector<std::string, 4> profilePaths = {profilePath.str()};
  for (const std::string &dir : O.Profiling.OutputProfiles)
    profilePaths.push_back(siteProfilePath(S, F, dir));
  std::string profiles;
  for (const std::string &path : profilePaths) {
    auto buf = MemoryBuffer::getFile(path);
    if (!buf)
      return userError(F, Twine("cannot read the profile ") + path +
                              " again to key the DP table of " + F.getName() +
                              " (" + buf.getError().message() +
                              "); a profile directory must not change while a "
                              "compile reads it.");
    profiles += contentDigest((*buf)->getBuffer()) + "\n";
  }
  key.addContent("profile", profiles);

  for (const auto &[name, value] : solveFlagValues(O))
    key.add(("-" + name).str(), value);
  // Under a target a matrix product enters with its domain error, sampled
  // over another range by default.
  if (st.tau > 0.0 || O.Solver.Tau > 0.0)
    key.add("matrix product error", "domain error");

  std::string candidates;
  raw_string_ostream os(candidates);
  for (const SiteUnit *unit : units) {
    os << unit->keyHeader() << '\n';
    for (size_t i = 0; i < unit->numCandidates(); ++i)
      os << "  " << unit->candidateName(i) << '\n';
  }
  key.addContent("candidate set", candidates);

  SmallString<128> path(O.Herbie.Cache);
  sys::path::append(path, "dp_" + key.digest() + ".json");
  table.path = path.str().str();
  key.addCompiler(kPassIdentity);
  return Error::success();
}

// Binds the cached frontier `points` to `units` in `table`; false when a step
// names no candidate of this compile.
static bool bindTable(ArrayRef<StoredPoint> points, ArrayRef<SiteUnit *> units,
                      DPTable &table) {
  auto bind = [&](const StoredStep &s) -> std::optional<SolutionStep> {
    for (SiteUnit *unit : units) {
      if (unit->info().letter != s.unit || unit->indexInKind() != s.index)
        continue;
      if (!unit->info().pricingDrops) {
        if (s.candidate < unit->numCandidates())
          return SolutionStep(unit, s.candidate);
        return std::nullopt;
      }
      if (auto i = findCandidate(*unit, s.candidate, s.candidateKey))
        return SolutionStep(unit, *i);
      return std::nullopt;
    }
    return std::nullopt;
  };
  for (const StoredPoint &point : points) {
    table.costToAccuracyMap[point.cost] = point.accuracyCost;
    SmallVector<SolutionStep> &bound = table.costToSolutionMap[point.cost];
    for (const StoredStep &s : point.steps) {
      std::optional<SolutionStep> step = bind(s);
      if (!step)
        return false;
      bound.push_back(*step);
    }
  }
  return true;
}

Expected<std::unique_ptr<DPTable>> lookupDPTable(Session &S,
                                                 const SiteState &st,
                                                 ArrayRef<SiteUnit *> units,
                                                 StringRef profilePath) {
  auto table = std::make_unique<DPTable>();
  if (Error E = keySiteTable(S, st, units, profilePath, *table))
    return std::move(E);
  const StringRef fnName = st.F->getName();
  CacheEntryRead read = readCacheEntry(table->path, table->key);
  std::vector<StoredPoint> points;
  json::Path::Root parsed;
  const json::Object *value = read.value ? read.value->getAsObject() : nullptr;
  const json::Value *frontier = value ? value->get("frontier") : nullptr;
  if (read.value && (!frontier || !fromJSON(*frontier, points, parsed)))
    read.unreadable = "its table does not parse";
  else if (read.value && !bindTable(points, units, *table))
    read.unreadable = "a step names no candidate of this compile";
  if (!read.unreadable.empty()) {
    if (S.Opts.Diagnostics.Print)
      llvm::errs() << "[poseidon] DP cache entry for " << fnName
                   << " is unreadable (" << table->path << ": "
                   << read.unreadable << "); re-solving.\n";
    table->costToAccuracyMap.clear();
    table->costToSolutionMap.clear();
    return table;
  }
  if (!read.differs.empty()) {
    if (S.Opts.Diagnostics.Print)
      llvm::errs() << "[poseidon] DP cache entry for " << fnName << ' '
                   << (read.differs == "format"
                           ? "was written in another cache format"
                           : "was solved for a different " + read.differs)
                   << "; re-solving.\n";
    return table;
  }
  if (!read.value)
    return table;
  table->baseline = value->getNumber("baseline_accuracy_cost");
  table->hit = true;
  return table;
}

static void writeDPTable(const Options &O, const Function &F,
                         const DPTable &table, StringRef fnName,
                         const CostMap &costToAccuracyMap,
                         const SolutionMap &costToSolutionMap,
                         double baseline) {
  // llvm::json renders nan/inf as a bare `nan` token, which is not JSON. Refuse
  // to cache this function rather than drop entries: the loader treats a table
  // as a complete frontier, so a pruned one would change a later solve's
  // answer.
  unsigned nonFinite = 0;
  for (const auto &pair : costToAccuracyMap)
    if (!std::isfinite(pair.second))
      ++nonFinite;
  if (nonFinite) {
    if (O.Diagnostics.Print)
      llvm::errs() << "[poseidon] not caching the DP table for " << fnName
                   << ": " << nonFinite << " of " << costToAccuracyMap.size()
                   << " frontier points have a non-finite accuracy cost\n";
    return;
  }

  json::Array frontier;
  for (const auto &[cost, accuracyCost] : costToAccuracyMap) {
    json::Array steps;
    for (const SolutionStep &step : costToSolutionMap.at(cost)) {
      const SiteUnit &unit = unitOf(step);
      json::Object stored{
          {"unit", unit.info().letter},
          {"index", static_cast<int64_t>(unit.indexInKind())},
          {"candidate", static_cast<int64_t>(step.candidateIndex)}};
      if (unit.info().pricingDrops)
        stored["candidate_key"] =
            candidateKey(unit.candidateName(step.candidateIndex));
      steps.push_back(std::move(stored));
    }
    frontier.push_back(json::Object{{"cost", cost},
                                    {"accuracy_cost", accuracyCost},
                                    {"steps", std::move(steps)}});
  }
  json::Object value{{"site", fnName}, {"frontier", std::move(frontier)}};
  if (std::isfinite(baseline))
    value["baseline_accuracy_cost"] = baseline;

  Error E = Error::success();
  if (std::error_code EC = sys::fs::create_directories(O.Herbie.Cache))
    E = errorCodeToError(EC);
  else
    E = writeCacheEntry(table.path, table.key, std::move(value));
  if (E) {
    diagnose(F,
             "cannot write the DP cache entry " + table.path + ": " +
                 toString(std::move(E)),
             DS_Warning);
    return;
  }
  if (O.Diagnostics.Print)
    llvm::errs() << "[poseidon] DP table for " << fnName << " cached to "
                 << table.path << '\n';
}

// A site tolerance is a per-operation relative rounding level, so everything
// the ratio is built from has to be measured. `at` is the site the error is
// reported at.
static Error checkToleranceInputs(const Function &at, StringRef siteName,
                                  double accScale, double baseline) {
  if (!(accScale > 0.0) || !std::isfinite(accScale)) {
    std::string scaleStr;
    llvm::raw_string_ostream(scaleStr) << accScale;
    return userError(at,
                     Twine("site ") + siteName +
                         " was given an accuracy tolerance but its profile "
                         "carries a total sensitivity of " +
                         scaleStr +
                         ", so no rewrite's modelled error can be expressed "
                         "as a per-operation relative level; re-run profile "
                         "generation for this site");
  }

  if (!std::isfinite(baseline))
    return userError(at,
                     Twine("site ") + siteName +
                         " was given an accuracy tolerance but its own "
                         "baseline accuracy cost is not finite, so no "
                         "candidate's modelled error can be placed on an "
                         "absolute scale; give the site a cost budget "
                         "instead of a tolerance, or re-run profile "
                         "generation if its profile is not of this program");
  return Error::success();
}

StringRef selectorName(SolveSelector selector) {
  switch (selector) {
  case SolveSelector::DP:
    return "dp";
  case SolveSelector::Joint:
    return "joint";
  case SolveSelector::Pinned:
    return "pinned";
  }
  llvm_unreachable("every selector has a name");
}

// Whether a budget or an accuracy target can select a frontier point of this
// accuracy cost. A point whose cost is not finite stays on the frontier when
// nothing cheaper beats it, and no selection takes it.
static bool selectable(double accCost) { return std::isfinite(accCost); }

// Whether the target or the budget admits a frontier point of this cost and
// accuracy cost.
using Admits = function_ref<bool(DPCost cost, double acc)>;

// The most accurate selectable point within the budget. The frontier's
// accuracy cost falls as its cost rises, so each selectable point is the
// one its own share selects.
static std::optional<DPCost> selectWithinBudget(const CostMap &frontier,
                                                Admits withinBudget) {
  std::optional<DPCost> best;
  for (const auto &[cost, acc] : frontier)
    if (withinBudget(cost, acc) && selectable(acc) &&
        (!best || acc < frontier.at(*best)))
      best = cost;
  return best;
}

// The selectable point whose modelled relative error is within the target:
// the cheapest one under -poseidon-tau-cheapest, else the most accurate one.
// Two costs that differ by less than the cheapest point's steps can be
// mispriced by (costTieBand) are a tie: of the points within the target that
// save cost and are no costlier than that, the most accurate is taken, the
// cheaper of two equally accurate ones.
static std::optional<DPCost> selectUnderTarget(const Options &O,
                                               const CostScale &scale,
                                               const CostMap &frontier,
                                               const SolutionMap &solutions,
                                               Admits withinTarget) {
  auto meets = [&](DPCost cost, double acc) {
    return selectable(acc) && withinTarget(cost, acc);
  };
  std::optional<DPCost> best;
  for (const auto &[cost, acc] : frontier) {
    if (!meets(cost, acc))
      continue;
    if (!best) {
      best = cost;
      continue;
    }
    const double bestAcc = frontier.at(*best);
    if (O.Solver.TauCheapest
            ? (cost < *best || (cost == *best && acc < bestAcc))
            : acc < bestAcc)
      best = cost;
  }
  if (!best || !O.Solver.TauCheapest || *best >= 0)
    return best;
  double band = 0.0;
  for (const SolutionStep &step : solutions.at(*best))
    band += step.unit->costTieBand(step.candidateIndex);
  std::optional<DPCost> width = scale.toDPCost(band);
  if (!width)
    return best;
  DPCost tied = *best;
  for (auto it = frontier.upper_bound(*best);
       it != frontier.end() && it->first < 0 && it->first <= *best + *width;
       ++it)
    if (meets(it->first, it->second) && it->second < frontier.at(tied))
      tied = it->first;
  if (tied != *best && O.Diagnostics.Print)
    llvm::errs() << "[poseidon] cost " << scale.share(tied) << " ties with "
                 << scale.share(*best)
                 << " within the measured prices' band; the more accurate "
                    "point is selected\n";
  return tied;
}

static void printDPTable(const CostScale &scale,
                         const CostMap &costToAccuracyMap,
                         const SolutionMap &costToSolutionMap) {
  llvm::errs() << "[poseidon] DP table:\n";
  for (const auto &pair : costToAccuracyMap) {
    llvm::errs() << "  computation cost " << scale.share(pair.first)
                 << ", accuracy cost " << pair.second << ", steps:\n";
    auto it = costToSolutionMap.find(pair.first);
    if (it == costToSolutionMap.end())
      continue;
    for (const auto &step : it->second)
      llvm::errs() << "    " << unitOf(step).choiceText(step.candidateIndex)
                   << '\n';
  }
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
    llvm::errs() << "[poseidon] none of the " << numCandidates
                 << " candidate(s), alone or combined, is cheaper or more "
                    "accurate than the code as written, which is the only "
                    "frontier point.\n";
  else
    llvm::errs() << "[poseidon] the only frontier point is a solution of "
                 << only.size() << " step(s) from the " << numCandidates
                 << " candidate(s); it is no costlier and no less accurate "
                    "than the code as written and every other "
                    "combination.\n";
}

SmallVector<OptimizationUnit *>
solveOrder(const Options &O, ArrayRef<ArrayRef<SiteUnit *>> units) {
  SmallVector<SmallVector<OptimizationUnit *>, 4> siteUnits;
  for (ArrayRef<SiteUnit *> site : units)
    siteUnits.emplace_back(site.begin(), site.end());
  SmallVector<ArrayRef<OptimizationUnit *>, 4> sites(siteUnits.begin(),
                                                     siteUnits.end());
  return O.Solver.EarlyPrune ? prunedOrder(sites) : exactOrder(sites);
}

Expected<SmallVector<SolutionStep>>
solveSites(Session &S, ArrayRef<SiteState *> sites,
           ArrayRef<ArrayRef<SiteUnit *>> units, double tau, bool joint,
           SolveRecord *record) {
  const Options &O = S.Opts;
  const bool print = O.Diagnostics.Print;
  // Only a single site's table is cached: the key names one site's candidates.
  DPTable *table = sites.size() == 1
                       ? sites.front()->get<SolvedFrom>().table.get()
                       : nullptr;

  std::string names;
  double profiledCost = 0.0;
  double accScale = 0.0;
  for (SiteState *st : sites) {
    if (!names.empty())
      names += ",";
    names += st->F->getName().str();
    profiledCost += st->profiledCost;
    accScale += st->accScale;
  }
  const CostScale scale(profiledCost);

  if (print) {
    llvm::errs() << "[poseidon] Starting accuracy DP solver";
    if (joint)
      llvm::errs() << " over " << sites.size() << " site(s)";
    llvm::errs() << " with cost budget: " << O.Solver.CostBudget << " of "
                 << (joint ? "their" : "the site's") << " profiled cost\n";
    if (tau > 0.0)
      llvm::errs() << "[poseidon] Per-operation relative error tolerance: "
                   << tau << '\n';
  }

  CostMap costToAccuracyMap;
  SolutionMap costToSolutionMap;
  const bool loadedFromCache = table && table->hit;
  unsigned unpriced = 0;
  double baseline = 0.0;
  // The modelled error of the code as written. The DP table is relative to it
  // (costToAccuracyMap[0] == 0) while a tolerance is an absolute level.
  for (SiteState *st : sites)
    for (const Technique *T : techniques())
      baseline += T->baseline(*st, unpriced);

  if (loadedFromCache) {
    costToAccuracyMap = table->costToAccuracyMap;
    costToSolutionMap = table->costToSolutionMap;
    baseline =
        table->baseline.value_or(std::numeric_limits<double>::quiet_NaN());
    if (print)
      llvm::errs() << "[poseidon] Cache entry for " << names
                   << " found. Loaded DP tables from cache.\n";
  } else {
    if (print)
      llvm::errs() << "[poseidon] Solving the DP for " << names << ".\n";

    costToAccuracyMap[0] = 0;
    costToSolutionMap[0] = {};

    // The exact solve is the default. The pruned one compares partial
    // solutions that both constrain a later unit, which may price them
    // differently (a precision unit is priced against the rewrites chosen on
    // its subgraph), so its frontier can miss points the exact solve finds.
    Error searched =
        !O.Solver.EarlyPrune
            ? solveExact(solveOrder(O, units), scale, costToAccuracyMap,
                         costToSolutionMap, /*progress=*/print)
            : solvePruned(solveOrder(O, units), scale, costToAccuracyMap,
                          costToSolutionMap, /*progress=*/print);
    if (searched)
      return std::move(searched);
    for (ArrayRef<SiteUnit *> site : units)
      for (SiteUnit *unit : site)
        if (Error E = unit->takePricingError())
          return std::move(E);

    if (table)
      writeDPTable(O, *sites.front()->F, *table, names, costToAccuracyMap,
                   costToSolutionMap, baseline);
  }

  if (O.Diagnostics.PrintDPTable)
    printDPTable(scale, costToAccuracyMap, costToSolutionMap);

  if (record) {
    record->selector = joint ? SolveSelector::Joint : SolveSelector::DP;
    record->fromCache = loadedFromCache;
    record->table = costToAccuracyMap;
    record->solutions = costToSolutionMap;
    record->baseline = baseline;
  }

  size_t numCandidates = 0;
  double compositions = 1.0;
  for (ArrayRef<SiteUnit *> site : units)
    for (const SiteUnit *unit : site) {
      numCandidates += unit->numCandidates();
      if (!unit->consumesFootprint())
        // +1 for the "do nothing" possibility
        compositions *= unit->numCandidates() + 1;
    }
  if (print) {
    llvm::errs() << "[poseidon] Critical computation cost range: ["
                 << scale.share(costToAccuracyMap.begin()->first) << ", "
                 << scale.share(costToAccuracyMap.rbegin()->first) << "]\n";
    llvm::errs() << "[poseidon] DP table contains " << costToAccuracyMap.size()
                 << " entries.\n";
    if (costToAccuracyMap.size() == 1)
      noteSinglePointFrontier(costToSolutionMap, numCandidates);
    llvm::errs() << "[poseidon] Total candidate compositions: " << compositions
                 << '\n';
    if (costToSolutionMap.count(0) && costToSolutionMap.at(0).empty())
      llvm::errs() << "[poseidon] the no-op solution (cost budget used 0) is "
                      "Pareto-optimal\n";
  }

  if (tau > 0.0) {
    if (unpriced && !loadedFromCache && print)
      llvm::errs() << "[poseidon] " << names << ": " << unpriced
                   << " subgraph output(s) carry no priced candidate, so the "
                      "baseline below covers only the priced ones and the "
                      "reported relative error is a lower bound.\n";
    if (Error E =
            checkToleranceInputs(*sites.front()->F, names, accScale, baseline))
      return std::move(E);
  }
  const DPCost budget = tau > 0.0 ? 0 : scale.fromShare(O.Solver.CostBudget);
  auto admits = [&](DPCost cost, double acc) {
    return tau > 0.0 ? (baseline + acc) / accScale <= tau : cost <= budget;
  };
  const std::optional<DPCost> best =
      tau > 0.0 ? selectUnderTarget(O, scale, costToAccuracyMap,
                                    costToSolutionMap, admits)
                : selectWithinBudget(costToAccuracyMap, admits);
  if (record)
    for (const auto &[cost, acc] : costToAccuracyMap)
      record->verdicts[cost] = cost == best        ? PointVerdict::Selected
                               : !selectable(acc)  ? PointVerdict::Unselectable
                               : admits(cost, acc) ? PointVerdict::Outranked
                                                   : PointVerdict::Excluded;

  if (tau > 0.0) {
    if (!best) {
      double bestAchievable = std::numeric_limits<double>::infinity();
      for (const auto &pair : costToAccuracyMap)
        bestAchievable = std::min(bestAchievable, pair.second);
      if (print)
        llvm::errs() << "[poseidon] No solution found that meets accuracy "
                        "tolerance "
                     << tau << "; the best relative accuracy in the DP table "
                     << "is " << (baseline + bestAchievable) / accScale << '\n';
      return SmallVector<SolutionStep>();
    }
    const double accCost = costToAccuracyMap.at(*best);
    if (print)
      llvm::errs() << "[poseidon] " << names << ": tau=" << tau
                   << " S=" << accScale << " A0=" << baseline
                   << " selected cost=" << scale.share(*best)
                   << " accCost=" << accCost
                   << " rel=" << (baseline + accCost) / accScale << '\n';
  } else {
    if (!best) {
      if (print)
        llvm::errs() << "[poseidon] No solution found within the computation "
                        "cost budget\n";
      return SmallVector<SolutionStep>();
    }
    if (print) {
      llvm::errs() << "[poseidon] Minimum accuracy cost within budget: "
                   << costToAccuracyMap.at(*best) << '\n';
      llvm::errs() << "[poseidon] Computation cost budget used: "
                   << scale.share(*best) << '\n';
    }
  }

  assert(costToSolutionMap.count(*best) && "every frontier point has a "
                                           "solution");
  if (record)
    record->selected = *best;
  return costToSolutionMap.at(*best);
}

} // namespace poseidon
