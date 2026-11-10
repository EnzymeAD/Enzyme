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

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/xxhash.h"
#if __has_include("llvm/Support/VCSRevision.h")
#include "llvm/Support/VCSRevision.h"
#endif

#include "BuildIdentity.h"
#include "CostModel.h"
#include "Optimize.h"
#include "Pass/Session.h"
#include "Solvers.h"
#include "Types.h"

#include "Herbie.h"
#include "Techniques/Matmul/Matmul.h"
#include "Techniques/Precision.h"
#include "Utils.h"

#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/StructuralHash.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <chrono>
#include <random>
#include <set>

using namespace llvm;

namespace poseidon {

// Dispatch and raise prices are measured wall-clock ratios that repeat to about
// 0.6% run to run, so a smaller modelled cost gap is a tie.
static constexpr double kCostTieBandRel = 0.006;

// Pricing is also where candidates are dropped, and a table hit skips pricing,
// so the hit sees every candidate the table's positions were taken without; a
// cached step names its candidate as well as its position.
static std::string candidateKey(StringRef text) {
  return utohexstr(xxh3_64bits(text));
}

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

// A site tolerance is a per-operation relative rounding level, so everything
// the ratio is built from has to be real and measured; a missing piece is an
// upstream defect, never something to substitute a value for.
static void checkToleranceInputs(StringRef siteName, double accScale,
                                 double baseline) {
  if (!(accScale > 0.0) || !std::isfinite(accScale)) {
    std::string scaleStr;
    llvm::raw_string_ostream(scaleStr) << accScale;
    report_fatal_error(
        Twine("Poseidon: site ") + siteName +
        " was given an accuracy tolerance but its profile carries a total "
        "sensitivity of " +
        scaleStr +
        ", so no rewrite's modelled error can be expressed as a "
        "per-operation relative level; re-run profile generation for this "
        "site.");
  }

  if (!std::isfinite(baseline))
    report_fatal_error(
        Twine("Poseidon: site ") + siteName +
        " was given an accuracy tolerance but its own baseline accuracy cost "
        "is not finite, so no candidate's modelled error can be placed on an "
        "absolute scale.");
}

// The DP table is anchored at the ORIGINAL program: costToAccuracyMap[0] == 0
// and every entry is a sum of deltas. A tolerance is an ABSOLUTE level, so the
// site's own modelled error has to be added back. Each subgraph contributes it
// once: a CandidateSubgraph's perOutputInitialAccCost already covers all of its
// subgraph's outputs (getAdjustedAccCostDelta excludes the ones a
// CandidateOutput claims), so the per-output CandidateOutput baselines are only
// counted for subgraphs that have no CandidateSubgraph at all. CandidateMatmul
// baselines are zero by construction.
static double baselineAccCost(ArrayRef<CandidateOutput> COs,
                              ArrayRef<CandidateSubgraph> CSs,
                              unsigned *unpricedOut = nullptr) {
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
    // A CandidateOutput with no candidates was never priced (Herbie produced
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

static std::string tablePath(const Options &O) {
  return O.Herbie.Cache + "/table.json";
}

namespace {
struct StoredStep {
  std::string itemType, candidateKey;
  size_t itemIndex = 0, candidateIndex = 0;
};
bool fromJSON(const json::Value &V, StoredStep &S, json::Path P) {
  json::ObjectMapper O(V, P);
  return O && O.map("itemType", S.itemType) &&
         O.map("itemIndex", S.itemIndex) &&
         O.map("candidateIndex", S.candidateIndex) &&
         O.mapOptional("candidateKey", S.candidateKey);
}
} // namespace

// Everything a site's DP table is computed from, the code that prices and
// solves it included. The candidates are part of it although the inputs before
// them decide them: a Herbie search can end differently under the same inputs
// (a timeout), and under an accuracy target the matrix products are left out
// of the table.
static CacheKey solveKey(Session &S, const FunctionFPState &st,
                         ArrayRef<SiteUnit *> units, StringRef profilePath) {
  const Options &O = S.Opts;
  const Function &F = *st.F;
  CacheKey key;
  // Two LLVM snapshots report one version string; the revision, which an LLVM
  // built without it does not define, tells them apart.
  std::string compiler =
      std::string(kPassIdentity) + " LLVM " + LLVM_VERSION_STRING;
#ifdef LLVM_REVISION
  compiler += " " LLVM_REVISION;
#endif
  key.add("compiler", compiler);
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
  key.add("device", getCostModelNativeArch(S.CostModel));
  key.add("cost model", utohexstr(getCostModelFingerprint(S.CostModel)));

  SmallVector<std::string, 4> profilePaths = {profilePath.str()};
  for (const std::string &dir : O.Profiling.OutputProfiles) {
    SmallString<128> path(dir);
    sys::path::append(path, siteProfileStem(S, F) + ".fpprofile");
    profilePaths.push_back(path.str().str());
  }
  std::string profiles;
  for (const std::string &path : profilePaths) {
    auto buf = MemoryBuffer::getFile(path);
    if (!buf)
      report_fatal_error(Twine("Poseidon: cannot read the profile ") + path +
                         " again to key the DP table of " + F.getName());
    profiles += contentDigest((*buf)->getBuffer()) + "\n";
  }
  key.addContent("profile", profiles);

  for (const auto &[name, value] : solveFlagValues(O))
    key.add(("-" + name).str(), value);
  // Only a site that sent Herbie cores depends on which Herbie answered them.
  if (!st.COs.empty())
    addHerbieKeyParts(S, key);

  std::string candidates;
  raw_string_ostream os(candidates);
  for (const SiteUnit *unit : units) {
    os << unit->keyHeader() << "\n";
    for (size_t i = 0; i < unit->numCandidates(); ++i)
      os << "  " << unit->candidateName(i) << "\n";
  }
  key.addContent("candidate set", candidates);
  return key;
}

// Fills `table` from the cache when it holds a table for `fnName` under
// table.key, and otherwise says why not unless there is none at all.
static bool readSiteTable(const Options &O, StringRef fnName, SiteTable &table,
                          ArrayRef<SiteUnit *> units) {
  const std::string path = tablePath(O);
  if (!sys::fs::exists(path))
    return false;
  auto miss = [&](const std::string &why) {
    llvm::errs() << "[poseidon] DP cache entry for " << fnName << " " << why
                 << "; re-solving.\n";
    return false;
  };
  auto unreadable = [&](const std::string &why) {
    return miss("is unreadable (" + path + ": " + why + ")");
  };

  auto buf = MemoryBuffer::getFile(path);
  if (!buf)
    return unreadable(buf.getError().message());
  Expected<json::Value> root = json::parse((*buf)->getBuffer());
  if (!root)
    return unreadable(toString(root.takeError()));
  json::Object *funcs = root->getAsObject()
                            ? root->getAsObject()->getObject("functions")
                            : nullptr;
  if (!funcs)
    return miss("was written in another cache format");
  json::Object *entry = funcs->getObject(fnName);
  if (!entry)
    return false;
  if (std::optional<std::string> part =
          table.key.firstDifference(entry->getObject("key")))
    return miss(*part == "format" ? "was written in another cache format"
                                  : "was solved for a different " + *part);

  std::map<std::string, double> accMap;
  std::map<std::string, std::vector<StoredStep>> solMap;
  json::Path::Root parsed;
  const json::Value *accVal = entry->get("costToAccuracyMap");
  const json::Value *solVal = entry->get("costToSolutionMap");
  if (!accVal || !solVal || !fromJSON(*accVal, accMap, parsed) ||
      !fromJSON(*solVal, solMap, parsed))
    return unreadable("its table does not parse");
  auto cost = [](StringRef text) -> std::optional<InstructionCost> {
    int64_t c;
    if (text.getAsInteger(10, c))
      return std::nullopt;
    return InstructionCost(c);
  };
  // Steps name their candidates by key: pricing drops candidates and a hit
  // prices nothing, so a solved table's positions are not the replay's.
  auto bind = [&](const StoredStep &s) -> std::optional<SolutionStep> {
    for (SiteUnit *unit : units) {
      if (unit->info().tag != s.itemType || unit->indexInKind() != s.itemIndex)
        continue;
      if (!unit->info().pricingDrops) {
        if (s.candidateIndex < unit->numCandidates())
          return SolutionStep(unit, s.candidateIndex);
        return std::nullopt;
      }
      if (auto i = findCandidate(*unit, s.candidateIndex, s.candidateKey))
        return SolutionStep(unit, *i);
      return std::nullopt;
    }
    return std::nullopt;
  };
  for (const auto &[text, acc] : accMap) {
    std::optional<InstructionCost> c = cost(text);
    if (!c)
      return unreadable("its table does not parse");
    table.costToAccuracyMap[*c] = acc;
  }
  for (const auto &[text, steps] : solMap) {
    std::optional<InstructionCost> c = cost(text);
    if (!c)
      return unreadable("its table does not parse");
    SmallVector<SolutionStep> &bound = table.costToSolutionMap[*c];
    for (const StoredStep &s : steps) {
      std::optional<SolutionStep> step = bind(s);
      if (!step)
        return unreadable("a step names no candidate of this compile");
      bound.push_back(*step);
    }
  }
  table.baseline = entry->getNumber("baselineAccCost");
  table.hit = true;
  return true;
}

static void
writeSiteTable(const Options &O, StringRef fnName, const CacheKey &key,
               const std::map<InstructionCost, double> &costToAccuracyMap,
               const std::map<InstructionCost, SmallVector<SolutionStep>>
                   &costToSolutionMap,
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
    llvm::errs()
        << "[poseidon] NOT caching the DP table for " << fnName << ": "
        << nonFinite << " of " << costToAccuracyMap.size()
        << " frontier points have a non-finite accuracy cost, so the "
           "table is not serializable and, more to the point, not "
           "authoritative. This function will be re-solved on the next "
           "compile; every other function's table is untouched.\n";
    return;
  }

  json::Object entryObj;
  entryObj["key"] = key.toJSON();

  json::Object costAccMap;
  for (const auto &pair : costToAccuracyMap) {
    costAccMap[std::to_string(GET_INSTRUCTION_COST(pair.first))] = pair.second;
  }
  entryObj["costToAccuracyMap"] = std::move(costAccMap);

  json::Object costSolMap;
  for (const auto &pair : costToSolutionMap) {
    json::Array stepsArray;
    for (const auto &step : pair.second) {
      const SiteUnit &unit = unitOf(step);
      json::Object stepObj;
      stepObj["candidateIndex"] = static_cast<int64_t>(step.candidateIndex);
      stepObj["itemType"] = unit.info().tag;
      stepObj["itemIndex"] = static_cast<int64_t>(unit.indexInKind());
      if (unit.info().pricingDrops)
        stepObj["candidateKey"] =
            candidateKey(unit.candidateName(step.candidateIndex));
      stepsArray.push_back(std::move(stepObj));
    }
    costSolMap[std::to_string(GET_INSTRUCTION_COST(pair.first))] =
        std::move(stepsArray);
  }
  entryObj["costToSolutionMap"] = std::move(costSolMap);
  if (std::isfinite(baseline))
    entryObj["baselineAccCost"] = baseline;

  // Read-modify-write: the file holds one table per function, so a later site
  // in the same compilation unit must extend it rather than clobber the
  // earlier sites' tables. A file that does not parse is replaced.
  const std::string path = tablePath(O);
  json::Object functions;
  if (auto fileOrErr = MemoryBuffer::getFile(path)) {
    if (auto prior = json::parse(fileOrErr.get()->getBuffer())) {
      if (json::Object *priorObj = prior->getAsObject())
        if (json::Object *priorFuncs = priorObj->getObject("functions"))
          functions = std::move(*priorFuncs);
    } else {
      consumeError(prior.takeError());
    }
  }
  functions[fnName] = std::move(entryObj);

  json::Object jsonObj;
  jsonObj["functions"] = std::move(functions);

  if (Error E = writeFileAtomically(
          path,
          formatv("{0:2}", json::Value(std::move(jsonObj))).str() + "\n")) {
    llvm::errs() << "Error writing cache file: " << toString(std::move(E))
                 << "\n";
    return;
  }
  llvm::errs() << "DP table for " << fnName << " cached to file.\n";
}

static void priceCandidates(FunctionFPState &st) {
  for (SiteUnit *unit : st.units)
    unit->price();
}

SmallVector<SiteUnit *> solvedUnits(const FunctionFPState &st, bool target) {
  SmallVector<SiteUnit *> units;
  for (SiteUnit *unit : st.units)
    if (!target || !isa<CandidateMatmul>(unit))
      units.push_back(unit);
  return units;
}

void FunctionFPState::sealUnits(Session &S) {
  assert(!unitsSealed && "a site's units are sealed once");
  unitsSealed = true;
  for (size_t i = 0; i < COs.size(); ++i) {
    COs[i].bind(S, *this, i);
    units.push_back(&COs[i]);
  }
  for (size_t i = 0; i < CSs.size(); ++i) {
    CSs[i].bind(S, *this, i);
    units.push_back(&CSs[i]);
  }
  for (size_t i = 0; i < CMs.size(); ++i) {
    CMs[i].bind(S, *this, i);
    units.push_back(&CMs[i]);
  }
}

void priceUnlessCached(Session &S, FunctionFPState &st, StringRef profilePath) {
  const Options &O = S.Opts;
  // A joint or a manual solve reads no table, and an accuracy target leaves
  // the matrix products to errorBudgetSelector (fpOptimize).
  if (!O.Herbie.Cache.empty() && !O.Solver.JointDP &&
      O.Solver.ApplyRewrites.empty()) {
    const bool target = st.errTol > 0.0 || O.Solver.Tau > 0.0;
    const SmallVector<SiteUnit *> units = solvedUnits(st, target);
    st.table = std::make_unique<SiteTable>();
    st.table->key = solveKey(S, st, units, profilePath);
    if (readSiteTable(O, st.F->getName(), *st.table, units))
      return;
  }
  priceCandidates(st);
}

static void
printDPTable(const std::map<InstructionCost, double> &costToAccuracyMap,
             const std::map<InstructionCost, SmallVector<SolutionStep>>
                 &costToSolutionMap) {
  llvm::errs() << "\n*** DP Table ***\n";
  for (const auto &pair : costToAccuracyMap) {

    llvm::errs() << "Computation cost: " << pair.first
                 << ", Accuracy cost: " << pair.second << "\n";
    llvm::errs() << "\tSolution steps: \n";
    auto it = costToSolutionMap.find(pair.first);
    if (it == costToSolutionMap.end())
      continue;
    for (const auto &step : it->second)
      llvm::errs() << "\t\t" << unitOf(step).choiceText(step.candidateIndex)
                   << "\n";
  }
  llvm::errs() << "*** End of DP Table ***\n\n";
}

static void
emitReport(Session &S, StringRef funcName,
           const std::map<InstructionCost, double> &costToAccuracyMap,
           const std::map<InstructionCost, SmallVector<SolutionStep>>
               &costToSolutionMap,
           ArrayRef<SiteUnit *> units) {
  const Options &O = S.Opts;
  if (O.Diagnostics.ReportPath.empty())
    return;

  std::error_code EC;
  if (!llvm::sys::fs::exists(O.Diagnostics.ReportPath)) {
    EC = llvm::sys::fs::create_directories(O.Diagnostics.ReportPath);
    if (EC) {
      llvm::errs() << "Error creating report directory: " << EC.message()
                   << "\n";
      return;
    }
  }

  json::Object report;
  report["function"] = funcName.str();
  report["num_pareto_points"] = static_cast<int64_t>(costToAccuracyMap.size());

  if (!costToAccuracyMap.empty()) {
    report["cost_range_min"] =
        GET_INSTRUCTION_COST(costToAccuracyMap.begin()->first);
    report["cost_range_max"] =
        GET_INSTRUCTION_COST(costToAccuracyMap.rbegin()->first);
  }

  for (const UnitKindInfo &kind : unitKinds()) {
    if (kind.reportSection.empty())
      continue;
    json::Array summary;
    for (SiteUnit *unit : units)
      if (unit->unitKind() == kind.kind)
        unit->addReportSummary(summary);
    report[kind.reportSection] = std::move(summary);
  }

  json::Array paretoPoints;
  for (const auto &pair : costToAccuracyMap) {
    json::Object point;
    point["computation_cost"] = GET_INSTRUCTION_COST(pair.first);
    point["accuracy_cost"] = pair.second;

    auto it = costToSolutionMap.find(pair.first);
    if (it != costToSolutionMap.end()) {
      json::Array steps;
      for (const auto &step : it->second) {
        steps.push_back(unitOf(step).stepJSON(step.candidateIndex));
      }
      point["steps"] = std::move(steps);
    }
    paretoPoints.push_back(std::move(point));
  }
  report["pareto_points"] = std::move(paretoPoints);

  std::string jsonFile =
      (Twine(O.Diagnostics.ReportPath) + "/" + funcName + ".json").str();
  raw_fd_ostream jsonOut(jsonFile, EC, sys::fs::OF_Text);
  if (EC) {
    llvm::errs() << "Error writing JSON report: " << EC.message() << "\n";
  } else {
    jsonOut << formatv("{0:2}", json::Value(std::move(report))) << "\n";
    llvm::errs() << "Poseidon JSON report written to " << jsonFile << "\n";
  }

  std::string textFile =
      (Twine(O.Diagnostics.ReportPath) + "/" + funcName + ".txt").str();
  raw_fd_ostream textOut(textFile, EC, sys::fs::OF_Text);
  if (EC) {
    llvm::errs() << "Error writing text report: " << EC.message() << "\n";
    return;
  }

  textOut << "=== Poseidon Report: " << funcName << " ===\n";
  textOut << "Pareto table: " << costToAccuracyMap.size() << " points";
  if (!costToAccuracyMap.empty()) {
    textOut << ", cost range ["
            << GET_INSTRUCTION_COST(costToAccuracyMap.begin()->first) << ", "
            << GET_INSTRUCTION_COST(costToAccuracyMap.rbegin()->first) << "]";
  }
  textOut << "\n";
  bool first = true;
  for (const UnitKindInfo &kind : unitKinds()) {
    if (kind.reportSection.empty())
      continue;
    textOut << (first ? "" : ", ") << kind.reportTitle << ": "
            << llvm::count_if(units, [&](const SiteUnit *unit) {
                 return unit->unitKind() == kind.kind;
               });
    first = false;
  }
  textOut << "\n\n";
  for (const UnitKindInfo &kind : unitKinds()) {
    if (kind.reportSection.empty())
      continue;
    for (SiteUnit *unit : units)
      if (unit->unitKind() == kind.kind)
        unit->writeReportSummary(textOut);
    textOut << "\n";
  }

  unsigned pointIdx = 0;
  for (const auto &pair : costToAccuracyMap) {
    textOut << "--- Pareto Point #" << pointIdx++
            << ": Cost=" << GET_INSTRUCTION_COST(pair.first)
            << ", Accuracy=" << pair.second << " ---\n";
    auto it = costToSolutionMap.find(pair.first);
    if (it != costToSolutionMap.end() && !it->second.empty()) {
      for (const auto &step : it->second) {
        unitOf(step).writeReportStep(textOut, step.candidateIndex, "  ");
      }
    } else {
      textOut << "  (no changes)\n";
    }
    textOut << "\n";
  }

  llvm::errs() << "Poseidon text report written to " << textFile << "\n";

  {
    std::string configFile =
        (Twine(O.Diagnostics.ReportPath) + "/validate_config.json").str();
    {
      json::Object cfg;
      cfg["function"] = funcName.str();
      cfg["profile_path"] = O.Profiling.ProfileUse;
      cfg["cache_path"] = O.Herbie.Cache;
      json::Array budgetArr;
      for (const auto &pair : costToAccuracyMap)
        budgetArr.push_back(GET_INSTRUCTION_COST(pair.first));
      cfg["budgets"] = std::move(budgetArr);
      json::Array accArr;
      for (const auto &pair : costToAccuracyMap)
        accArr.push_back(pair.second);
      cfg["estimated_accuracy_costs"] = std::move(accArr);

      raw_fd_ostream cfgOut(configFile, EC, sys::fs::OF_Text);
      if (EC) {
        llvm::errs() << "Error writing validate config: " << EC.message()
                     << "\n";
      } else {
        cfgOut << formatv("{0:2}", json::Value(std::move(cfg))) << "\n";
      }
    }
  }

  {
    json::Array rewrites;
    for (SiteUnit *unit : units)
      unit->addCuratedRewrites(rewrites);

    auto rewriteVec =
        SmallVector<json::Value>(rewrites.begin(), rewrites.end());
    llvm::sort(rewriteVec, [](const json::Value &a, const json::Value &b) {
      auto *ao = a.getAsObject();
      auto *bo = b.getAsObject();
      StringRef aCat = ao->getString("category").value_or("");
      StringRef bCat = bo->getString("category").value_or("");
      auto catRank = [](StringRef c) -> int {
        if (c == "free_win")
          return 0;
        if (c == "accuracy_for_speed")
          return 1;
        if (c == "speed_for_accuracy")
          return 2;
        return 3;
      };
      int ar = catRank(aCat), br = catRank(bCat);
      if (ar != br)
        return ar < br;
      double ae = ao->getNumber("efficiency").value_or(0);
      double be = bo->getNumber("efficiency").value_or(0);
      return ae > be;
    });

    json::Array sortedRewrites;
    for (auto &v : rewriteVec)
      sortedRewrites.push_back(std::move(v));

    std::string rewritesFile =
        (Twine(O.Diagnostics.ReportPath) + "/" + funcName + "_rewrites.json")
            .str();
    raw_fd_ostream rwOut(rewritesFile, EC, sys::fs::OF_Text);
    if (EC) {
      llvm::errs() << "Error writing rewrites report: " << EC.message() << "\n";
    } else {
      json::Object root;
      root["function"] = funcName.str();
      root["total_rewrites"] = static_cast<int64_t>(sortedRewrites.size());
      root["rewrites"] = std::move(sortedRewrites);
      rwOut << formatv("{0:2}", json::Value(std::move(root))) << "\n";
      llvm::errs() << "Poseidon curated rewrites written to " << rewritesFile
                   << "\n";
    }
  }
}

SmallVector<SolutionStep>
errorBudgetSelector(const Options &O, SmallVector<CandidateMatmul, 4> &CMs,
                    double budget, double confidence) {
  SmallVector<SolutionStep> steps;
  for (auto &cm : CMs) {
    long best = -1;
    double bestComp = cm.initialCompCost;
    for (size_t i = 0; i < cm.candidates.size(); ++i) {
      const CandidateMatmul::Option &c = cm.candidates[i];
      if (c.domainError <= budget && c.compCost < bestComp) {
        bestComp = c.compCost;
        best = (long)i;
      }
    }
    // Inside the calibration noise band two costs are a tie: among qualifying
    // candidates within band of the cheapest, the winner is the smallest
    // modelled domainError, then the smallest compCost, then the lowest index.
    // The guard never widens the qualifying set.
    if (best >= 0) {
      const double band = bestComp + kCostTieBandRel * std::fabs(bestComp);
      long tieBest = best;
      for (size_t i = 0; i < cm.candidates.size(); ++i) {
        const CandidateMatmul::Option &c = cm.candidates[i];
        if (c.domainError > budget || c.compCost >= cm.initialCompCost)
          continue;
        if (c.compCost > band)
          continue;
        const CandidateMatmul::Option &b = cm.candidates[tieBest];
        if (c.domainError < b.domainError ||
            (c.domainError == b.domainError && c.compCost < b.compCost))
          tieBest = (long)i;
      }
      if (tieBest != best && O.Diagnostics.Print)
        llvm::errs() << "Poseidon error-budget: cost tie inside "
                     << (kCostTieBandRel * 100.0) << "% calibration band ("
                     << matmulOptionLabel(cm.candidates[best]) << " cost "
                     << cm.candidates[best].compCost << " vs "
                     << matmulOptionLabel(cm.candidates[tieBest]) << " cost "
                     << cm.candidates[tieBest].compCost
                     << ") -> broken on modelled domain error ("
                     << cm.candidates[tieBest].domainError << " < "
                     << cm.candidates[best].domainError << ")\n";
      best = tieBest;
    }
    if (best < 0) {
      if (O.Diagnostics.Print)
        llvm::errs() << "Poseidon error-budget: matmul -> baseline F64 (no "
                        "rewrite clears "
                     << budget << " at confidence " << confidence
                     << " faster than F64)\n";
      continue;
    }
    if (O.Diagnostics.Print)
      llvm::errs() << "Poseidon error-budget: matmul -> "
                   << matmulOptionLabel(cm.candidates[best]) << " (domain err "
                   << cm.candidates[best].domainError << " <= " << budget
                   << " at confidence " << confidence << ")\n";
    steps.emplace_back(&cm, (size_t)best);
  }
  return steps;
}

// Write the DP table's cost breakpoints to <cache>/budgets.txt for the Pareto-
// variant workflow, on every solve and replay. Parallel variant builds reusing
// a cached table write the same breakpoints, and the replace is atomic, so
// none of them reads a partial file.
static void
writeBudgetsFile(const Options &O,
                 const std::map<InstructionCost, double> &costToAccuracyMap) {
  if (O.Herbie.Cache.empty())
    return;
  std::string budgetsFile = O.Herbie.Cache + "/budgets.txt";
  std::string budgetsStr;
  for (const auto &pair : costToAccuracyMap)
    budgetsStr += std::to_string(GET_INSTRUCTION_COST(pair.first)) + ",";
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
  llvm::errs() << "Starting accuracy DP solver with computation budget: "
               << O.Solver.CompCostBudget << "\n";
  if (errorTol > 0.0) {
    llvm::errs() << "Per-operation relative error tolerance: " << errorTol
                 << "\n";
  }

  using CostMap = std::map<InstructionCost, double>;
  using SolutionMap = std::map<InstructionCost, SmallVector<SolutionStep>>;

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
      solveExact(exactOrder({site}), costToAccuracyMap, costToSolutionMap);
    } else {
      // Each prune also compares partial solutions a later unit treats
      // differently (a CS is priced against the COs chosen on its subgraph),
      // so the frontier can miss points the exact solve finds.
      solvePruned(prunedOrder({site}), costToAccuracyMap, costToSolutionMap,
                  /*progress=*/true);
    }

    if (table)
      writeSiteTable(O, F.getName(), table->key, costToAccuracyMap,
                     costToSolutionMap, baselineAccCost(COs, CSs));
  }

  if (O.Diagnostics.Print && O.Diagnostics.ShowTable)
    printDPTable(costToAccuracyMap, costToSolutionMap);

  writeBudgetsFile(O, costToAccuracyMap);

  if (!loadedFromCache)
    emitReport(S, F.getName(), costToAccuracyMap, costToSolutionMap, units);

  if (record) {
    record->selector = "dp";
    record->fromCache = loadedFromCache;
    record->table = costToAccuracyMap;
    record->solutions = costToSolutionMap;
    record->accScale = accScale;
    record->tau = errorTol;
    record->baseline =
        loadedFromCache
            ? cachedBaseline.value_or(std::numeric_limits<double>::quiet_NaN())
            : baselineAccCost(COs, CSs);
  }

  llvm::errs() << "Critical computation cost range: ["
               << costToAccuracyMap.begin()->first << ", "
               << costToAccuracyMap.rbegin()->first << "]\n";

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
  InstructionCost bestCompCost = 0;

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
    checkToleranceInputs(F.getName(), accScale, baseline);

    bool foundSolution = false;
    double bestRel = 0.0;

    for (const auto &pair : costToAccuracyMap) {
      InstructionCost compCost = pair.first;
      double accCost = pair.second;
      double rel = (baseline + accCost) / accScale;

      if (std::isfinite(accCost) && accCost < O.Model.ExponentPenalty &&
          rel <= errorTol) {
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
                 << " selected cost=" << bestCompCost
                 << " accCost=" << minAccCost << " rel=" << bestRel << "\n";
  } else {
    for (const auto &pair : costToAccuracyMap) {
      InstructionCost compCost = pair.first;
      double accCost = pair.second;

      if (compCost <= O.Solver.CompCostBudget && std::isfinite(accCost) &&
          accCost < O.Model.ExponentPenalty && accCost < minAccCost) {
        minAccCost = accCost;
        bestCompCost = compCost;
      }
    }

    // A kept whole-kernel candidate has positive cost and worse model accuracy
    // than the baseline, so min-accuracy selection can never choose it; an
    // exact positive-budget match dispatches it directly.

    if (minAccCost == std::numeric_limits<double>::infinity()) {
      llvm::errs() << "No solution found within the computation cost budget!\n";
      return {};
    }

    llvm::errs() << "Minimum accuracy cost within budget: " << minAccCost
                 << "\n";
    llvm::errs() << "Computation cost budget used: " << bestCompCost << "\n";
  }

  assert(costToSolutionMap.find(bestCompCost) != costToSolutionMap.end() &&
         "[poseidon] DP solver: expected a solution!");

  if (record)
    record->selected = GET_INSTRUCTION_COST(bestCompCost);
  return SmallVector<SolutionStep>(costToSolutionMap[bestCompCost].begin(),
                                   costToSolutionMap[bestCompCost].end());
}

SmallVector<SolutionStep>
jointAccuracyDPSolver(Session &S, ArrayRef<FunctionFPState *> states,
                      double errorTol, SolveRecord *record) {
  const Options &O = S.Opts;
  llvm::errs() << "Starting JOINT accuracy DP solver over " << states.size()
               << " function(s) with shared computation budget: "
               << O.Solver.CompCostBudget << "\n";

  using CostMap = std::map<InstructionCost, double>;
  using SolutionMap = std::map<InstructionCost, SmallVector<SolutionStep>>;
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
    solveExact(exactOrder(sites), costToAccuracyMap, costToSolutionMap);
  } else {
    solvePruned(prunedOrder(sites), costToAccuracyMap, costToSolutionMap,
                /*progress=*/false);
  }

  llvm::errs() << "JOINT DP table contains " << costToAccuracyMap.size()
               << " entries; cost range [" << costToAccuracyMap.begin()->first
               << ", " << costToAccuracyMap.rbegin()->first << "]\n";

  writeBudgetsFile(O, costToAccuracyMap);

  if (record) {
    record->selector = "joint";
    record->table = costToAccuracyMap;
    record->solutions = costToSolutionMap;
    record->tau = errorTol;
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
  InstructionCost bestCompCost = 0;
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
    checkToleranceInputs(names, accScale, baseline);

    bool found = false;
    double bestRel = 0.0;
    for (const auto &pair : costToAccuracyMap) {
      double rel = (baseline + pair.second) / accScale;
      if (!std::isfinite(pair.second) ||
          pair.second >= O.Model.ExponentPenalty || rel > errorTol)
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
                 << " selected cost=" << bestCompCost
                 << " accCost=" << minAccCost << " rel=" << bestRel << "\n";
  } else {
    for (const auto &pair : costToAccuracyMap)
      if (pair.first <= O.Solver.CompCostBudget && pair.second < minAccCost) {
        minAccCost = pair.second;
        bestCompCost = pair.first;
      }
    if (minAccCost == std::numeric_limits<double>::infinity()) {
      llvm::errs()
          << "JOINT: no solution within the computation cost budget!\n";
      return {};
    }
  }
  llvm::errs() << "JOINT: minimum accuracy cost within budget: " << minAccCost
               << "; computation cost used: " << bestCompCost << "\n";

  assert(costToSolutionMap.find(bestCompCost) != costToSolutionMap.end() &&
         "joint DP solver: expected a solution!");
  if (record)
    record->selected = GET_INSTRUCTION_COST(bestCompCost);
  return SmallVector<SolutionStep>(costToSolutionMap[bestCompCost].begin(),
                                   costToSolutionMap[bestCompCost].end());
}

static std::optional<ApplyItem> parseUnitRef(StringRef s) {
  ApplyItem item;
  const UnitKindInfo *kind =
      llvm::find_if(unitKinds(), [&](const UnitKindInfo &k) {
        return s.starts_with(k.idPrefix);
      });
  if (kind == unitKinds().end())
    return std::nullopt;
  s = s.drop_front(kind->idPrefix.size());
  item.kind = kind->kind;
  size_t digits = s.find_first_not_of("0123456789");
  if (digits == 0 || digits == StringRef::npos ||
      s.take_front(digits).getAsInteger(10, item.unit))
    return std::nullopt;
  s = s.drop_front(digits);
  if (s.consume_front("=")) {
    if (s.empty())
      return std::nullopt;
    item.name = s.str();
    return item;
  }
  size_t ordinal;
  if (!s.consume_front("_") || s.getAsInteger(10, ordinal))
    return std::nullopt;
  item.ordinal = ordinal;
  return item;
}

// A name may itself hold ':' (a Herbie precision annotation), so the unit is
// read first and a site prefix only when the item does not start with one.
static ApplyItem parseApplyItem(StringRef text) {
  std::optional<ApplyItem> item = parseUnitRef(text);
  if (!item) {
    auto [site, rest] = text.split(':');
    if (!site.empty() && (item = parseUnitRef(rest)))
      item->site = site.str();
  }
  if (!item)
    report_fatal_error("[poseidon] -poseidon-apply-rewrites: cannot read '" +
                       Twine(text) +
                       "'; an item is [<site>:]<unit>_<ordinal> or "
                       "[<site>:]<unit>=<name>, <unit> being R<i>, PT<i> or "
                       "M<i>");
  item->text = text.str();
  return *item;
}

static std::vector<ApplyItem> &applyItems(Session &S) {
  PinState &P = S.Pins;
  if (P.spec != S.Opts.Solver.ApplyRewrites) {
    P.spec = S.Opts.Solver.ApplyRewrites;
    P.items.clear();
    SmallVector<StringRef> texts;
    StringRef(P.spec).split(texts, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    for (StringRef text : texts)
      P.items.push_back(parseApplyItem(text.trim()));
  }
  return P.items;
}

static bool namesSite(const ApplyItem &item, StringRef key,
                      ArrayRef<std::string> aliases) {
  return item.site.empty() || item.site == key ||
         is_contained(aliases, item.site);
}

static std::string listSites(const std::set<std::string> &keys) {
  std::string out;
  for (const std::string &key : keys) {
    out += "\n  " + key;
    std::string shown = llvm::demangle(key);
    if (shown != key)
      out += " (" + shown + ")";
  }
  return out;
}

// Sites are noted before they are solved, so a name two sites answer to is
// refused before the second one applies it.
void noteSiteForPins(Session &S, const Function &body) {
  if (S.Opts.Solver.ApplyRewrites.empty())
    return;
  PinState &P = S.Pins;
  std::string key = siteKeyOfBody(body);
  if (is_contained(P.sites, key))
    return;
  P.sites.push_back(key);
  SmallVector<std::string, 2> aliases = siteAliases(key);
  for (ApplyItem &item : applyItems(S)) {
    if (item.site.empty() || !namesSite(item, key, aliases))
      continue;
    item.named.insert(key);
    if (item.named.size() > 1)
      report_fatal_error("[poseidon] -poseidon-apply-rewrites item '" +
                             Twine(item.text) + "' names the site '" +
                             item.site +
                             "', which more than one site of this module "
                             "answers to; give one of them by its symbol:" +
                             listSites(item.named),
                         /*gen_crash_diag=*/false);
  }
}

void checkPinnedSites(Session &S) {
  PinState &P = S.Pins;
  // The host half of a CUDA or HIP unit is given the items too, and the sites
  // they name are the device half's.
  if (!P.sites.empty())
    for (const ApplyItem &item : applyItems(S))
      if (!item.site.empty() && item.named.empty())
        report_fatal_error(
            "[poseidon] -poseidon-apply-rewrites item '" + Twine(item.text) +
                "' names the site '" + item.site +
                "', which is no site of this module. Its sites are:" +
                listSites(
                    std::set<std::string>(P.sites.begin(), P.sites.end())),
            /*gen_crash_diag=*/false);
  P.sites.clear();
  P.printed.clear();
  for (ApplyItem &item : P.items)
    item.named.clear();
}

// The position of the candidate an item names in a unit's list. A candidate
// pricing dropped is not in the list, and is told apart from a name the unit
// never had.
static size_t findNamed(const ApplyItem &item, StringRef site, StringRef unit,
                        const SiteUnit &cands) {
  std::optional<size_t> found;
  for (size_t i = 0; i < cands.numCandidates(); ++i) {
    if (cands.candidateName(i) != item.name)
      continue;
    if (found)
      report_fatal_error("[poseidon] " + Twine(site) + ": " + unit +
                             " has more than one candidate named '" +
                             item.name + "'; give it as " + unit + "_<ordinal>",
                         /*gen_crash_diag=*/false);
    found = i;
  }
  if (found)
    return *found;
  for (const auto &[name, why] : cands.droppedCandidates())
    if (name == item.name)
      report_fatal_error("[poseidon] " + Twine(site) + ": " + unit +
                             " has no candidate named '" + item.name +
                             "': pricing dropped it, because " + why,
                         /*gen_crash_diag=*/false);
  std::string names;
  for (size_t i = 0; i < cands.numCandidates(); ++i)
    names += "\n  " + cands.candidateName(i);
  if (names.empty())
    names = " none";
  report_fatal_error("[poseidon] " + Twine(site) + ": " + unit +
                         " has no candidate named '" + item.name +
                         "'. Its candidates are:" + names,
                     /*gen_crash_diag=*/false);
}

SmallVector<SolutionStep> parseManualRewrites(Session &S, FunctionFPState &st) {
  const std::string key = siteKey(S, *st.F);
  const SmallVector<std::string, 2> aliases = siteAliases(key);
  SmallVector<SolutionStep> steps;
  SmallPtrSet<const SiteUnit *, 8> appliedUnits;
  for (const ApplyItem &item : applyItems(S)) {
    if (!namesSite(item, key, aliases))
      continue;
    const std::string &id = item.text;
    const UnitKindInfo &kind = infoOf(item.kind);
    SiteUnit *unit = nullptr;
    size_t ofKind = 0;
    for (SiteUnit *u : st.units) {
      if (u->unitKind() != item.kind)
        continue;
      if (u->indexInKind() == item.unit)
        unit = u;
      ++ofKind;
    }
    if (!unit)
      report_fatal_error("[poseidon] " + Twine(kind.tag) + " index " +
                         Twine(item.unit) + " out of range (" + Twine(ofKind) +
                         " " + kind.tag + "s)");
    size_t pos;
    if (item.ordinal) {
      std::optional<size_t> found = unit->positionOfOrdinal(*item.ordinal);
      if (!found && kind.pricingDrops)
        report_fatal_error("[poseidon] " + Twine(kind.tag) + " " +
                           Twine(item.unit) + " has no candidate " +
                           Twine(*item.ordinal) + " that pricing kept ('" +
                           Twine(id) + "')");
      if (!found)
        report_fatal_error("[poseidon] candidate index " +
                           Twine(*item.ordinal) + " out of range for " +
                           kind.tag + " " + Twine(item.unit) + " (" +
                           Twine(unit->numCandidates()) + " candidates)");
      pos = *found;
    } else {
      pos =
          findNamed(item, key, (kind.idPrefix + Twine(item.unit)).str(), *unit);
    }
    if (!appliedUnits.insert(unit).second)
      report_fatal_error("[poseidon] " + Twine(kind.tag) + " " +
                         Twine(item.unit) + " already has " + kind.appliedNoun +
                         " applied (duplicate ID '" + Twine(id) + "')");
    llvm::errs() << "[poseidon] Selecting " << id << ": "
                 << unit->selectionText(pos) << "\n";
    steps.emplace_back(unit, pos);
  }
  return steps;
}

void printPin(Session &S, const FunctionFPState &st,
              ArrayRef<SolutionStep> applied) {
  const std::string key = siteKey(S, *st.F);
  std::string value;
  for (const SolutionStep &step : applied) {
    const SiteUnit &unit = unitOf(step);
    std::string item = (unit.info().idPrefix + Twine(unit.indexInKind()) + "=" +
                        unit.candidateName(step.candidateIndex))
                           .str();
    if (!value.empty())
      value += ",";
    value += key + ":" + item;
  }
  if (!value.empty() && !S.Pins.printed.insert(value).second)
    return;
  llvm::errs() << "[poseidon] pin: " << value << "\n";
}

bool applySolution(ArrayRef<SolutionStep> steps,
                   SmallVectorImpl<SolutionStep> &applied) {
  if (steps.empty())
    return false;
  llvm::errs() << "\n!!! Applying solution (" << steps.size()
               << " step(s)) "
                  "... !!!\n";
  // The binary64 and binary32 searches of one output are separate items, and
  // the first rewrite erases the output the second would replace.
  SmallPtrSet<Value *, 8> rewrittenOutputs;
  for (const auto &step : steps) {
    SiteUnit &unit = unitOf(step);
    llvm::errs() << "Applying solution for "
                 << unit.choiceText(step.candidateIndex) << "\n";
    ArrayRef<Value *> outputs = unit.rewrittenOutputs();
    if (llvm::any_of(outputs,
                     [&](Value *V) { return rewrittenOutputs.contains(V); })) {
      llvm::errs() << "[poseidon] output already rewritten by an "
                      "earlier step; skipping this one\n";
      continue;
    }
    rewrittenOutputs.insert(outputs.begin(), outputs.end());
    unit.materialize(step.candidateIndex);
    applied.push_back(step);
  }
  llvm::errs() << "!!! Solution applied !!!\n\n";
  return true;
}

} // namespace poseidon
