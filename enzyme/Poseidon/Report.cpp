//=- Report.cpp - the DP table print and the report files -------------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Report.h"
#include "Pass/Session.h"

#include "FPProfileName.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace poseidon {

void printDPTable(const CostScale &scale, const CostMap &costToAccuracyMap,
                  const SolutionMap &costToSolutionMap) {
  llvm::errs() << "\n*** DP Table ***\n";
  for (const auto &pair : costToAccuracyMap) {

    llvm::errs() << "Computation cost: " << scale.share(pair.first)
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

void emitReport(Session &S, StringRef funcName, const CostScale &scale,
                const CostMap &costToAccuracyMap,
                const SolutionMap &costToSolutionMap,
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
  // Named as the site's profile is, which keeps a mangled name within a file
  // name's length.
  const std::string stem = profileNameStem(funcName.str());

  json::Object report;
  report["function"] = funcName.str();
  report["num_pareto_points"] = static_cast<int64_t>(costToAccuracyMap.size());

  if (!costToAccuracyMap.empty()) {
    report["cost_range_min"] = scale.share(costToAccuracyMap.begin()->first);
    report["cost_range_max"] = scale.share(costToAccuracyMap.rbegin()->first);
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
    point["computation_cost"] = scale.share(pair.first);
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
      (Twine(O.Diagnostics.ReportPath) + "/" + stem + ".json").str();
  raw_fd_ostream jsonOut(jsonFile, EC, sys::fs::OF_Text);
  if (EC) {
    llvm::errs() << "Error writing JSON report: " << EC.message() << "\n";
  } else {
    jsonOut << formatv("{0:2}", json::Value(std::move(report))) << "\n";
    llvm::errs() << "Poseidon JSON report written to " << jsonFile << "\n";
  }

  std::string textFile =
      (Twine(O.Diagnostics.ReportPath) + "/" + stem + ".txt").str();
  raw_fd_ostream textOut(textFile, EC, sys::fs::OF_Text);
  if (EC) {
    llvm::errs() << "Error writing text report: " << EC.message() << "\n";
    return;
  }

  textOut << "=== Poseidon Report: " << funcName << " ===\n";
  textOut << "Pareto table: " << costToAccuracyMap.size() << " points";
  if (!costToAccuracyMap.empty()) {
    textOut << ", cost range [" << scale.share(costToAccuracyMap.begin()->first)
            << ", " << scale.share(costToAccuracyMap.rbegin()->first) << "]";
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
            << ": Cost=" << scale.share(pair.first)
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
        budgetArr.push_back(scale.share(pair.first));
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
        (Twine(O.Diagnostics.ReportPath) + "/" + stem + "_rewrites.json").str();
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

} // namespace poseidon
