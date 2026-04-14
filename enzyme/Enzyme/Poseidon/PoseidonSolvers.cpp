//=- PoseidonSolvers.cpp - Solver utilities for Poseidon ------------------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements solver-related utilities for the Poseidon optimization
// pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "Poseidon.h"
#include "PoseidonSolvers.h"
#include "PoseidonTypes.h"

#include "PoseidonHerbieUtils.h"
#include "PoseidonPrecUtils.h"
#include "PoseidonUtils.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <random>

using namespace llvm;

extern "C" {
cl::opt<std::string> FPOptSolverType("fpopt-solver-type", cl::init("dp"),
                                     cl::Hidden,
                                     cl::desc("Which solver to use; "
                                              "either 'dp' or 'greedy'"));
cl::opt<int64_t> FPOptComputationCostBudget(
    "fpopt-comp-cost-budget", cl::init(0L), cl::Hidden,
    cl::desc("The maximum computation cost budget for the solver"));
cl::opt<bool> FPOptShowTable(
    "fpopt-show-table", cl::init(false), cl::Hidden,
    cl::desc(
        "Print the full DP table (highly verbose for large applications)"));
cl::list<int64_t> FPOptShowTableCosts(
    "fpopt-show-table-costs", cl::ZeroOrMore, cl::CommaSeparated, cl::Hidden,
    cl::desc(
        "Comma-separated list of computation costs for which to print DP table "
        "entries. If provided, only specified computation costs are "
        "printed. "));
cl::opt<bool> FPOptEarlyPrune(
    "fpopt-early-prune", cl::init(true), cl::Hidden,
    cl::desc("Prune dominated candidates in expression transformation phases"));
cl::opt<double> FPOptCostDominanceThreshold(
    "fpopt-cost-dom-thres", cl::init(0.0), cl::Hidden,
    cl::desc("The threshold for cost dominance in DP solver"));
cl::opt<double> FPOptAccuracyDominanceThreshold(
    "fpopt-acc-dom-thres", cl::init(0.0), cl::Hidden,
    cl::desc("The threshold for accuracy dominance in DP solver"));
cl::opt<bool> FPOptRefineDPTable(
    "fpopt-refine-dp", cl::init(false), cl::Hidden,
    cl::desc("After initial DP build, materialize each solution in a cloned\n"
             "function, run O3, and recompute cost deltas. Only applies when\n"
             "generating a new cache, not when loading from cache."));
cl::opt<std::string> FPOptReportPath(
    "fpopt-report-path", cl::init(""), cl::Hidden,
    cl::desc("Directory to write Poseidon optimization reports.\n"
             "Emits <func>.json (Pareto table with source locations),\n"
             "<func>.txt (human-readable), <func>_rewrites.json\n"
             "(curated per-rewrite analysis with IDs), and\n"
             "validate_config.json + validate.py (validation script)."));
cl::opt<std::string> FPOptApplyRewrites(
    "fpopt-apply-rewrites", cl::init(""), cl::Hidden,
    cl::desc("Comma-separated rewrite IDs to apply (e.g.\n"
             "R0_1,R3_0,PT1_2). IDs are from the _rewrites.json\n"
             "report. Bypasses the DP solver. At most one candidate\n"
             "per expression (R) or subgraph (PT)."));
}

#if LLVM_VERSION_MAJOR >= 21
#define GET_INSTRUCTION_COST(cost) (cost.getValue())
#else
#define GET_INSTRUCTION_COST(cost) (cost.getValue().value())
#endif

static json::Value jsonFloat(double v) {
  if (std::isfinite(v))
    return json::Value(v);
  return json::Value(nullptr);
}

static json::Object getSourceLocationJSON(Value *V) {
  json::Object loc;
  if (auto *I = dyn_cast<Instruction>(V)) {
    if (const auto &DL = I->getDebugLoc()) {
      loc["file"] = DL->getFilename().str();
      loc["line"] = static_cast<int64_t>(DL.getLine());
      loc["col"] = static_cast<int64_t>(DL.getCol());
    }
  }
  return loc;
}

static std::string getSourceLocationStr(Value *V) {
  if (auto *I = dyn_cast<Instruction>(V)) {
    if (const auto &DL = I->getDebugLoc()) {
      return (DL->getFilename() + ":" + Twine(DL.getLine()) + ":" +
              Twine(DL.getCol()))
          .str();
    }
  }
  return "";
}

static std::string getValueStr(Value *V) {
  std::string str;
  raw_string_ostream OS(str);
  V->print(OS);
  return str;
}

// Collect unique source locations from a set of LLVM instructions.
static json::Array getSourceLocationsJSON(ArrayRef<Instruction *> insts) {
  json::Array locs;
  SmallSet<std::string, 4> seen;
  for (auto *I : insts) {
    auto locStr = getSourceLocationStr(I);
    if (!locStr.empty() && seen.insert(locStr).second) {
      locs.push_back(getSourceLocationJSON(I));
    }
  }
  return locs;
}

static json::Object
buildStepJSON(const SolutionStep &step,
              const std::map<InstructionCost, double> &costToAccuracyMap) {
  json::Object stepObj;
  std::visit(
      [&](auto *item) {
        using T = std::decay_t<decltype(*item)>;
        if constexpr (std::is_same_v<T, CandidateOutput>) {
          stepObj["type"] = "rewrite";
          stepObj["original_expr"] = item->expr;
          auto &cand = item->candidates[step.candidateIndex];
          stepObj["rewritten_expr"] = cand.expr;
          stepObj["herbie_cost"] = jsonFloat(cand.herbieCost);
          stepObj["herbie_accuracy"] = jsonFloat(cand.herbieAccuracy);
          stepObj["initial_herbie_cost"] = jsonFloat(item->initialHerbieCost);
          stepObj["initial_herbie_accuracy"] =
              jsonFloat(item->initialHerbieAccuracy);
          stepObj["computation_cost_delta"] =
              GET_INSTRUCTION_COST(item->getCompCostDelta(step.candidateIndex));
          stepObj["accuracy_cost_delta"] =
              jsonFloat(item->getAccCostDelta(step.candidateIndex));
          stepObj["gradient"] = jsonFloat(item->grad);
          stepObj["executions"] = static_cast<int64_t>(item->executions);

          // Source locations from the output instruction and erasable insts
          SmallVector<Instruction *, 8> insts;
          if (auto *I = dyn_cast<Instruction>(item->oldOutput))
            insts.push_back(I);
          for (auto *I : item->erasableInsts)
            insts.push_back(I);
          stepObj["source_locations"] = getSourceLocationsJSON(insts);

          // Affected LLVM IR instructions
          json::Array affectedIR;
          if (auto *I = dyn_cast<Instruction>(item->oldOutput))
            affectedIR.push_back(getValueStr(I));
          for (auto *I : item->erasableInsts) {
            if (I != item->oldOutput)
              affectedIR.push_back(getValueStr(I));
          }
          stepObj["affected_instructions"] = std::move(affectedIR);
        } else if constexpr (std::is_same_v<T, CandidateSubgraph>) {
          auto &cand = item->candidates[step.candidateIndex];
          stepObj["type"] = "precision_change";
          stepObj["description"] = cand.desc;
          stepObj["candidate_index"] =
              static_cast<int64_t>(step.candidateIndex);

          json::Array changes;
          for (const auto &change : cand.changes) {
            json::Object changeObj;
            changeObj["from"] = getPrecisionChangeTypeString(change.oldType);
            changeObj["to"] = getPrecisionChangeTypeString(change.newType);
            changeObj["num_operations"] =
                static_cast<int64_t>(change.nodes.size());

            SmallVector<Instruction *, 8> insts;
            json::Array nodeIR;
            for (auto *node : change.nodes) {
              if (auto *I = dyn_cast<Instruction>(node->value)) {
                insts.push_back(I);
                nodeIR.push_back(getValueStr(I));
              }
            }
            changeObj["source_locations"] = getSourceLocationsJSON(insts);
            changeObj["affected_instructions"] = std::move(nodeIR);
            changes.push_back(std::move(changeObj));
          }
          stepObj["changes"] = std::move(changes);
        }
      },
      step.item);
  return stepObj;
}

static void writeTextReportStep(raw_ostream &OS, const SolutionStep &step,
                                unsigned indent) {
  std::string pad(indent, ' ');
  std::visit(
      [&](auto *item) {
        using T = std::decay_t<decltype(*item)>;
        if constexpr (std::is_same_v<T, CandidateOutput>) {
          auto &cand = item->candidates[step.candidateIndex];
          OS << pad << "[Rewrite] " << item->expr << "  -->  " << cand.expr
             << "\n";
          if (!std::isnan(item->initialHerbieAccuracy) &&
              !std::isnan(cand.herbieAccuracy))
            OS << pad << "  Herbie accuracy: " << item->initialHerbieAccuracy
               << " -> " << cand.herbieAccuracy << " bits\n";
          OS << pad << "  Gradient: " << item->grad
             << ", Executions: " << item->executions << "\n";

          // Source locations
          SmallSet<std::string, 4> seen;
          auto printLoc = [&](Instruction *I) {
            auto loc = getSourceLocationStr(I);
            if (!loc.empty() && seen.insert(loc).second)
              OS << pad << "  Source: " << loc << "\n";
          };
          if (auto *I = dyn_cast<Instruction>(item->oldOutput))
            printLoc(I);
          for (auto *I : item->erasableInsts)
            printLoc(I);

          // Affected IR
          OS << pad << "  Affected IR:\n";
          if (auto *I = dyn_cast<Instruction>(item->oldOutput))
            OS << pad << "    " << *I << "\n";
          for (auto *I : item->erasableInsts) {
            if (I != item->oldOutput)
              OS << pad << "    " << *I << "\n";
          }
        } else if constexpr (std::is_same_v<T, CandidateSubgraph>) {
          auto &cand = item->candidates[step.candidateIndex];
          OS << pad << "[Precision] " << cand.desc << " (#"
             << step.candidateIndex << ")\n";
          for (const auto &change : cand.changes) {
            OS << pad << "  " << getPrecisionChangeTypeString(change.oldType)
               << " -> " << getPrecisionChangeTypeString(change.newType)
               << " for " << change.nodes.size() << " operations\n";
            SmallSet<std::string, 4> seen;
            for (auto *node : change.nodes) {
              if (auto *I = dyn_cast<Instruction>(node->value)) {
                auto loc = getSourceLocationStr(I);
                if (!loc.empty() && seen.insert(loc).second)
                  OS << pad << "    Source: " << loc << "\n";
              }
            }
          }
        }
      },
      step.item);
}

static void
emitPoseidonReport(StringRef funcName,
                   const std::map<InstructionCost, double> &costToAccuracyMap,
                   const std::map<InstructionCost, SmallVector<SolutionStep>>
                       &costToSolutionMap,
                   SmallVector<CandidateOutput, 4> &COs,
                   SmallVector<CandidateSubgraph, 4> &CSs) {
  if (FPOptReportPath.empty())
    return;

  std::error_code EC;
  if (!llvm::sys::fs::exists(FPOptReportPath)) {
    EC = llvm::sys::fs::create_directories(FPOptReportPath);
    if (EC) {
      llvm::errs() << "Error creating report directory: " << EC.message()
                   << "\n";
      return;
    }
  }

  // Build JSON report
  json::Object report;
  report["function"] = funcName.str();
  report["num_pareto_points"] = static_cast<int64_t>(costToAccuracyMap.size());

  if (!costToAccuracyMap.empty()) {
    report["cost_range_min"] =
        GET_INSTRUCTION_COST(costToAccuracyMap.begin()->first);
    report["cost_range_max"] =
        GET_INSTRUCTION_COST(costToAccuracyMap.rbegin()->first);
  }

  // Candidate summary
  json::Array coSummary;
  for (const auto &CO : COs) {
    json::Object co;
    co["original_expr"] = CO.expr;
    co["num_candidates"] = static_cast<int64_t>(CO.candidates.size());
    co["gradient"] = jsonFloat(CO.grad);
    co["executions"] = static_cast<int64_t>(CO.executions);
    co["initial_accuracy_cost"] = jsonFloat(CO.initialAccCost);
    co["initial_computation_cost"] = GET_INSTRUCTION_COST(CO.initialCompCost);
    if (auto *I = dyn_cast<Instruction>(CO.oldOutput)) {
      auto loc = getSourceLocationJSON(I);
      if (!loc.empty())
        co["source_location"] = std::move(loc);
    }
    coSummary.push_back(std::move(co));
  }
  report["candidate_outputs"] = std::move(coSummary);

  json::Array csSummary;
  for (const auto &CS : CSs) {
    json::Object cs;
    cs["num_candidates"] = static_cast<int64_t>(CS.candidates.size());
    cs["initial_accuracy_cost"] = jsonFloat(CS.initialAccCost);
    cs["initial_computation_cost"] = GET_INSTRUCTION_COST(CS.initialCompCost);
    csSummary.push_back(std::move(cs));
  }
  report["candidate_subgraphs"] = std::move(csSummary);

  // Pareto table
  json::Array paretoPoints;
  for (const auto &pair : costToAccuracyMap) {
    json::Object point;
    point["computation_cost"] = GET_INSTRUCTION_COST(pair.first);
    point["accuracy_cost"] = pair.second;

    auto it = costToSolutionMap.find(pair.first);
    if (it != costToSolutionMap.end()) {
      json::Array steps;
      for (const auto &step : it->second) {
        steps.push_back(buildStepJSON(step, costToAccuracyMap));
      }
      point["steps"] = std::move(steps);
    }
    paretoPoints.push_back(std::move(point));
  }
  report["pareto_points"] = std::move(paretoPoints);

  // Write JSON
  std::string jsonFile =
      (Twine(FPOptReportPath) + "/" + funcName + ".json").str();
  raw_fd_ostream jsonOut(jsonFile, EC, sys::fs::OF_Text);
  if (EC) {
    llvm::errs() << "Error writing JSON report: " << EC.message() << "\n";
  } else {
    jsonOut << formatv("{0:2}", json::Value(std::move(report))) << "\n";
    llvm::errs() << "Poseidon JSON report written to " << jsonFile << "\n";
  }

  // Write human-readable text report
  std::string textFile =
      (Twine(FPOptReportPath) + "/" + funcName + ".txt").str();
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
  textOut << "Candidate outputs: " << COs.size()
          << ", Candidate subgraphs: " << CSs.size() << "\n\n";

  // Per-CO summary
  for (size_t i = 0; i < COs.size(); ++i) {
    textOut << "Expression #" << i << ": " << COs[i].expr << "\n";
    textOut << "  Gradient: " << COs[i].grad
            << ", Executions: " << COs[i].executions << "\n";
    if (auto *I = dyn_cast<Instruction>(COs[i].oldOutput)) {
      auto loc = getSourceLocationStr(I);
      if (!loc.empty())
        textOut << "  Source: " << loc << "\n";
    }
    textOut << "  Candidates: " << COs[i].candidates.size() << "\n";
    for (size_t j = 0; j < COs[i].candidates.size(); ++j) {
      auto &cand = COs[i].candidates[j];
      textOut << "    [" << j << "] " << cand.expr;
      if (!std::isnan(cand.herbieAccuracy))
        textOut << "  (accuracy: " << cand.herbieAccuracy << " bits)";
      textOut << "\n";
    }
  }
  textOut << "\n";

  // Pareto points
  unsigned pointIdx = 0;
  for (const auto &pair : costToAccuracyMap) {
    textOut << "--- Pareto Point #" << pointIdx++
            << ": Cost=" << GET_INSTRUCTION_COST(pair.first)
            << ", Accuracy=" << pair.second << " ---\n";
    auto it = costToSolutionMap.find(pair.first);
    if (it != costToSolutionMap.end() && !it->second.empty()) {
      for (const auto &step : it->second) {
        writeTextReportStep(textOut, step, 2);
      }
    } else {
      textOut << "  (no changes)\n";
    }
    textOut << "\n";
  }

  llvm::errs() << "Poseidon text report written to " << textFile << "\n";

  // Emit validation config + script
  {
    std::string configFile =
        (Twine(FPOptReportPath) + "/validate_config.json").str();
    {
      json::Object cfg;
      cfg["function"] = funcName.str();
      cfg["profile_path"] = FPProfileUse.getValue();
      cfg["cache_path"] = FPOptCachePath.getValue();
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

  // --- Curated per-rewrite analysis ---
  // Enumerate every individual rewrite candidate, categorize by its marginal
  // impact, and rank tradeoffs by efficiency (benefit per unit cost).
  {
    json::Array rewrites;
    for (size_t coIdx = 0; coIdx < COs.size(); ++coIdx) {
      auto &CO = COs[coIdx];
      for (size_t ci = 0; ci < CO.candidates.size(); ++ci) {
        auto &cand = CO.candidates[ci];
        double accDelta = CO.getAccCostDelta(ci);
        auto compDelta = CO.getCompCostDelta(ci);
        int64_t compDeltaVal = GET_INSTRUCTION_COST(compDelta);

        // Skip candidates with no usable data
        if (std::isnan(accDelta))
          continue;

        // Categorize
        // comp < 0 means faster, acc < 0 means more accurate
        std::string category;
        double efficiency = 0.0;
        if (compDeltaVal <= 0 && accDelta <= 0) {
          category = "free_win";
          // Rank by combined magnitude of improvement
          efficiency = std::abs(accDelta) + std::abs((double)compDeltaVal);
        } else if (compDeltaVal <= 0 && accDelta > 0) {
          category = "speed_for_accuracy";
          // Efficiency: speed gained per unit of accuracy lost
          efficiency = (accDelta > 1e-30)
                           ? std::abs((double)compDeltaVal) / accDelta
                           : std::abs((double)compDeltaVal);
        } else if (compDeltaVal > 0 && accDelta <= 0) {
          category = "accuracy_for_speed";
          // Efficiency: accuracy gained per unit of speed lost
          efficiency = (compDeltaVal > 0)
                           ? std::abs(accDelta) / (double)compDeltaVal
                           : std::abs(accDelta);
        } else {
          continue; // Both worse — skip
        }

        json::Object rw;
        std::string id = "R" + std::to_string(coIdx) + "_" + std::to_string(ci);
        rw["id"] = id;
        rw["original_expr"] = CO.expr;
        rw["rewritten_expr"] = cand.expr;
        rw["category"] = category;
        rw["efficiency"] = jsonFloat(efficiency);
        rw["computation_cost_delta"] = compDeltaVal;
        rw["accuracy_cost_delta"] = jsonFloat(accDelta);
        rw["gradient"] = jsonFloat(CO.grad);
        rw["executions"] = static_cast<int64_t>(CO.executions);
        rw["herbie_accuracy"] = jsonFloat(cand.herbieAccuracy);
        rw["initial_herbie_accuracy"] = jsonFloat(CO.initialHerbieAccuracy);

        // Source location
        if (auto *I = dyn_cast<Instruction>(CO.oldOutput)) {
          auto loc = getSourceLocationJSON(I);
          if (!loc.empty())
            rw["source_location"] = std::move(loc);
        }

        rewrites.push_back(std::move(rw));
      }
    }

    // Also include precision tuning candidates
    for (size_t csIdx = 0; csIdx < CSs.size(); ++csIdx) {
      auto &CS = CSs[csIdx];
      for (size_t ci = 0; ci < CS.candidates.size(); ++ci) {
        auto &pt = CS.candidates[ci];
        double accDelta = CS.getAccCostDelta(ci);
        auto compDelta = CS.getCompCostDelta(ci);
        int64_t compDeltaVal = GET_INSTRUCTION_COST(compDelta);

        if (std::isnan(accDelta))
          continue;

        std::string category;
        double efficiency = 0.0;
        if (compDeltaVal <= 0 && accDelta <= 0) {
          category = "free_win";
          efficiency = std::abs(accDelta) + std::abs((double)compDeltaVal);
        } else if (compDeltaVal <= 0 && accDelta > 0) {
          category = "speed_for_accuracy";
          efficiency = (accDelta > 1e-30)
                           ? std::abs((double)compDeltaVal) / accDelta
                           : std::abs((double)compDeltaVal);
        } else if (compDeltaVal > 0 && accDelta <= 0) {
          category = "accuracy_for_speed";
          efficiency = (compDeltaVal > 0)
                           ? std::abs(accDelta) / (double)compDeltaVal
                           : std::abs(accDelta);
        } else {
          continue;
        }

        json::Object rw;
        std::string id =
            "PT" + std::to_string(csIdx) + "_" + std::to_string(ci);
        rw["id"] = id;
        rw["type"] = "precision_change";
        rw["description"] = pt.desc;
        rw["category"] = category;
        rw["efficiency"] = jsonFloat(efficiency);
        rw["computation_cost_delta"] = compDeltaVal;
        rw["accuracy_cost_delta"] = jsonFloat(accDelta);

        // Collect affected instructions + deduplicated source locations
        SmallVector<Instruction *, 16> ptInsts;
        for (const auto &change : pt.changes) {
          for (auto *node : change.nodes) {
            if (auto *I = dyn_cast<Instruction>(node->value))
              ptInsts.push_back(I);
          }
        }
        rw["source_locations"] = getSourceLocationsJSON(ptInsts);
        json::Array affectedIR;
        for (auto *I : ptInsts)
          affectedIR.push_back(getValueStr(I));
        rw["affected_instructions"] = std::move(affectedIR);

        rewrites.push_back(std::move(rw));
      }
    }

    // Sort: free_wins first, then by efficiency descending
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
      return ae > be; // higher efficiency first
    });

    json::Array sortedRewrites;
    for (auto &v : rewriteVec)
      sortedRewrites.push_back(std::move(v));

    std::string rewritesFile =
        (Twine(FPOptReportPath) + "/" + funcName + "_rewrites.json").str();
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

// Given the cost budget `FPOptComputationCostBudget`, we want to minimize the
// accuracy cost of the rewritten expressions.
bool accuracyGreedySolver(
    SmallVector<CandidateOutput, 4> &COs,
    SmallVector<CandidateSubgraph, 4> &CSs,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap) {
  bool changed = false;
  llvm::errs() << "Starting accuracy greedy solver with computation budget: "
               << FPOptComputationCostBudget << "\n";
  InstructionCost totalComputationCost = 0;

  SmallVector<size_t, 4> aoIndices;
  for (size_t i = 0; i < COs.size(); ++i) {
    aoIndices.push_back(i);
  }
  std::mt19937 g(FPOptRandomSeed);
  std::shuffle(aoIndices.begin(), aoIndices.end(), g);

  for (size_t idx : aoIndices) {
    auto &CO = COs[idx];
    int bestCandidateIndex = -1;
    double bestAccuracyCost = std::numeric_limits<double>::infinity();
    InstructionCost bestCandidateComputationCost;

    for (const auto &candidate : enumerate(CO.candidates)) {
      size_t i = candidate.index();
      auto candCompCost = CO.getCompCostDelta(i);
      auto candAccCost = CO.getAccCostDelta(i);
      // llvm::errs() << "CO Candidate " << i << " for " << CO.expr
      //              << " has accuracy cost: " << candAccCost
      //              << " and computation cost: " << candCompCost << "\n";

      if (totalComputationCost + candCompCost <= FPOptComputationCostBudget) {
        if (candAccCost < bestAccuracyCost) {
          // llvm::errs() << "CO Candidate " << i << " selected!\n";
          bestCandidateIndex = i;
          bestAccuracyCost = candAccCost;
          bestCandidateComputationCost = candCompCost;
        }
      }
    }

    if (bestCandidateIndex != -1) {
      CO.apply(bestCandidateIndex, valueToNodeMap, symbolToValueMap);
      changed = true;
      totalComputationCost += bestCandidateComputationCost;
      if (FPOptPrint) {
        llvm::errs() << "Greedy solver selected candidate "
                     << bestCandidateIndex << " for " << CO.expr
                     << " with accuracy cost: " << bestAccuracyCost
                     << " and computation cost: "
                     << bestCandidateComputationCost << "\n";
      }
    }
  }

  SmallVector<size_t, 4> accIndices;
  for (size_t i = 0; i < CSs.size(); ++i) {
    accIndices.push_back(i);
  }
  std::shuffle(accIndices.begin(), accIndices.end(), g);

  for (size_t idx : accIndices) {
    auto &CS = CSs[idx];
    int bestCandidateIndex = -1;
    double bestAccuracyCost = std::numeric_limits<double>::infinity();
    InstructionCost bestCandidateComputationCost;

    for (const auto &candidate : enumerate(CS.candidates)) {
      size_t i = candidate.index();
      auto candCompCost = CS.getCompCostDelta(i);
      auto candAccCost = CS.getAccCostDelta(i);
      // llvm::errs() << "CS Candidate " << i << " (" << candidate.value().desc
      //              << ") has accuracy cost: " << candAccCost
      //              << " and computation cost: " << candCompCost << "\n";

      if (totalComputationCost + candCompCost <= FPOptComputationCostBudget) {
        if (candAccCost < bestAccuracyCost) {
          // llvm::errs() << "CS Candidate " << i << " selected!\n";
          bestCandidateIndex = i;
          bestAccuracyCost = candAccCost;
          bestCandidateComputationCost = candCompCost;
        }
      }
    }

    if (bestCandidateIndex != -1) {
      CS.apply(bestCandidateIndex);
      changed = true;
      totalComputationCost += bestCandidateComputationCost;
      if (FPOptPrint) {
        llvm::errs() << "Greedy solver selected candidate "
                     << bestCandidateIndex << " for "
                     << CS.candidates[bestCandidateIndex].desc
                     << " with accuracy cost: " << bestAccuracyCost
                     << " and computation cost: "
                     << bestCandidateComputationCost << "\n";
      }
    }
  }

  llvm::errs() << "Greedy solver finished with total computation cost: "
               << totalComputationCost
               << "; total allowance: " << FPOptComputationCostBudget << "\n";

  return changed;
}

bool accuracyDPSolver(
    Function &F, const TargetTransformInfo &TTI,
    SmallVector<CandidateOutput, 4> &COs,
    SmallVector<CandidateSubgraph, 4> &CSs,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap,
    double errorTol) {
  bool changed = false;
  llvm::errs() << "Starting accuracy DP solver with computation budget: "
               << FPOptComputationCostBudget << "\n";
  if (errorTol > 0.0) {
    llvm::errs() << "Absolute error tolerance: " << errorTol << "\n";
  }

  using CostMap = std::map<InstructionCost, double>;
  using SolutionMap = std::map<InstructionCost, SmallVector<SolutionStep>>;

  CostMap costToAccuracyMap;
  SolutionMap costToSolutionMap;
  CostMap newCostToAccuracyMap;
  SolutionMap newCostToSolutionMap;
  CostMap prunedCostToAccuracyMap;
  SolutionMap prunedCostToSolutionMap;

  std::string cacheFilePath = FPOptCachePath + "/table.json";
  bool loadedFromCache = false;

  if (llvm::sys::fs::exists(cacheFilePath)) {
    llvm::errs() << "Cache file found. Loading DP tables from cache.\n";
    loadedFromCache = true;

    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
        llvm::MemoryBuffer::getFile(cacheFilePath);
    if (std::error_code ec = fileOrErr.getError()) {
      llvm::errs() << "Error reading cache file: " << ec.message() << "\n";
      return changed;
    }
    llvm::StringRef buffer = fileOrErr.get()->getBuffer();
    llvm::Expected<llvm::json::Value> jsonOrErr = llvm::json::parse(buffer);
    if (!jsonOrErr) {
      llvm::errs() << "Error parsing JSON from cache file: "
                   << llvm::toString(jsonOrErr.takeError()) << "\n";
      return changed;
    }

    llvm::json::Object *jsonObj = jsonOrErr->getAsObject();
    if (!jsonObj) {
      llvm::errs() << "Invalid JSON format in cache file.\n";
      return changed;
    }

    if (llvm::json::Object *costAccMap =
            jsonObj->getObject("costToAccuracyMap")) {
      for (auto &pair : *costAccMap) {
        InstructionCost compCost(std::stoll(pair.first.str()));
        double accCost = pair.second.getAsNumber().value();
        costToAccuracyMap[compCost] = accCost;
      }
    } else {
      llvm_unreachable("Invalid costToAccuracyMap in cache file.");
    }

    if (llvm::json::Object *costSolMap =
            jsonObj->getObject("costToSolutionMap")) {
      for (auto &pair : *costSolMap) {
        InstructionCost compCost(std::stoll(pair.first.str()));
        SmallVector<SolutionStep> solutionSteps;

        llvm::json::Array *stepsArray = pair.second.getAsArray();
        if (!stepsArray) {
          llvm::errs() << "Invalid steps array in cache file.\n";
          return changed;
        }

        for (llvm::json::Value &stepVal : *stepsArray) {
          llvm::json::Object *stepObj = stepVal.getAsObject();
          if (!stepObj) {
            llvm_unreachable("Invalid step object in cache file.");
          }

          StringRef itemType = stepObj->getString("itemType").value();
          size_t candidateIndex = stepObj->getInteger("candidateIndex").value();
          size_t itemIndex = stepObj->getInteger("itemIndex").value();

          if (itemType == "CO") {
            if (itemIndex >= COs.size()) {
              llvm_unreachable("Invalid CandidateOutput index in cache file.");
            }
            solutionSteps.emplace_back(&COs[itemIndex], candidateIndex);
          } else if (itemType == "CS") {
            if (itemIndex >= CSs.size()) {
              llvm_unreachable(
                  "Invalid CandidateSubgraph index in cache file.");
            }
            solutionSteps.emplace_back(&CSs[itemIndex], candidateIndex);
          } else {
            llvm_unreachable("Invalid itemType in cache file.");
          }
        }

        costToSolutionMap[compCost] = solutionSteps;
      }
    } else {
      llvm::errs() << "costToSolutionMap not found in cache file.\n";
      return changed;
    }

    llvm::errs() << "Loaded DP tables from cache.\n";

  } else {
    llvm::errs() << "Cache file not found. Proceeding to solve DP.\n";

    costToAccuracyMap[0] = 0;
    costToSolutionMap[0] = {};

    std::unordered_map<CandidateOutput *, size_t> aoPtrToIndex;
    for (size_t i = 0; i < COs.size(); ++i) {
      aoPtrToIndex[&COs[i]] = i;
    }
    std::unordered_map<CandidateSubgraph *, size_t> accPtrToIndex;
    for (size_t i = 0; i < CSs.size(); ++i) {
      accPtrToIndex[&CSs[i]] = i;
    }

    int COCounter = 0;

    for (auto &CO : COs) {
      // It is possible to apply zero candidate for an CO.
      // When no candidate is applied, the resulting accuracy cost
      // and solution steps remain the same.
      newCostToAccuracyMap = costToAccuracyMap;
      newCostToSolutionMap = costToSolutionMap;

      for (const auto &pair : costToAccuracyMap) {
        InstructionCost currCompCost = pair.first;
        double currAccCost = pair.second;

        for (const auto &candidate : enumerate(CO.candidates)) {
          size_t i = candidate.index();
          auto candCompCost = CO.getCompCostDelta(i);
          auto candAccCost = CO.getAccCostDelta(i);

          // Don't apply a candidate that strictly makes things worse
          if (candCompCost >= 0 && candAccCost >= 0.) {
            continue;
          }

          InstructionCost newCompCost = currCompCost + candCompCost;
          double newAccCost = currAccCost + candAccCost;

          // if (FPOptPrint)
          //   llvm::errs() << "CO candidate " << i
          //                << " has accuracy cost: " << candAccCost
          //                << " and computation cost: " << candCompCost <<
          //                "\n";

          if (newCostToAccuracyMap.find(newCompCost) ==
                  newCostToAccuracyMap.end() ||
              newCostToAccuracyMap[newCompCost] > newAccCost) {
            newCostToAccuracyMap[newCompCost] = newAccCost;
            newCostToSolutionMap[newCompCost] = costToSolutionMap[currCompCost];
            newCostToSolutionMap[newCompCost].emplace_back(&CO, i);
            // if (FPOptPrint)
            //   llvm::errs() << "Updating accuracy map (CO candidate " << i
            //                << "): computation cost " << newCompCost
            //                << " -> accuracy cost " << newAccCost << "\n";
          }
        }
      }

      // TODO: Do not prune CO parts of the DP table since COs influence CSs
      if (!FPOptEarlyPrune) {
        costToAccuracyMap = newCostToAccuracyMap;
        costToSolutionMap = newCostToSolutionMap;

        llvm::errs() << "##### Finished processing " << ++COCounter << " of "
                     << COs.size() << " COs #####\n";
        llvm::errs() << "Current DP table sizes: " << costToAccuracyMap.size()
                     << "\n";
        continue;
      }

      for (const auto &l : newCostToAccuracyMap) {
        InstructionCost currCompCost = l.first;
        double currAccCost = l.second;

        bool dominated = false;
        for (const auto &r : newCostToAccuracyMap) {
          InstructionCost otherCompCost = r.first;
          double otherAccCost = r.second;

          if (currCompCost - otherCompCost >
                  std::fabs(FPOptCostDominanceThreshold *
                            GET_INSTRUCTION_COST(otherCompCost)) &&
              currAccCost - otherAccCost >=
                  std::fabs(FPOptAccuracyDominanceThreshold * otherAccCost)) {
            // if (FPOptPrint)
            //   llvm::errs() << "CO candidate with computation cost: "
            //                << currCompCost
            //                << " and accuracy cost: " << currAccCost
            //                << " is dominated by candidate with computation
            //                cost:"
            //                << otherCompCost
            //                << " and accuracy cost: " << otherAccCost << "\n";
            dominated = true;
            break;
          }
        }

        if (!dominated) {
          prunedCostToAccuracyMap[currCompCost] = currAccCost;
          prunedCostToSolutionMap[currCompCost] =
              newCostToSolutionMap[currCompCost];
        }
      }

      costToAccuracyMap = prunedCostToAccuracyMap;
      costToSolutionMap = prunedCostToSolutionMap;
      prunedCostToAccuracyMap.clear();
      prunedCostToSolutionMap.clear();

      llvm::errs() << "##### Finished processing " << ++COCounter << " of "
                   << COs.size() << " COs #####\n";
      llvm::errs() << "Current DP table sizes: " << costToAccuracyMap.size()
                   << "\n";
    }

    int CSCounter = 0;

    for (auto &CS : CSs) {
      // It is possible to apply zero candidate for an CS.
      // When no candidate is applied, the resulting accuracy cost
      // and solution steps remain the same.
      newCostToAccuracyMap = costToAccuracyMap;
      newCostToSolutionMap = costToSolutionMap;

      for (const auto &pair : costToAccuracyMap) {
        InstructionCost currCompCost = pair.first;
        double currAccCost = pair.second;

        for (const auto &candidate : enumerate(CS.candidates)) {
          size_t i = candidate.index();
          auto candCompCost =
              CS.getAdjustedCompCostDelta(i, costToSolutionMap[currCompCost]);
          auto candAccCost =
              CS.getAdjustedAccCostDelta(i, costToSolutionMap[currCompCost],
                                         valueToNodeMap, symbolToValueMap);

          // Don't ever try to apply a strictly useless candidate
          if (candCompCost >= 0 && candAccCost >= 0.) {
            continue;
          }

          InstructionCost newCompCost = currCompCost + candCompCost;
          double newAccCost = currAccCost + candAccCost;

          // if (FPOptPrint)
          //   llvm::errs() << "CS candidate " << i << " ("
          //                << candidate.value().desc
          //                << ") has accuracy cost: " << candAccCost
          //                << " and computation cost: " << candCompCost <<
          //                "\n";

          if (newCostToAccuracyMap.find(newCompCost) ==
                  newCostToAccuracyMap.end() ||
              newCostToAccuracyMap[newCompCost] > newAccCost) {
            newCostToAccuracyMap[newCompCost] = newAccCost;
            newCostToSolutionMap[newCompCost] = costToSolutionMap[currCompCost];
            newCostToSolutionMap[newCompCost].emplace_back(&CS, i);
            // if (FPOptPrint) {
            // llvm::errs() << "CS candidate " << i << " ("
            //              << candidate.value().desc
            //              << ") added; has accuracy cost: " << candAccCost
            //              << " and computation cost: " << candCompCost <<
            //              "\n";
            // llvm::errs() << "Updating accuracy map (CS candidate " << i
            //              << "): computation cost " << newCompCost
            //              << " -> accuracy cost " << newAccCost << "\n";
            // }
          }
        }
      }

      for (const auto &l : newCostToAccuracyMap) {
        InstructionCost currCompCost = l.first;
        double currAccCost = l.second;

        bool dominated = false;
        for (const auto &r : newCostToAccuracyMap) {
          InstructionCost otherCompCost = r.first;
          double otherAccCost = r.second;

          if (currCompCost - otherCompCost >
                  std::fabs(FPOptCostDominanceThreshold *
                            GET_INSTRUCTION_COST(otherCompCost)) &&
              currAccCost - otherAccCost >=
                  std::fabs(FPOptAccuracyDominanceThreshold * otherAccCost)) {
            // if (FPOptPrint)
            //   llvm::errs() << "CS candidate with computation cost: "
            //                << currCompCost
            //                << " and accuracy cost: " << currAccCost
            //                << " is dominated by candidate with computation
            //                cost:"
            //                << otherCompCost
            //                << " and accuracy cost: " << otherAccCost << "\n";
            dominated = true;
            break;
          }
        }

        if (!dominated) {
          prunedCostToAccuracyMap[currCompCost] = currAccCost;
          prunedCostToSolutionMap[currCompCost] =
              newCostToSolutionMap[currCompCost];
        }
      }

      costToAccuracyMap = prunedCostToAccuracyMap;
      costToSolutionMap = prunedCostToSolutionMap;
      prunedCostToAccuracyMap.clear();
      prunedCostToSolutionMap.clear();

      llvm::errs() << "##### Finished processing " << ++CSCounter << " of "
                   << CSs.size() << " CSs #####\n";
      llvm::errs() << "Current DP table sizes: " << costToAccuracyMap.size()
                   << "\n";
    }

    json::Object jsonObj;

    json::Object costAccMap;
    for (const auto &pair : costToAccuracyMap) {
      costAccMap[std::to_string(GET_INSTRUCTION_COST(pair.first))] =
          pair.second;
    }
    jsonObj["costToAccuracyMap"] = std::move(costAccMap);

    json::Object costSolMap;
    for (const auto &pair : costToSolutionMap) {
      json::Array stepsArray;
      for (const auto &step : pair.second) {
        json::Object stepObj;
        stepObj["candidateIndex"] = static_cast<int64_t>(step.candidateIndex);

        std::visit(
            [&](auto *item) {
              using T = std::decay_t<decltype(*item)>;
              if constexpr (std::is_same_v<T, CandidateOutput>) {
                stepObj["itemType"] = "CO";
                size_t index = aoPtrToIndex[item];
                stepObj["itemIndex"] = static_cast<int64_t>(index);
              } else if constexpr (std::is_same_v<T, CandidateSubgraph>) {
                stepObj["itemType"] = "CS";
                size_t index = accPtrToIndex[item];
                stepObj["itemIndex"] = static_cast<int64_t>(index);
              }
            },
            step.item);
        stepsArray.push_back(std::move(stepObj));
      }
      costSolMap[std::to_string(GET_INSTRUCTION_COST(pair.first))] =
          std::move(stepsArray);
    }
    jsonObj["costToSolutionMap"] = std::move(costSolMap);

    if (!FPOptRefineDPTable) {
      std::error_code EC;
      llvm::raw_fd_ostream cacheFile(cacheFilePath, EC, llvm::sys::fs::OF_Text);
      if (EC) {
        llvm::errs() << "Error writing cache file: " << EC.message() << "\n";
      } else {
        cacheFile << llvm::formatv("{0:2}",
                                   llvm::json::Value(std::move(jsonObj)))
                  << "\n";
        cacheFile.close();
        llvm::errs() << "DP tables cached to file.\n";
      }
    } else if (!COs.empty() || !CSs.empty()) {
      ValueToValueMapTy BaseVMap;
      Function *BaseClone = CloneFunction(&F, BaseVMap);
      runPoseidonFunctionSimplify(*BaseClone, OptimizationLevel::O3);
      InstructionCost BaseCost = getCompCost(BaseClone, TTI);
      BaseClone->eraseFromParent();

      using CostMap = std::map<InstructionCost, double>;
      using SolutionMap = std::map<InstructionCost, SmallVector<SolutionStep>>;
      CostMap refinedCostToAccuracyMap;
      SolutionMap refinedCostToSolutionMap;

      for (const auto &pair : costToSolutionMap) {
        const SmallVector<SolutionStep> &steps = pair.second;

        ValueToValueMapTy VMap;
        Function *FClone = CloneFunction(&F, VMap);

        for (const auto &step : steps) {
          std::visit(
              [&](auto *item) {
                using T = std::decay_t<decltype(*item)>;
                if constexpr (std::is_same_v<T, CandidateOutput>) {
                  auto &CO = *item;
                  Instruction *oldI = cast<Instruction>(CO.oldOutput);
                  Instruction *clonedOldI = cast<Instruction>(VMap[oldI]);
                  IRBuilder<> builder(clonedOldI->getParent(),
                                      ++BasicBlock::iterator(clonedOldI));
                  builder.setFastMathFlags(clonedOldI->getFastMathFlags());
                  auto parsedNode =
                      parseHerbieExpr(CO.candidates[step.candidateIndex].expr,
                                      valueToNodeMap, symbolToValueMap);
                  Value *newVal = parsedNode->getLLValue(builder, &VMap);
                  clonedOldI->replaceAllUsesWith(newVal);
                } else if constexpr (std::is_same_v<T, CandidateSubgraph>) {
                  auto &CS = *item;
                  PTCandidate &pt = CS.candidates[step.candidateIndex];
                  pt.apply(*CS.subgraph, &VMap);
                } else {
                  llvm_unreachable("accuracyDPSolver refine: unexpected step");
                }
              },
              step.item);
        }

        runPoseidonFunctionSimplify(*FClone, OptimizationLevel::O3);
        InstructionCost NewTotal = getCompCost(FClone, TTI);
        InstructionCost NewDelta = NewTotal - BaseCost;
        FClone->eraseFromParent();

        double accCost = costToAccuracyMap[pair.first];
        auto it = refinedCostToAccuracyMap.find(NewDelta);
        if (it == refinedCostToAccuracyMap.end() || it->second > accCost) {
          refinedCostToAccuracyMap[NewDelta] = accCost;
          refinedCostToSolutionMap[NewDelta] = steps;
        }
      }

      // One-pass domination pruning
      std::map<InstructionCost, double> refinedPrunedAcc;
      std::map<InstructionCost, SmallVector<SolutionStep>> refinedPrunedSol;
      for (const auto &l : refinedCostToAccuracyMap) {
        InstructionCost currCompCost = l.first;
        double currAccCost = l.second;
        bool dominated = false;
        for (const auto &r : refinedCostToAccuracyMap) {
          InstructionCost otherCompCost = r.first;
          double otherAccCost = r.second;
          if (currCompCost - otherCompCost >
                  std::fabs(FPOptCostDominanceThreshold *
                            GET_INSTRUCTION_COST(otherCompCost)) &&
              currAccCost - otherAccCost >=
                  std::fabs(FPOptAccuracyDominanceThreshold * otherAccCost)) {
            dominated = true;
            break;
          }
        }
        if (!dominated) {
          refinedPrunedAcc[currCompCost] = currAccCost;
          refinedPrunedSol[currCompCost] =
              refinedCostToSolutionMap[currCompCost];
        }
      }

      costToAccuracyMap = std::move(refinedPrunedAcc);
      costToSolutionMap = std::move(refinedPrunedSol);

      json::Object jsonObj;

      json::Object costAccMap;
      for (const auto &p : costToAccuracyMap) {
        costAccMap[std::to_string(GET_INSTRUCTION_COST(p.first))] = p.second;
      }
      jsonObj["costToAccuracyMap"] = std::move(costAccMap);

      json::Object costSolMap;
      std::unordered_map<CandidateOutput *, size_t> aoPtrToIndex;
      for (size_t i = 0; i < COs.size(); ++i)
        aoPtrToIndex[&COs[i]] = i;
      std::unordered_map<CandidateSubgraph *, size_t> accPtrToIndex;
      for (size_t i = 0; i < CSs.size(); ++i)
        accPtrToIndex[&CSs[i]] = i;

      for (const auto &p : costToSolutionMap) {
        json::Array stepsArray;
        for (const auto &step : p.second) {
          json::Object stepObj;
          stepObj["candidateIndex"] = static_cast<int64_t>(step.candidateIndex);
          std::visit(
              [&](auto *item) {
                using T = std::decay_t<decltype(*item)>;
                if constexpr (std::is_same_v<T, CandidateOutput>) {
                  stepObj["itemType"] = "CO";
                  size_t index = aoPtrToIndex[item];
                  stepObj["itemIndex"] = static_cast<int64_t>(index);
                } else if constexpr (std::is_same_v<T, CandidateSubgraph>) {
                  stepObj["itemType"] = "CS";
                  size_t index = accPtrToIndex[item];
                  stepObj["itemIndex"] = static_cast<int64_t>(index);
                }
              },
              step.item);
          stepsArray.push_back(std::move(stepObj));
        }
        costSolMap[std::to_string(GET_INSTRUCTION_COST(p.first))] =
            std::move(stepsArray);
      }
      jsonObj["costToSolutionMap"] = std::move(costSolMap);

      std::error_code EC;
      llvm::raw_fd_ostream cacheFile(cacheFilePath, EC, llvm::sys::fs::OF_Text);
      if (EC) {
        llvm::errs() << "Error writing refined cache file: " << EC.message()
                     << "\n";
      } else {
        cacheFile << llvm::formatv("{0:2}",
                                   llvm::json::Value(std::move(jsonObj)))
                  << "\n";
        cacheFile.close();
        llvm::errs() << "Refined DP tables cached to file.\n";
      }
    }
  }

  if (FPOptPrint) {
    if (FPOptShowTable) {
      llvm::errs() << "\n*** DP Table ***\n";
      for (const auto &pair : costToAccuracyMap) {
        if (!FPOptShowTableCosts.empty()) {
          bool shouldPrint = false;
          for (auto selectedCost : FPOptShowTableCosts)
            if (pair.first == selectedCost) {
              shouldPrint = true;
              break;
            }
          if (!shouldPrint)
            continue;
        }

        llvm::errs() << "Computation cost: " << pair.first
                     << ", Accuracy cost: " << pair.second << "\n";
        llvm::errs() << "\tSolution steps: \n";
        for (const auto &step : costToSolutionMap[pair.first]) {
          std::visit(
              [&](auto *item) {
                using T = std::decay_t<decltype(*item)>;
                if constexpr (std::is_same_v<T, CandidateOutput>) {
                  llvm::errs()
                      << "\t\t" << item->expr << " --(" << step.candidateIndex
                      << ")-> " << item->candidates[step.candidateIndex].expr
                      << "\n";
                } else if constexpr (std::is_same_v<T, CandidateSubgraph>) {
                  llvm::errs() << "\t\tCS: "
                               << item->candidates[step.candidateIndex].desc
                               << " (#" << step.candidateIndex << ")\n";
                  if (FPOptShowPTDetails) {
                    auto &candidate = item->candidates[step.candidateIndex];
                    for (const auto &change : candidate.changes) {
                      llvm::errs()
                          << "\t\t\tChanging from "
                          << getPrecisionChangeTypeString(change.oldType)
                          << " to "
                          << getPrecisionChangeTypeString(change.newType)
                          << ":\n";
                      for (auto *val : change.nodes) {
                        llvm::errs() << "\t\t\t\t" << *val->value << "\n";
                      }
                    }
                  }
                } else {
                  llvm_unreachable(
                      "accuracyDPSolver: Unexpected type of solution step");
                }
              },
              step.item);
        }
      }
      llvm::errs() << "*** End of DP Table ***\n\n";
    }
  }

  if (!FPOptCachePath.empty()) {
    std::string budgetsFile = FPOptCachePath + "/budgets.txt";
    if (!llvm::sys::fs::exists(budgetsFile)) {
      std::string budgetsStr;
      for (const auto &pair : costToAccuracyMap) {
        budgetsStr += std::to_string(GET_INSTRUCTION_COST(pair.first)) + ",";
      }

      if (!budgetsStr.empty())
        budgetsStr.pop_back();

      std::error_code EC;
      llvm::raw_fd_ostream Out(budgetsFile, EC, llvm::sys::fs::OF_Text);
      if (EC) {
        llvm::errs() << "Error opening " << budgetsFile << ": " << EC.message()
                     << "\n";
      } else {
        Out << budgetsStr;
      }
    }
  }

  if (!loadedFromCache)
    emitPoseidonReport(F.getName(), costToAccuracyMap, costToSolutionMap, COs,
                       CSs);

  llvm::errs() << "Critical computation cost range: ["
               << costToAccuracyMap.begin()->first << ", "
               << costToAccuracyMap.rbegin()->first << "]\n";

  llvm::errs() << "DP table contains " << costToAccuracyMap.size()
               << " entries.\n";

  double totalCandidateCompositions = 1.0;
  for (const auto &CO : COs) {
    // +1 for the "do nothing" possibility
    totalCandidateCompositions *= CO.candidates.size() + 1;
  }
  for (const auto &CS : CSs) {
    totalCandidateCompositions *= CS.candidates.size() + 1;
  }
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
    InstructionCost minCompCost = std::numeric_limits<InstructionCost>::max();
    bool foundSolution = false;

    for (const auto &pair : costToAccuracyMap) {
      InstructionCost compCost = pair.first;
      double accCost = pair.second;

      if (accCost <= errorTol) {
        if (compCost < minCompCost) {
          minCompCost = compCost;
          minAccCost = accCost;
          bestCompCost = compCost;
          foundSolution = true;
        }
      }
    }

    if (!foundSolution) {
      llvm::errs() << "No solution found that meets accuracy tolerance "
                   << errorTol << "!\n";
      llvm::errs() << "Best achievable accuracy in DP table: "
                   << costToAccuracyMap.begin()->second << "\n";
      return changed;
    }

    llvm::errs() << "Found solution meeting accuracy tolerance " << errorTol
                 << "\n";
    llvm::errs() << "Accuracy cost achieved: " << minAccCost << "\n";
    llvm::errs() << "Computation cost required: " << bestCompCost << "\n";
  } else {
    for (const auto &pair : costToAccuracyMap) {
      InstructionCost compCost = pair.first;
      double accCost = pair.second;

      if (compCost <= FPOptComputationCostBudget && accCost < minAccCost) {
        minAccCost = accCost;
        bestCompCost = compCost;
      }
    }

    if (minAccCost == std::numeric_limits<double>::infinity()) {
      llvm::errs() << "No solution found within the computation cost budget!\n";
      return changed;
    }

    llvm::errs() << "Minimum accuracy cost within budget: " << minAccCost
                 << "\n";
    llvm::errs() << "Computation cost budget used: " << bestCompCost << "\n";
  }

  assert(costToSolutionMap.find(bestCompCost) != costToSolutionMap.end() &&
         "FPOpt DP solver: expected a solution!");

  llvm::errs() << "\n!!! DP solver: Applying solution ... !!!\n";
  for (const auto &solution : costToSolutionMap[bestCompCost]) {
    std::visit(
        [&](auto *item) {
          using T = std::decay_t<decltype(*item)>;
          if constexpr (std::is_same_v<T, CandidateOutput>) {
            llvm::errs() << "Applying solution for " << item->expr << " --("
                         << solution.candidateIndex << ")-> "
                         << item->candidates[solution.candidateIndex].expr
                         << "\n";
            item->apply(solution.candidateIndex, valueToNodeMap,
                        symbolToValueMap);
          } else if constexpr (std::is_same_v<T, CandidateSubgraph>) {
            llvm::errs() << "Applying solution for CS: "
                         << item->candidates[solution.candidateIndex].desc
                         << " (#" << solution.candidateIndex << ")\n";
            item->apply(solution.candidateIndex);
          } else {
            llvm_unreachable(
                "accuracyDPSolver: Unexpected type of solution step");
          }
        },
        solution.item);
    changed = true;
  }
  llvm::errs() << "!!! DP Solver: Solution applied !!!\n\n";

  return changed;
}