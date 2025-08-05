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

#include <random>

using namespace llvm;

extern "C" {
cl::opt<std::string> FPOptSolverType("fpopt-solver-type", cl::init("dp"),
                                     cl::Hidden,
                                     cl::desc("Which solver to use; "
                                              "either 'dp' or 'greedy'"));
cl::opt<int64_t> FPOptComputationCostBudget(
    "fpopt-comp-cost-budget", cl::init(100000000000L), cl::Hidden,
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
    "fpopt-early-prune", cl::init(false), cl::Hidden,
    cl::desc("Prune dominated candidates in expression transformation phases"));
cl::opt<double> FPOptCostDominanceThreshold(
    "fpopt-cost-dom-thres", cl::init(0.05), cl::Hidden,
    cl::desc("The threshold for cost dominance in DP solver"));
cl::opt<double> FPOptAccuracyDominanceThreshold(
    "fpopt-acc-dom-thres", cl::init(0.05), cl::Hidden,
    cl::desc("The threshold for accuracy dominance in DP solver"));
}

#if LLVM_VERSION_MAJOR >= 21
#define GET_INSTRUCTION_COST(cost) (cost.getValue())
#else
#define GET_INSTRUCTION_COST(cost) (cost.getValue().value())
#endif

// Given the cost budget `FPOptComputationCostBudget`, we want to minimize the
// accuracy cost of the rewritten expressions.
bool accuracyGreedySolver(
    SmallVector<ApplicableOutput, 4> &AOs, SmallVector<ApplicableFPCC, 4> &ACCs,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap) {
  bool changed = false;
  llvm::errs() << "Starting accuracy greedy solver with computation budget: "
               << FPOptComputationCostBudget << "\n";
  InstructionCost totalComputationCost = 0;

  SmallVector<size_t, 4> aoIndices;
  for (size_t i = 0; i < AOs.size(); ++i) {
    aoIndices.push_back(i);
  }
  std::mt19937 g(FPOptRandomSeed);
  std::shuffle(aoIndices.begin(), aoIndices.end(), g);

  for (size_t idx : aoIndices) {
    auto &AO = AOs[idx];
    int bestCandidateIndex = -1;
    double bestAccuracyCost = std::numeric_limits<double>::infinity();
    InstructionCost bestCandidateComputationCost;

    for (const auto &candidate : enumerate(AO.candidates)) {
      size_t i = candidate.index();
      auto candCompCost = AO.getCompCostDelta(i);
      auto candAccCost = AO.getAccCostDelta(i);
      // llvm::errs() << "AO Candidate " << i << " for " << AO.expr
      //              << " has accuracy cost: " << candAccCost
      //              << " and computation cost: " << candCompCost << "\n";

      if (totalComputationCost + candCompCost <= FPOptComputationCostBudget) {
        if (candAccCost < bestAccuracyCost) {
          // llvm::errs() << "AO Candidate " << i << " selected!\n";
          bestCandidateIndex = i;
          bestAccuracyCost = candAccCost;
          bestCandidateComputationCost = candCompCost;
        }
      }
    }

    if (bestCandidateIndex != -1) {
      AO.apply(bestCandidateIndex, valueToNodeMap, symbolToValueMap);
      changed = true;
      totalComputationCost += bestCandidateComputationCost;
      if (EnzymePrintFPOpt) {
        llvm::errs() << "Greedy solver selected candidate "
                     << bestCandidateIndex << " for " << AO.expr
                     << " with accuracy cost: " << bestAccuracyCost
                     << " and computation cost: "
                     << bestCandidateComputationCost << "\n";
      }
    }
  }

  SmallVector<size_t, 4> accIndices;
  for (size_t i = 0; i < ACCs.size(); ++i) {
    accIndices.push_back(i);
  }
  std::shuffle(accIndices.begin(), accIndices.end(), g);

  for (size_t idx : accIndices) {
    auto &ACC = ACCs[idx];
    int bestCandidateIndex = -1;
    double bestAccuracyCost = std::numeric_limits<double>::infinity();
    InstructionCost bestCandidateComputationCost;

    for (const auto &candidate : enumerate(ACC.candidates)) {
      size_t i = candidate.index();
      auto candCompCost = ACC.getCompCostDelta(i);
      auto candAccCost = ACC.getAccCostDelta(i);
      // llvm::errs() << "ACC Candidate " << i << " (" << candidate.value().desc
      //              << ") has accuracy cost: " << candAccCost
      //              << " and computation cost: " << candCompCost << "\n";

      if (totalComputationCost + candCompCost <= FPOptComputationCostBudget) {
        if (candAccCost < bestAccuracyCost) {
          // llvm::errs() << "ACC Candidate " << i << " selected!\n";
          bestCandidateIndex = i;
          bestAccuracyCost = candAccCost;
          bestCandidateComputationCost = candCompCost;
        }
      }
    }

    if (bestCandidateIndex != -1) {
      ACC.apply(bestCandidateIndex);
      changed = true;
      totalComputationCost += bestCandidateComputationCost;
      if (EnzymePrintFPOpt) {
        llvm::errs() << "Greedy solver selected candidate "
                     << bestCandidateIndex << " for "
                     << ACC.candidates[bestCandidateIndex].desc
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
    SmallVector<ApplicableOutput, 4> &AOs, SmallVector<ApplicableFPCC, 4> &ACCs,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap) {
  bool changed = false;
  llvm::errs() << "Starting accuracy DP solver with computation budget: "
               << FPOptComputationCostBudget << "\n";

  using CostMap = std::map<InstructionCost, double>;
  using SolutionMap = std::map<InstructionCost, SmallVector<SolutionStep>>;

  CostMap costToAccuracyMap;
  SolutionMap costToSolutionMap;
  CostMap newCostToAccuracyMap;
  SolutionMap newCostToSolutionMap;
  CostMap prunedCostToAccuracyMap;
  SolutionMap prunedCostToSolutionMap;

  std::string cacheFilePath = FPOptCachePath + "/table.json";

  if (llvm::sys::fs::exists(cacheFilePath)) {
    llvm::errs() << "Cache file found. Loading DP tables from cache.\n";

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

          if (itemType == "AO") {
            if (itemIndex >= AOs.size()) {
              llvm_unreachable("Invalid ApplicableOutput index in cache file.");
            }
            solutionSteps.emplace_back(&AOs[itemIndex], candidateIndex);
          } else if (itemType == "ACC") {
            if (itemIndex >= ACCs.size()) {
              llvm_unreachable("Invalid ApplicableFPCC index in cache file.");
            }
            solutionSteps.emplace_back(&ACCs[itemIndex], candidateIndex);
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

    std::unordered_map<ApplicableOutput *, size_t> aoPtrToIndex;
    for (size_t i = 0; i < AOs.size(); ++i) {
      aoPtrToIndex[&AOs[i]] = i;
    }
    std::unordered_map<ApplicableFPCC *, size_t> accPtrToIndex;
    for (size_t i = 0; i < ACCs.size(); ++i) {
      accPtrToIndex[&ACCs[i]] = i;
    }

    int AOCounter = 0;

    for (auto &AO : AOs) {
      // It is possible to apply zero candidate for an AO.
      // When no candidate is applied, the resulting accuracy cost
      // and solution steps remain the same.
      newCostToAccuracyMap = costToAccuracyMap;
      newCostToSolutionMap = costToSolutionMap;

      for (const auto &pair : costToAccuracyMap) {
        InstructionCost currCompCost = pair.first;
        double currAccCost = pair.second;

        for (const auto &candidate : enumerate(AO.candidates)) {
          size_t i = candidate.index();
          auto candCompCost = AO.getCompCostDelta(i);
          auto candAccCost = AO.getAccCostDelta(i);

          // Don't ever try to apply a strictly useless candidate
          if (candCompCost >= 0 && candAccCost >= 0.) {
            continue;
          }

          InstructionCost newCompCost = currCompCost + candCompCost;
          double newAccCost = currAccCost + candAccCost;

          // if (EnzymePrintFPOpt)
          //   llvm::errs() << "AO candidate " << i
          //                << " has accuracy cost: " << candAccCost
          //                << " and computation cost: " << candCompCost <<
          //                "\n";

          if (newCostToAccuracyMap.find(newCompCost) ==
                  newCostToAccuracyMap.end() ||
              newCostToAccuracyMap[newCompCost] > newAccCost) {
            newCostToAccuracyMap[newCompCost] = newAccCost;
            newCostToSolutionMap[newCompCost] = costToSolutionMap[currCompCost];
            newCostToSolutionMap[newCompCost].emplace_back(&AO, i);
            // if (EnzymePrintFPOpt)
            //   llvm::errs() << "Updating accuracy map (AO candidate " << i
            //                << "): computation cost " << newCompCost
            //                << " -> accuracy cost " << newAccCost << "\n";
          }
        }
      }

      // TODO: Do not prune AO parts of the DP table since AOs influence ACCs
      if (!FPOptEarlyPrune) {
        costToAccuracyMap = newCostToAccuracyMap;
        costToSolutionMap = newCostToSolutionMap;

        llvm::errs() << "##### Finished processing " << ++AOCounter << " of "
                     << AOs.size() << " AOs #####\n";
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
            // if (EnzymePrintFPOpt)
            //   llvm::errs() << "AO candidate with computation cost: "
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

      llvm::errs() << "##### Finished processing " << ++AOCounter << " of "
                   << AOs.size() << " AOs #####\n";
      llvm::errs() << "Current DP table sizes: " << costToAccuracyMap.size()
                   << "\n";
    }

    int ACCCounter = 0;

    for (auto &ACC : ACCs) {
      // It is possible to apply zero candidate for an ACC.
      // When no candidate is applied, the resulting accuracy cost
      // and solution steps remain the same.
      newCostToAccuracyMap = costToAccuracyMap;
      newCostToSolutionMap = costToSolutionMap;

      for (const auto &pair : costToAccuracyMap) {
        InstructionCost currCompCost = pair.first;
        double currAccCost = pair.second;

        for (const auto &candidate : enumerate(ACC.candidates)) {
          size_t i = candidate.index();
          auto candCompCost =
              ACC.getAdjustedCompCostDelta(i, costToSolutionMap[currCompCost]);
          auto candAccCost =
              ACC.getAdjustedAccCostDelta(i, costToSolutionMap[currCompCost],
                                          valueToNodeMap, symbolToValueMap);

          // Don't ever try to apply a strictly useless candidate
          if (candCompCost >= 0 && candAccCost >= 0.) {
            continue;
          }

          InstructionCost newCompCost = currCompCost + candCompCost;
          double newAccCost = currAccCost + candAccCost;

          // if (EnzymePrintFPOpt)
          //   llvm::errs() << "ACC candidate " << i << " ("
          //                << candidate.value().desc
          //                << ") has accuracy cost: " << candAccCost
          //                << " and computation cost: " << candCompCost <<
          //                "\n";

          if (newCostToAccuracyMap.find(newCompCost) ==
                  newCostToAccuracyMap.end() ||
              newCostToAccuracyMap[newCompCost] > newAccCost) {
            newCostToAccuracyMap[newCompCost] = newAccCost;
            newCostToSolutionMap[newCompCost] = costToSolutionMap[currCompCost];
            newCostToSolutionMap[newCompCost].emplace_back(&ACC, i);
            // if (EnzymePrintFPOpt) {
            // llvm::errs() << "ACC candidate " << i << " ("
            //              << candidate.value().desc
            //              << ") added; has accuracy cost: " << candAccCost
            //              << " and computation cost: " << candCompCost <<
            //              "\n";
            // llvm::errs() << "Updating accuracy map (ACC candidate " << i
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
            // if (EnzymePrintFPOpt)
            //   llvm::errs() << "ACC candidate with computation cost: "
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

      llvm::errs() << "##### Finished processing " << ++ACCCounter << " of "
                   << ACCs.size() << " ACCs #####\n";
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
              if constexpr (std::is_same_v<T, ApplicableOutput>) {
                stepObj["itemType"] = "AO";
                size_t index = aoPtrToIndex[item];
                stepObj["itemIndex"] = static_cast<int64_t>(index);
              } else if constexpr (std::is_same_v<T, ApplicableFPCC>) {
                stepObj["itemType"] = "ACC";
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

    std::error_code EC;
    llvm::raw_fd_ostream cacheFile(cacheFilePath, EC, llvm::sys::fs::OF_Text);
    if (EC) {
      llvm::errs() << "Error writing cache file: " << EC.message() << "\n";
    } else {
      cacheFile << llvm::formatv("{0:2}", llvm::json::Value(std::move(jsonObj)))
                << "\n";
      cacheFile.close();
      llvm::errs() << "DP tables cached to file.\n";
    }
  }

  if (EnzymePrintFPOpt) {
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
                if constexpr (std::is_same_v<T, ApplicableOutput>) {
                  llvm::errs()
                      << "\t\t" << item->expr << " --(" << step.candidateIndex
                      << ")-> " << item->candidates[step.candidateIndex].expr
                      << "\n";
                } else if constexpr (std::is_same_v<T, ApplicableFPCC>) {
                  llvm::errs() << "\t\tACC: "
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

  llvm::errs() << "Critical computation cost range: ["
               << costToAccuracyMap.begin()->first << ", "
               << costToAccuracyMap.rbegin()->first << "]\n";

  llvm::errs() << "DP table contains " << costToAccuracyMap.size()
               << " entries.\n";

  double totalCandidateCompositions = 1.0;
  for (const auto &AO : AOs) {
    // +1 for the "do nothing" possibility
    totalCandidateCompositions *= AO.candidates.size() + 1;
  }
  for (const auto &ACC : ACCs) {
    totalCandidateCompositions *= ACC.candidates.size() + 1;
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

  llvm::errs() << "Minimum accuracy cost within budget: " << minAccCost << "\n";
  llvm::errs() << "Computation cost budget used: " << bestCompCost << "\n";

  assert(costToSolutionMap.find(bestCompCost) != costToSolutionMap.end() &&
         "FPOpt DP solver: expected a solution!");

  llvm::errs() << "\n!!! DP solver: Applying solution ... !!!\n";
  for (const auto &solution : costToSolutionMap[bestCompCost]) {
    std::visit(
        [&](auto *item) {
          using T = std::decay_t<decltype(*item)>;
          if constexpr (std::is_same_v<T, ApplicableOutput>) {
            llvm::errs() << "Applying solution for " << item->expr << " --("
                         << solution.candidateIndex << ")-> "
                         << item->candidates[solution.candidateIndex].expr
                         << "\n";
            item->apply(solution.candidateIndex, valueToNodeMap,
                        symbolToValueMap);
          } else if constexpr (std::is_same_v<T, ApplicableFPCC>) {
            llvm::errs() << "Applying solution for ACC: "
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