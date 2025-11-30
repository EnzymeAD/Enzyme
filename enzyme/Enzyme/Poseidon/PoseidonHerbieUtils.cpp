//=- PoseidonHerbieUtils.cpp - Herbie integration utilities ---------------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities for integrating with the Herbie tool for
// floating-point expression optimization.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InstructionCost.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"

#include "../Utils.h"
#include "Poseidon.h"
#include "PoseidonEvaluators.h"
#include "PoseidonHerbieUtils.h"
#include "PoseidonSolvers.h"
#include "PoseidonTypes.h"
#include "PoseidonUtils.h"

#include <fstream>
#include <iomanip>
#include <regex>
#include <string>
#include <unordered_map>

using namespace llvm;

extern "C" {
cl::opt<int> HerbieNumThreads("herbie-num-threads", cl::init(8), cl::Hidden,
                              cl::desc("Number of threads Herbie uses"));
cl::opt<int> HerbieTimeout("herbie-timeout", cl::init(9999), cl::Hidden,
                           cl::desc("Herbie's timeout to use for each "
                                    "candidate expressions."));
cl::opt<int>
    HerbieNumPoints("herbie-num-pts", cl::init(1024), cl::Hidden,
                    cl::desc("Number of input points Herbie uses to evaluate "
                             "candidate expressions."));
cl::opt<int> HerbieNumIters(
    "herbie-num-iters", cl::init(6), cl::Hidden,
    cl::desc("Number of times Herbie attempts to improve accuracy."));
cl::opt<int> HerbieNumEnodes(
    "herbie-num-enodes", cl::init(8000), cl::Hidden,
    cl::desc("Number of equivalence graph nodes to use when doing algebraic "
             "reasoning in Herbie."));
cl::opt<bool> HerbieDisableNumerics(
    "herbie-disable-numerics", cl::init(false), cl::Hidden,
    cl::desc("Disable Herbie rewrite rules that produce numerical shorthands "
             "expm1, log1p, fma, and hypot"));
cl::opt<bool> HerbieDisableArithmetic(
    "herbie-disable-arithmetic", cl::init(false), cl::Hidden,
    cl::desc("Disable Herbie rewrite rules on basic arithmetic fasts."));
cl::opt<bool> HerbieDisableFractions(
    "herbie-disable-fractions", cl::init(false), cl::Hidden,
    cl::desc("Disable Herbie rewrite rules on fraction arithmetic."));
cl::opt<bool>
    HerbieDisableTaylor("herbie-disable-taylor", cl::init(false), cl::Hidden,
                        cl::desc("Disable Herbie's series expansion"));
cl::opt<bool> HerbieDisableSetupSimplify(
    "herbie-disable-setup-simplify", cl::init(false), cl::Hidden,
    cl::desc("Stop Herbie from pre-simplifying expressions"));
cl::opt<bool> HerbieDisableGenSimplify(
    "herbie-disable-gen-simplify", cl::init(false), cl::Hidden,
    cl::desc("Stop Herbie from simplifying expressions "
             "during the main improvement loop"));
cl::opt<bool> HerbieDisableRegime(
    "herbie-disable-regime", cl::init(false), cl::Hidden,
    cl::desc("Stop Herbie from branching between expressions candidates"));
cl::opt<bool> HerbieDisableBranchExpr(
    "herbie-disable-branch-expr", cl::init(false), cl::Hidden,
    cl::desc("Stop Herbie from branching on expressions"));
cl::opt<bool> HerbieDisableAvgError(
    "herbie-disable-avg-error", cl::init(false), cl::Hidden,
    cl::desc("Make Herbie choose the candidates with the least maximum error"));
}

std::shared_ptr<FPNode> parseHerbieExpr(
    const std::string &expr,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap) {
  // if (FPOptPrint)
  //   llvm::errs() << "Parsing: " << expr << "\n";
  std::string trimmedExpr = expr;
  trimmedExpr.erase(0, trimmedExpr.find_first_not_of(" "));
  trimmedExpr.erase(trimmedExpr.find_last_not_of(" ") + 1);

  // Arguments
  if (trimmedExpr.front() != '(' && trimmedExpr.front() != '#') {
    if (auto node = valueToNodeMap[symbolToValueMap[trimmedExpr]]) {
      return node;
    }
  }

  // Constants
  static const std::regex constantPattern(
      "^#s\\(literal\\s+([-+]?\\d+(/\\d+)?|[-+]?inf\\.0)\\s+(\\w+)\\)$");
  static const std::regex plainConstantPattern(
      R"(^([-+]?(\d+(\.\d+)?)(/\d+)?|[-+]?inf\.0))");

  {
    std::smatch matches;
    if (std::regex_match(trimmedExpr, matches, constantPattern)) {
      std::string value = matches[1].str();
      std::string dtype = matches[3].str();
      if (dtype == "binary64") {
        dtype = "f64";
      } else if (dtype == "binary32") {
        dtype = "f32";
      } else {
        std::string msg =
            "Herbie expr parser: Unexpected constant dtype: " + dtype;
        llvm_unreachable(msg.c_str());
      }
      // if (FPOptPrint)
      //   llvm::errs() << "Herbie expr parser: Found __const " << value
      //                << " with dtype " << dtype << "\n";
      return std::make_shared<FPConst>(value, dtype);
    } else if (std::regex_match(trimmedExpr, matches, plainConstantPattern)) {
      std::string value = matches[1].str();
      std::string dtype = "f64"; // Assume f64 by default
      return std::make_shared<FPConst>(value, dtype);
    }
  }

  if (trimmedExpr.substr(0, 9) == "#s(approx") {
    if (trimmedExpr.back() != ')') {
      llvm_unreachable(("Malformed approx expression: " + trimmedExpr).c_str());
    }
    std::string inner = trimmedExpr.substr(9, trimmedExpr.size() - 9 - 1);
    inner.erase(0, inner.find_first_not_of(" "));
    inner.erase(inner.find_last_not_of(" ") + 1);

    int depth = 0;
    size_t splitPos = std::string::npos;
    for (size_t i = 0; i < inner.size(); ++i) {
      if (inner[i] == '(')
        depth++;
      else if (inner[i] == ')')
        depth--;
      else if (inner[i] == ' ' && depth == 0) {
        splitPos = i;
        break;
      }
    }
    if (splitPos == std::string::npos) {
      llvm_unreachable(("Malformed approx expression: " + trimmedExpr).c_str());
    }
    std::string resultPart = inner.substr(splitPos + 1);
    resultPart.erase(0, resultPart.find_first_not_of(" "));
    resultPart.erase(resultPart.find_last_not_of(" ") + 1);
    return parseHerbieExpr(resultPart, valueToNodeMap, symbolToValueMap);
  }

  if (trimmedExpr.front() != '(' || trimmedExpr.back() != ')') {
    llvm::errs() << "Unexpected subexpression: " << trimmedExpr << "\n";
    assert(0 && "Failed to parse Herbie expression");
  }

  trimmedExpr = trimmedExpr.substr(1, trimmedExpr.size() - 2);

  auto endOp = trimmedExpr.find(' ');
  std::string fullOp = trimmedExpr.substr(0, endOp);

  size_t pos = fullOp.find('.');
  std::string dtype;
  std::string op;
  if (pos != std::string::npos) {
    op = fullOp.substr(0, pos);
    dtype = fullOp.substr(pos + 1);
    assert(dtype == "f64" || dtype == "f32");
    // llvm::errs() << "Herbie expr parser: Found operator " << op
    //              << " with dtype " << dtype << "\n";
  } else {
    op = fullOp;
    // llvm::errs() << "Herbie expr parser: Found operator " << op << "\n";
  }

  auto node = std::make_shared<FPNode>(op, dtype);

  int depth = 0;
  auto start = trimmedExpr.find_first_not_of(" ", endOp);
  std::string::size_type curr;
  for (curr = start; curr < trimmedExpr.size(); ++curr) {
    if (trimmedExpr[curr] == '(')
      depth++;
    if (trimmedExpr[curr] == ')')
      depth--;
    if (depth == 0 && trimmedExpr[curr] == ' ') {
      node->addOperand(parseHerbieExpr(trimmedExpr.substr(start, curr - start),
                                       valueToNodeMap, symbolToValueMap));
      start = curr + 1;
    }
  }
  if (start < curr) {
    node->addOperand(parseHerbieExpr(trimmedExpr.substr(start, curr - start),
                                     valueToNodeMap, symbolToValueMap));
  }

  return node;
}

bool improveViaHerbie(
    const std::vector<std::string> &inputExprs,
    std::vector<CandidateOutput> &COs, Module *M,
    const TargetTransformInfo &TTI,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap,
    int subgraphIdx) {
  std::string Program = HERBIE_BINARY;
  llvm::errs() << "random seed: " << std::to_string(FPOptRandomSeed) << "\n";

  SmallVector<std::string> BaseArgs = {
      Program,        "report",
      "--seed",       std::to_string(FPOptRandomSeed),
      "--timeout",    std::to_string(HerbieTimeout),
      "--threads",    std::to_string(HerbieNumThreads),
      "--num-points", std::to_string(HerbieNumPoints),
      "--num-iters",  std::to_string(HerbieNumIters),
      "--num-enodes", std::to_string(HerbieNumEnodes)};

  BaseArgs.push_back("--disable");
  BaseArgs.push_back("generate:proofs");

  if (HerbieDisableNumerics) {
    BaseArgs.push_back("--disable");
    BaseArgs.push_back("rules:numerics");
  }

  if (HerbieDisableArithmetic) {
    BaseArgs.push_back("--disable");
    BaseArgs.push_back("rules:arithmetic");
  }

  if (HerbieDisableFractions) {
    BaseArgs.push_back("--disable");
    BaseArgs.push_back("rules:fractions");
  }

  if (HerbieDisableSetupSimplify) {
    BaseArgs.push_back("--disable");
    BaseArgs.push_back("setup:simplify");
  }

  if (HerbieDisableGenSimplify) {
    BaseArgs.push_back("--disable");
    BaseArgs.push_back("generate:simplify");
  }

  if (HerbieDisableTaylor) {
    BaseArgs.push_back("--disable");
    BaseArgs.push_back("generate:taylor");
  }

  if (HerbieDisableRegime) {
    BaseArgs.push_back("--disable");
    BaseArgs.push_back("reduce:regimes");
  }

  if (HerbieDisableBranchExpr) {
    BaseArgs.push_back("--disable");
    BaseArgs.push_back("reduce:branch-expressions");
  }

  if (HerbieDisableAvgError) {
    BaseArgs.push_back("--disable");
    BaseArgs.push_back("reduce:avg-error");
  }

  SmallVector<SmallVector<std::string>> BaseArgsList;
  BaseArgsList.push_back(BaseArgs);

  std::vector<std::unordered_set<std::string>> seenExprs(COs.size());

  bool success = false;

  auto processHerbieOutput = [&](const std::string &content,
                                 bool skipEvaluation = false) -> bool {
    Expected<json::Value> parsed = json::parse(content);
    if (!parsed) {
      llvm::errs() << "Failed to parse Herbie result!\n";
      return false;
    }

    json::Object *obj = parsed->getAsObject();
    json::Array &tests = *obj->getArray("tests");

    for (size_t testIndex = 0; testIndex < tests.size(); ++testIndex) {
      auto &test = *tests[testIndex].getAsObject();

      StringRef bestExpr = test.getString("output").value();
      if (bestExpr == "#f") {
        continue;
      }

      StringRef ID = test.getString("name").value();
      size_t index = std::stoul(ID.str());
      if (index >= COs.size()) {
        llvm::errs() << "Invalid CO index: " << index << "\n";
        continue;
      }

      CandidateOutput &CO = COs[index];
      auto &seenExprSet = seenExprs[index];

      double bits = test.getNumber("bits").value();
      json::Array &costAccuracy = *test.getArray("cost-accuracy");

      json::Array &initial = *costAccuracy[0].getAsArray();
      double initialCostVal = initial[0].getAsNumber().value();
      double initialAccuracy = 1.0 - initial[1].getAsNumber().value() / bits;
      double initialCost = 1.0;

      CO.initialHerbieCost = initialCost;
      CO.initialHerbieAccuracy = initialAccuracy;

      if (seenExprSet.count(bestExpr.str()) == 0) {
        seenExprSet.insert(bestExpr.str());

        json::Array &best = *costAccuracy[1].getAsArray();
        double bestCost = best[0].getAsNumber().value() / initialCostVal;
        double bestAccuracy = 1.0 - best[1].getAsNumber().value() / bits;

        RewriteCandidate bestCandidate(bestCost, bestAccuracy, bestExpr.str());
        if (!skipEvaluation) {
          bestCandidate.CompCost = getCompCost(
              bestExpr.str(), M, TTI, valueToNodeMap, symbolToValueMap,
              cast<Instruction>(CO.oldOutput)->getFastMathFlags());
        }
        CO.candidates.push_back(bestCandidate);
      }

      json::Array &alternatives = *costAccuracy[2].getAsArray();

      // Handle alternatives
      for (size_t j = 0; j < alternatives.size(); ++j) {
        json::Array &entry = *alternatives[j].getAsArray();
        StringRef expr = entry[2].getAsString().value();

        if (seenExprSet.count(expr.str()) != 0) {
          continue;
        }
        seenExprSet.insert(expr.str());

        double cost = entry[0].getAsNumber().value() / initialCostVal;
        double accuracy = 1.0 - entry[1].getAsNumber().value() / bits;

        RewriteCandidate candidate(cost, accuracy, expr.str());
        if (!skipEvaluation) {
          candidate.CompCost =
              getCompCost(expr.str(), M, TTI, valueToNodeMap, symbolToValueMap,
                          cast<Instruction>(CO.oldOutput)->getFastMathFlags());
        }
        CO.candidates.push_back(candidate);
      }

      if (!skipEvaluation) {
        setUnifiedAccuracyCost(CO, valueToNodeMap, symbolToValueMap);
      }
    }
    return true;
  };

  for (size_t baseArgsIndex = 0; baseArgsIndex < BaseArgsList.size();
       ++baseArgsIndex) {
    const auto &BaseArgs = BaseArgsList[baseArgsIndex];
    std::string content;

    // Try to get cached Herbie output first
    std::string cacheFilePath;
    bool cached = false;

    if (!FPOptCachePath.empty()) {
      cacheFilePath = FPOptCachePath + "/cachedHerbieOutput_" +
                      std::to_string(subgraphIdx) + "_" +
                      std::to_string(baseArgsIndex) + ".txt";
      std::ifstream cacheFile(cacheFilePath);
      if (cacheFile) {
        content.assign((std::istreambuf_iterator<char>(cacheFile)),
                       std::istreambuf_iterator<char>());
        cacheFile.close();
        llvm::errs() << "Using cached Herbie output from " << cacheFilePath
                     << "\n";
        cached = true;
      }
    }

    // If we have cached output, process it directly
    if (cached) {
      llvm::errs() << "Herbie output: " << content << "\n";
      std::string dpCacheFilePath = FPOptCachePath + "/table.json";
      bool skipEvaluation = FPOptSolverType == "dp" &&
                            !FPOptCachePath.empty() &&
                            llvm::sys::fs::exists(dpCacheFilePath);
      if (processHerbieOutput(content, skipEvaluation)) {
        success = true;
      }
      continue;
    }

    // No cached result, need to run Herbie
    SmallString<32> tmpin, tmpout;

    if (llvm::sys::fs::createUniqueFile("herbie_input_%%%%%%%%%%%%%%%%", tmpin,
                                        llvm::sys::fs::perms::owner_all)) {
      llvm::errs() << "Failed to create a unique input file.\n";
      continue;
    }

    if (llvm::sys::fs::createUniqueDirectory("herbie_output_%%%%%%%%%%%%%%%%",
                                             tmpout)) {
      llvm::errs() << "Failed to create a unique output directory.\n";
      if (auto EC = llvm::sys::fs::remove(tmpin))
        llvm::errs() << "Warning: Failed to remove temporary input file: "
                     << EC.message() << "\n";
      continue;
    }

    std::ofstream input(tmpin.c_str());
    if (!input) {
      llvm::errs() << "Failed to open input file.\n";
      if (auto EC = llvm::sys::fs::remove(tmpin))
        llvm::errs() << "Warning: Failed to remove temporary input file: "
                     << EC.message() << "\n";
      if (auto EC = llvm::sys::fs::remove_directories(tmpout))
        llvm::errs() << "Warning: Failed to remove temporary output directory: "
                     << EC.message() << "\n";
      continue;
    }
    for (const auto &expr : inputExprs) {
      input << expr << "\n";
    }
    input.close();

    SmallVector<StringRef> Args;
    Args.reserve(BaseArgs.size());
    for (const auto &arg : BaseArgs) {
      Args.emplace_back(arg);
    }

    Args.push_back(tmpin);
    Args.push_back(tmpout);

    std::string ErrMsg;
    bool ExecutionFailed = false;

    if (FPOptPrint) {
      llvm::errs() << "Executing Herbie with arguments: ";
      for (const auto &arg : Args) {
        llvm::errs() << arg << " ";
      }
      llvm::errs() << "\n";
    }

    llvm::sys::ExecuteAndWait(Program, Args, /*Env=*/{},
                              /*Redirects=*/{},
                              /*SecondsToWait=*/0, /*MemoryLimit=*/0, &ErrMsg,
                              &ExecutionFailed);

    std::remove(tmpin.c_str());
    if (ExecutionFailed) {
      llvm::errs() << "Execution failed: " << ErrMsg << "\n";
      if (auto EC = llvm::sys::fs::remove_directories(tmpout))
        llvm::errs() << "Warning: Failed to remove temporary output directory: "
                     << EC.message() << "\n";
      continue;
    }

    std::ifstream output((tmpout + "/results.json").str());
    if (!output) {
      llvm::errs() << "Failed to open output file.\n";
      if (auto EC = llvm::sys::fs::remove_directories(tmpout))
        llvm::errs() << "Warning: Failed to remove temporary output directory: "
                     << EC.message() << "\n";
      continue;
    }
    content.assign((std::istreambuf_iterator<char>(output)),
                   std::istreambuf_iterator<char>());
    output.close();
    if (auto EC = llvm::sys::fs::remove_directories(tmpout))
      llvm::errs() << "Warning: Failed to remove temporary output directory: "
                   << EC.message() << "\n";

    llvm::errs() << "Herbie output: " << content << "\n";

    // Save output to cache if needed
    if (!FPOptCachePath.empty()) {
      if (auto EC = llvm::sys::fs::create_directories(FPOptCachePath, true))
        llvm::errs() << "Warning: Could not create cache directory: "
                     << EC.message() << "\n";
      std::ofstream cacheFile(cacheFilePath);
      if (!cacheFile) {
        llvm_unreachable("Failed to open cache file for writing");
      } else {
        cacheFile << content;
        cacheFile.close();
        llvm::errs() << "Saved Herbie output to cache file " << cacheFilePath
                     << "\n";
      }
    }

    // Process the output
    if (processHerbieOutput(content, false)) {
      success = true;
    }
  }

  return success;
}

std::string getHerbieOperator(const Instruction &I) {
  switch (I.getOpcode()) {
  case Instruction::FNeg:
    return "neg";
  case Instruction::FAdd:
    return "+";
  case Instruction::FSub:
    return "-";
  case Instruction::FMul:
    return "*";
  case Instruction::FDiv:
    return "/";
  case Instruction::Call: {
    const CallInst *CI = dyn_cast<CallInst>(&I);
    assert(CI && CI->getCalledFunction() &&
           "getHerbieOperator: Call without a function");

    StringRef funcName = CI->getCalledFunction()->getName();

    // LLVM intrinsics
    if (startsWith(funcName, "llvm.")) {
      std::regex regex("llvm\\.(\\w+)\\.?.*");
      std::smatch matches;
      std::string nameStr = funcName.str();
      if (std::regex_search(nameStr, matches, regex) && matches.size() > 1) {
        std::string intrinsic = matches[1];
        // Special case mappings
        if (intrinsic == "fmuladd")
          return "fma";
        if (intrinsic == "maxnum")
          return "fmax";
        if (intrinsic == "minnum")
          return "fmin";
        if (intrinsic == "powi")
          return "pow";
        return intrinsic;
      }
      assert(0 && "getHerbieOperator: Unknown LLVM intrinsic");
    }
    // libm functions
    else {
      std::string name = funcName.str();
      if (!name.empty() && name.back() == 'f') {
        name.pop_back();
      }
      return name;
    }
  }
  default:
    assert(0 && "getHerbieOperator: Unknown operator");
  }
}

std::string getPrecondition(
    const SmallSet<std::string, 8> &args,
    const std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    const std::unordered_map<std::string, Value *> &symbolToValueMap) {
  std::string preconditions;

  for (const auto &arg : args) {
    const auto node = valueToNodeMap.at(symbolToValueMap.at(arg));
    double lower = node->getLowerBound();
    double upper = node->getUpperBound();

    if (upper - lower < 1e-10 && !std::isinf(lower) && !std::isinf(upper)) {
      double midpoint = (lower + upper) / 2.0;
      double tolerance = std::max(1e-10, std::abs(midpoint) * 1e-6);
      lower = midpoint - tolerance;
      upper = midpoint + tolerance;
    }

    std::ostringstream lowerStr, upperStr;
    lowerStr << std::setprecision(std::numeric_limits<double>::max_digits10)
             << std::scientific << lower;
    upperStr << std::setprecision(std::numeric_limits<double>::max_digits10)
             << std::scientific << upper;

    preconditions +=
        " (<=" +
        (std::isinf(lower) ? (lower > 0 ? " INFINITY" : " (- INFINITY)")
                           : (" " + lowerStr.str())) +
        " " + arg +
        (std::isinf(upper) ? (upper > 0 ? " INFINITY" : " (- INFINITY)")
                           : (" " + upperStr.str())) +
        ")";
  }

  return preconditions.empty() ? "TRUE" : "(and" + preconditions + ")";
}

void setUnifiedAccuracyCost(
    CandidateOutput &CO,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap) {

  SmallVector<MapVector<Value *, double>, 4> sampledPoints;
  getSampledPoints(CO.subgraph->inputs.getArrayRef(), valueToNodeMap,
                   symbolToValueMap, sampledPoints);

  SmallVector<double, 4> goldVals;
  goldVals.resize(FPOptNumSamples);

  double origCost = 0.0;
  if (FPOptReductionEval == "geomean") {
    double sumLog = 0.0;
    unsigned count = 0;
    for (const auto &pair : enumerate(sampledPoints)) {
      std::shared_ptr<FPNode> node = valueToNodeMap[CO.oldOutput];
      SmallVector<double, 1> results;
      getMPFRValues({node.get()}, pair.value(), results, true, 53);
      double goldVal = results[0];
      goldVals[pair.index()] = goldVal;

      getFPValues({node.get()}, pair.value(), results);
      double realVal = results[0];
      double error = std::fabs(goldVal - realVal);
      if (!std::isnan(error)) {
        if (error == 0.0) {
          if (FPOptGeoMeanEps == 0.0)
            error = getOneULP(goldVal);
          else
            error += FPOptGeoMeanEps;
        } else if (FPOptGeoMeanEps != 0.0) {
          error += FPOptGeoMeanEps;
        }
        sumLog += std::log(error);
        ++count;
      }
    }
    assert(count != 0 && "No valid sample found for original expr");
    origCost = std::exp(sumLog / count);
  } else if (FPOptReductionEval == "arithmean") {
    double sum = 0.0;
    unsigned count = 0;
    for (const auto &pair : enumerate(sampledPoints)) {
      std::shared_ptr<FPNode> node = valueToNodeMap[CO.oldOutput];
      SmallVector<double, 1> results;
      getMPFRValues({node.get()}, pair.value(), results, true, 53);
      double goldVal = results[0];
      goldVals[pair.index()] = goldVal;

      getFPValues({node.get()}, pair.value(), results);
      double realVal = results[0];
      double error = std::fabs(goldVal - realVal);
      if (!std::isnan(error)) {
        sum += error;
        ++count;
      }
    }
    assert(count != 0 && "No valid sample found for original expr");
    origCost = sum / count;
  } else if (FPOptReductionEval == "maxabs") {
    double maxErr = 0.0;
    for (const auto &pair : enumerate(sampledPoints)) {
      std::shared_ptr<FPNode> node = valueToNodeMap[CO.oldOutput];
      SmallVector<double, 1> results;
      getMPFRValues({node.get()}, pair.value(), results, true, 53);
      double goldVal = results[0];
      goldVals[pair.index()] = goldVal;

      getFPValues({node.get()}, pair.value(), results);
      double realVal = results[0];
      double error = std::fabs(goldVal - realVal);
      if (!std::isnan(error))
        maxErr = std::max(maxErr, error);
    }
    origCost = maxErr;
  } else {
    llvm_unreachable("Unknown fpopt-reduction strategy");
  }
  CO.initialAccCost = origCost * std::fabs(CO.grad);
  if (std::isnan(CO.initialAccCost)) {
    llvm::errs() << "Warning: NaN in initialAccCost computation:\n";
    llvm::errs() << "  origCost = " << origCost << "\n";
    llvm::errs() << "  CO.grad = " << CO.grad << "\n";
    llvm::errs() << "  fabs(CO.grad) = " << std::fabs(CO.grad) << "\n";
  }

  SmallVector<RewriteCandidate, 4> newCandidates;
  for (auto &candidate : CO.candidates) {
    bool discardCandidate = false;
    double candCost = 0.0;

    std::shared_ptr<FPNode> parsedNode =
        parseHerbieExpr(candidate.expr, valueToNodeMap, symbolToValueMap);

    if (FPOptReductionEval == "geomean") {
      double sumLog = 0.0;
      unsigned count = 0;
      for (const auto &pair : enumerate(sampledPoints)) {
        SmallVector<double, 1> results;
        getFPValues({parsedNode.get()}, pair.value(), results);
        double realVal = results[0];
        double goldVal = goldVals[pair.index()];

        if (FPOptStrictMode && (!std::isnan(goldVal)) && std::isnan(realVal)) {
          discardCandidate = true;
          break;
        }

        double error = std::fabs(goldVal - realVal);
        if (!std::isnan(error)) {
          if (error == 0.0) {
            if (FPOptGeoMeanEps == 0.0)
              error = getOneULP(goldVal);
            else
              error += FPOptGeoMeanEps;
          } else if (FPOptGeoMeanEps != 0.0) {
            error += FPOptGeoMeanEps;
          }
          sumLog += std::log(error);
          ++count;
        }
      }
      if (!discardCandidate) {
        if (count == 0) {
          discardCandidate = true;
        } else {
          candCost = std::exp(sumLog / count);
        }
      }
    } else if (FPOptReductionEval == "arithmean") {
      double sum = 0.0;
      unsigned count = 0;
      for (const auto &pair : enumerate(sampledPoints)) {
        SmallVector<double, 1> results;
        getFPValues({parsedNode.get()}, pair.value(), results);
        double realVal = results[0];
        double goldVal = goldVals[pair.index()];

        if (FPOptStrictMode && !std::isnan(goldVal) && std::isnan(realVal)) {
          discardCandidate = true;
          break;
        }

        double error = std::fabs(goldVal - realVal);
        if (!std::isnan(error)) {
          sum += error;
          ++count;
        }
      }
      if (!discardCandidate) {
        if (count == 0) {
          discardCandidate = true;
        } else {
          candCost = sum / count;
        }
      }
    } else if (FPOptReductionEval == "maxabs") {
      double maxErr = 0.0;
      bool hasValid = false;
      for (const auto &pair : enumerate(sampledPoints)) {
        SmallVector<double, 1> results;
        getFPValues({parsedNode.get()}, pair.value(), results);
        double realVal = results[0];
        double goldVal = goldVals[pair.index()];

        if (FPOptStrictMode && !std::isnan(goldVal) && std::isnan(realVal)) {
          discardCandidate = true;
          break;
        }

        double error = std::fabs(goldVal - realVal);
        if (!std::isnan(error)) {
          hasValid = true;
          maxErr = std::max(maxErr, error);
        }
      }
      if (!discardCandidate) {
        if (!hasValid) {
          discardCandidate = true;
        } else {
          candCost = maxErr;
        }
      }
    } else {
      llvm_unreachable("Unknown fpopt-reduction strategy");
    }

    if (!discardCandidate) {
      candidate.accuracyCost = candCost * std::fabs(CO.grad);
      assert(!std::isnan(candidate.accuracyCost));
      newCandidates.push_back(std::move(candidate));
    }
  }
  CO.candidates = std::move(newCandidates);
}

InstructionCost getCompCost(
    const std::string &expr, Module *M, const TargetTransformInfo &TTI,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap,
    const FastMathFlags &FMF) {
  // llvm::errs() << "Evaluating cost of " << expr << "\n";
  SmallSet<std::string, 8> argStrSet;
  getUniqueArgs(expr, argStrSet);

  SetVector<Value *> args;
  SmallVector<Type *, 8> argTypes;
  SmallVector<std::string, 8> argNames;
  for (const auto &argStr : argStrSet) {
    Value *argValue = symbolToValueMap[argStr];
    args.insert(argValue);
    argTypes.push_back(argValue->getType());
    argNames.push_back(argStr);
  }

  auto parsedNode = parseHerbieExpr(expr, valueToNodeMap, symbolToValueMap);

  Type *ReturnType = nullptr;
  for (Type *ArgTy : argTypes) {
    if (ArgTy->isFloatingPointTy()) {
      ReturnType = ArgTy;
      break;
    }
    if (ArgTy->isVectorTy()) {
      if (auto *VT = dyn_cast<VectorType>(ArgTy)) {
        if (VT->getElementType()->isFloatingPointTy()) {
          ReturnType = ArgTy;
          break;
        }
      }
    }
  }
  if (!ReturnType) {
    if (parsedNode->dtype == "f32")
      ReturnType = Type::getFloatTy(M->getContext());
    else if (parsedNode->dtype == "f16")
      ReturnType = Type::getHalfTy(M->getContext());
    else
      ReturnType = Type::getDoubleTy(M->getContext());
  }

  FunctionType *FT = FunctionType::get(ReturnType, argTypes, false);
  Function *tempFunction =
      Function::Create(FT, Function::InternalLinkage, "tempFunc", M);

  ValueToValueMapTy VMap;
  Function::arg_iterator AI = tempFunction->arg_begin();
  for (const auto &argStr : argNames) {
    VMap[symbolToValueMap[argStr]] = &*AI;
    ++AI;
  }

  BasicBlock *entry =
      BasicBlock::Create(M->getContext(), "entry", tempFunction);

  IRBuilder<> builder(entry);

  builder.setFastMathFlags(FMF);
  Value *RetVal = parsedNode->getLLValue(builder, &VMap);
  assert(RetVal && "Parsed node did not produce a value");
  assert((RetVal->getType() == ReturnType) &&
         "Return value type mismatch with temp function return type");
  builder.CreateRet(RetVal);

  // llvm::errs() << "Temp function before optimizations:\n";
  // tempFunction->print(llvm::errs());

  runPoseidonFunctionSimplify(*tempFunction, OptimizationLevel::O3);

  // llvm::errs() << "Temp function after optimizations:\n";
  // tempFunction->print(llvm::errs());

  InstructionCost cost = getCompCost(tempFunction, TTI);

  tempFunction->eraseFromParent();
  return cost;
}