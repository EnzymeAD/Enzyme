//=- Herbie.cpp - Herbie integration utilities ----------------------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities for integrating with the Herbie tool for
// floating-point expression optimization.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"

#include "Core/HerbieRun.h"
#include "Herbie.h"
#include "Model/CostModel.h"
#include "Model/Evaluators.h"
#include "Model/Sampling.h"
#include "Pass/Session.h"
#include "Targets/DeviceMath.h"
#include "Targets/Target.h"
#include "Utils.h"

#include <iomanip>
#include <map>
#include <optional>
#include <regex>
#include <string>
#include <unordered_map>

using namespace llvm;

namespace poseidon {

// The Herbie platform this compile searches against. It is the device's, so it
// is derived from the resolved cost model.
namespace {
const HerbiePlatform &getHerbiePlatform(Session &S) {
  HerbiePlatform &P = S.Herbie.Platform;
  if (S.Herbie.PlatformResolved)
    return P;
  S.Herbie.PlatformResolved = true;

  const std::string &NativeArch = getCostModelNativeArch(S.CostModel);
  const Target &T = Target::measuredOn(NativeArch);
  P.Name = T.herbiePlatform(NativeArch);

  if (!S.Opts.Herbie.HerbiePlatform.empty()) {
    // A value that looks like a file is the platform itself; anything else is
    // a platform name compiled into the Herbie binary. The name keeps naming
    // the device family either way, and the digest below distinguishes files.
    StringRef V(S.Opts.Herbie.HerbiePlatform);
    if (V.contains('/') || V.ends_with(".rkt"))
      P.Path = V.str();
    else
      P.Name = V.str();
  } else if (T.isDevice()) {
    // Generated next to the CSV it was priced from, by
    // `poseidon-calibrate --only herbie-platform`.
    StringRef Csv(costModelPath(S.CostModel));
    P.Path =
        (Csv.ends_with(".csv") ? Csv.drop_back(4) : Csv).str() + ".herbie.rkt";
    if (!sys::fs::exists(P.Path))
      report_fatal_error(
          Twine("Poseidon: the Herbie algebraic search needs the platform "
                "generated from this device's cost model, and ") +
          P.Path +
          " does not exist. Generate it with 'poseidon-calibrate --only "
          "herbie-platform --out " +
          costModelPath(S.CostModel) +
          "'. Another architecture's platform is that architecture's prices "
          "and is never substituted.");
  }

  if (!P.Path.empty()) {
    P.Digest = readFileDigest(P.Path);
    if (P.Digest.empty())
      report_fatal_error(Twine("Poseidon: cannot read the Herbie platform ") +
                         P.Path);
  }
  return P;
}
} // namespace

// How this compile runs Herbie: its -poseidon-herbie-* flags.
static HerbieRunOptions herbieRunOptions(Session &S) {
  const Options &O = S.Opts;
  HerbieRunOptions Opts;
  Opts.Program = O.Herbie.HerbieBinary.empty() ? std::string(HERBIE_BINARY)
                                               : O.Herbie.HerbieBinary;
  Opts.Platform = getHerbiePlatform(S);
  Opts.Seed = O.Model.RandomSeed;
  Opts.NumPoints = O.Herbie.HerbieNumPts;
  Opts.NumIters = O.Herbie.HerbieNumIters;
  Opts.NumEnodes = O.Herbie.HerbieNumEnodes;
  Opts.CoreTimeout = O.Herbie.HerbieTimeout;
  Opts.NumThreads = O.Herbie.HerbieNumThreads;
  Opts.Processes = O.Herbie.HerbieProcesses;
  Opts.SubgraphTimeout = O.Herbie.HerbieSubgraphTimeout;
  Opts.CacheDir = O.Herbie.Cache;
  Opts.Print = O.Diagnostics.Print;
  return Opts;
}

void addHerbieKeyParts(Session &S, CacheKey &key) {
  addHerbieKeyParts(key, herbieRunOptions(S), S.Herbie.HerbieDigests);
}

bool improveViaHerbie(Session &S, const std::vector<std::string> &inputExprs,
                      std::vector<RewriteUnit> &COs, int subgraphIdx,
                      StringRef funcTag) {
  const Options &O = S.Opts;
  if (!O.Herbie.HerbieBinary.empty() && O.Diagnostics.Print)
    llvm::errs() << "Poseidon: using Herbie binary '" << O.Herbie.HerbieBinary
                 << "'\n";
  llvm::errs() << "random seed: " << std::to_string(O.Model.RandomSeed) << "\n";
  HerbieRunOptions Opts = herbieRunOptions(S);

  const std::string Label =
      "subgraph " + std::to_string(subgraphIdx) + " of " + funcTag.str();
  HerbieRunResult Run =
      runHerbie(inputExprs, Opts, Label, S.Herbie.HerbieDigests);
  if (Run.Result == HerbieRunResult::Outcome::Timeout)
    for (RewriteUnit &CO : COs)
      CO.herbieStatus = "timeout";
  if (Run.Result == HerbieRunResult::Outcome::Crashed) {
    if (O.Model.StrictMode)
      report_fatal_error(Twine("Poseidon strict mode: Herbie crashed (") +
                         Run.Crash + ") on " + Label);
    for (RewriteUnit &CO : COs)
      CO.herbieStatus = "crashed";
  }
  if (Run.Result != HerbieRunResult::Outcome::Ok)
    return false;

  std::vector<std::unordered_set<std::string>> seenExprs(COs.size());

  auto processHerbieOutput = [&](const std::string &content) -> bool {
    Expected<json::Value> parsed = json::parse(content);
    if (!parsed) {
      llvm::errs() << "Failed to parse Herbie result!\n";
      return false;
    }

    json::Object *obj = parsed->getAsObject();
    json::Array &tests = *obj->getArray("tests");

    for (size_t testIndex = 0; testIndex < tests.size(); ++testIndex) {
      auto &test = *tests[testIndex].getAsObject();

      StringRef testName = test.getString("name").value_or(StringRef("?"));
      StringRef status = test.getString("status").value_or(StringRef("?"));
      llvm::errs() << "[poseidon] Herbie core \"" << testName << "\" (subgraph "
                   << subgraphIdx << " of " << funcTag
                   << "): status=" << status;
      if (auto t = test.getNumber("time"))
        llvm::errs() << " time=" << (*t / 1000.0) << "s";
      if (auto s = test.getNumber("start"))
        llvm::errs() << " start-err-bits=" << *s;
      if (auto e = test.getNumber("end"))
        llvm::errs() << " end-err-bits=" << *e;
      llvm::errs() << "\n";

      StringRef bestExpr = test.getString("output").value();
      if (bestExpr == "#f") {
        llvm::errs() << "[poseidon] Herbie core \"" << testName
                     << "\": no output expression (status " << status
                     << ") -- no algebraic candidate from this core.\n";
        size_t noOutput;
        if (!testName.getAsInteger(10, noOutput) && noOutput < COs.size())
          COs[noOutput].herbieStatus = status.str();
        continue;
      }

      StringRef ID = test.getString("name").value();
      size_t index = std::stoul(ID.str());
      if (index >= COs.size()) {
        llvm::errs() << "Invalid CO index: " << index << "\n";
        continue;
      }

      RewriteUnit &CO = COs[index];
      CO.herbieAnswered = true;
      CO.herbieStatus = status.str();
      auto &seenExprSet = seenExprs[index];

      double bits = test.getNumber("bits").value();
      CO.herbieBits = bits;
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
        bestCandidate.ordinal = CO.candidates.size();
        CO.candidates.push_back(bestCandidate);
      }

      json::Array &alternatives = *costAccuracy[2].getAsArray();

      for (size_t j = 0; j < alternatives.size(); ++j) {
        json::Array &entry = *alternatives[j].getAsArray();
        if (entry.size() < 3)
          continue;
        StringRef expr = entry[2].getAsString().value();

        if (seenExprSet.count(expr.str()) != 0) {
          continue;
        }
        seenExprSet.insert(expr.str());

        double cost = entry[0].getAsNumber().value() / initialCostVal;
        double accuracy = 1.0 - entry[1].getAsNumber().value() / bits;

        RewriteCandidate candidate(cost, accuracy, expr.str());
        candidate.ordinal = CO.candidates.size();
        CO.candidates.push_back(candidate);
      }
    }
    return true;
  };

  return processHerbieOutput(Run.ResultsJson);
}

std::string getPrecondition(
    const Options &O, const SmallSet<std::string, 8> &args,
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

    if (O.Herbie.HerbiePreFloorBits > 0 && lower < 0 && upper > 0 &&
        std::isfinite(lower) && std::isfinite(upper)) {
      double floorMag = std::ldexp(std::max(-lower, upper),
                                   -(int)O.Herbie.HerbiePreFloorBits);
      std::ostringstream floorStr;
      floorStr << std::setprecision(std::numeric_limits<double>::max_digits10)
               << std::scientific << floorMag;
      preconditions += " (or (<= " + lowerStr.str() + " " + arg + " -" +
                       floorStr.str() + ") (<= " + floorStr.str() + " " + arg +
                       " " + upperStr.str() + "))";
      continue;
    }

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

namespace {

using RegimeArm = std::pair<const FPNode *, bool>;

void collectIfNodes(const FPNode *node, SmallPtrSetImpl<const FPNode *> &seen,
                    SmallVectorImpl<const FPNode *> &ifs) {
  if (!seen.insert(node).second)
    return;
  if (node->op == "if")
    ifs.push_back(node);
  for (const auto &operand : node->operands)
    collectIfNodes(operand.get(), seen, ifs);
}

// The arms the candidate actually evaluates at `point`, following each `if` the
// way the materialised select resolves it.
void collectTakenArms(const FPNode *node,
                      const MapVector<Value *, double> &point,
                      SmallPtrSetImpl<const FPNode *> &seen,
                      SmallVectorImpl<RegimeArm> &arms) {
  if (!seen.insert(node).second)
    return;
  if (node->op != "if") {
    for (const auto &operand : node->operands)
      collectTakenArms(operand.get(), point, seen, arms);
    return;
  }
  SmallVector<double, 1> cond;
  getFPValues({node->operands[0].get()}, point, cond);
  bool taken = cond[0] == 1.0;
  arms.push_back({node, taken});
  collectTakenArms(node->operands[0].get(), point, seen, arms);
  collectTakenArms(node->operands[taken ? 1 : 2].get(), point, seen, arms);
}

SmallVector<RegimeArm, 4> takenArms(const FPNode *root,
                                    const MapVector<Value *, double> &point) {
  SmallPtrSet<const FPNode *, 32> seen;
  SmallVector<RegimeArm, 4> arms;
  collectTakenArms(root, point, seen, arms);
  return arms;
}

struct ArmStats {
  double candSum = 0.0;
  double origSum = 0.0;
  unsigned count = 0;
};

} // namespace

namespace {

// One round of the regime-arm search. A round's points and the original's
// gold value and error at each depend only on the root and the round's seed, so
// every regime-split candidate of the root shares them.
struct ArmRound {
  SmallVector<MapVector<Value *, double>, 4> points;
  SmallVector<double, 4> gold, origError;
  SmallVector<bool, 4> priced;
};

struct RootSamples {
  Value *root = nullptr;
  double grad = 0.0;
  SmallVector<double, 4> goldVals, origErrors, origVals;
  double origCost = 0.0;
  mutable std::vector<ArmRound> armRounds;
};

bool sampleRoot(
    const Options &O, RootSamples &R,
    ArrayRef<MapVector<Value *, double>> sampledPoints,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap) {
  R.goldVals.resize(O.Model.NumSamples);
  R.origErrors.resize(O.Model.NumSamples);
  R.origVals.resize(O.Model.NumSamples);
  double sum = 0.0;
  unsigned count = 0;
  for (const auto &pair : enumerate(sampledPoints)) {
    std::shared_ptr<FPNode> node = valueToNodeMap[R.root];
    SmallVector<double, 1> results;
    getMPFRValues({node.get()}, pair.value(), results, true, 53);
    double goldVal = results[0];
    R.goldVals[pair.index()] = goldVal;

    getFPValues({node.get()}, pair.value(), results);
    double realVal = results[0];
    double error = sampleError(goldVal, realVal);
    R.origErrors[pair.index()] = error;
    R.origVals[pair.index()] = realVal;
    if (!std::isnan(error)) {
      sum += error;
      ++count;
    }
  }
  if (count == 0)
    return false;
  R.origCost = sum / count;
  return true;
}

std::optional<double>
priceRoot(const Options &O, const RootSamples &R,
          std::shared_ptr<FPNode> parsedNode, const std::string &exprForLog,
          const Subgraph &subgraph,
          ArrayRef<MapVector<Value *, double>> sampledPoints,
          std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
          std::unordered_map<std::string, Value *> &symbolToValueMap,
          std::string &why) {
  bool discardCandidate = false;
  double candCost = 0.0;

  SmallVector<const FPNode *, 2> ifNodes;
  if (O.Model.MinArmSamples > 0) {
    SmallPtrSet<const FPNode *, 32> seen;
    collectIfNodes(parsedNode.get(), seen, ifNodes);
  }
  std::map<RegimeArm, ArmStats> armStats;
  for (const FPNode *ifNode : ifNodes) {
    armStats[{ifNode, true}];
    armStats[{ifNode, false}];
  }

  double sum = 0.0;
  unsigned count = 0;
  unsigned broken = 0;
  for (const auto &pair : enumerate(sampledPoints)) {
    SmallVector<double, 1> results;
    getFPValues({parsedNode.get()}, pair.value(), results);
    double realVal = results[0];
    double goldVal = R.goldVals[pair.index()];
    if (std::isfinite(goldVal) && goldVal != 0.0) {
      double origRel =
          std::fabs(R.origVals[pair.index()] - goldVal) / std::fabs(goldVal);
      double candRel = std::fabs(realVal - goldVal) / std::fabs(goldVal);
      if (origRel < 1.0 && !(candRel < 1.0))
        ++broken;
    }

    if (std::isfinite(goldVal) && !std::isfinite(realVal)) {
      discardCandidate = true;
      why = "it returns a non-finite value at a sampled input where the "
            "exact result is finite";
      break;
    }

    double error = sampleError(goldVal, realVal);
    if (!std::isnan(error)) {
      sum += error;
      ++count;
      double origError = R.origErrors[pair.index()];
      if (!ifNodes.empty() && !std::isnan(origError))
        for (const RegimeArm &arm : takenArms(parsedNode.get(), pair.value())) {
          ArmStats &st = armStats[arm];
          st.candSum += error;
          st.origSum += origError;
          ++st.count;
        }
    }
  }
  if (!discardCandidate) {
    if (count == 0) {
      discardCandidate = true;
      why = "every sample of it scored NaN";
    } else {
      candCost = sum / count;
    }
  }
  if (!discardCandidate && O.Model.MaxBrokenShare >= 0.0 &&
      broken > O.Model.MaxBrokenShare * sampledPoints.size()) {
    if (O.Diagnostics.Print)
      llvm::errs() << "[poseidon] dropping candidate: relative error >= 1 on "
                   << broken << " of " << sampledPoints.size()
                   << " samples: " << exprForLog.substr(0, 120) << "\n";
    discardCandidate = true;
    why = formatv("its relative error is 1 or more on {0} of {1} samples",
                  broken, sampledPoints.size())
              .str();
  }

  // Uniform marginal sampling weights each regime by its share of the
  // profiled box, which need not be its share of the workload: correlated
  // inputs can put most of the run in an arm the box almost never reaches.
  // Price every arm on its own samples and charge the worst arm's excess
  // over the original.
  if (!discardCandidate && !ifNodes.empty()) {
    auto starved = [&] {
      for (const auto &kv : armStats)
        if (kv.second.count < O.Model.MinArmSamples)
          return true;
      return false;
    };
    size_t budget =
        static_cast<size_t>(O.Model.ArmSearchFactor) * O.Model.NumSamples;
    size_t drawn = 0;
    unsigned round = 0;
    std::shared_ptr<FPNode> origNode = valueToNodeMap[R.root];
    while (starved() && drawn < budget) {
      if (R.armRounds.size() <= round) {
        ArmRound fresh;
        getSampledPoints(O, subgraph.inputs.getArrayRef(), valueToNodeMap,
                         symbolToValueMap, fresh.points, nullptr,
                         O.Model.NumSamples, round + 1);
        fresh.gold.resize(fresh.points.size());
        fresh.origError.resize(fresh.points.size());
        fresh.priced.resize(fresh.points.size(), false);
        R.armRounds.push_back(std::move(fresh));
      }
      ArmRound &rd = R.armRounds[round++];
      if (rd.points.empty())
        break;
      drawn += rd.points.size();
      for (size_t p = 0; p < rd.points.size(); ++p) {
        const auto &point = rd.points[p];
        SmallVector<RegimeArm, 4> arms = takenArms(parsedNode.get(), point);
        bool wanted = false;
        for (const RegimeArm &arm : arms)
          if (armStats[arm].count < O.Model.MinArmSamples)
            wanted = true;
        if (!wanted)
          continue;
        SmallVector<double, 1> r;
        if (!rd.priced[p]) {
          getMPFRValues({origNode.get()}, point, r, true, 53);
          rd.gold[p] = r[0];
          getFPValues({origNode.get()}, point, r);
          rd.origError[p] = sampleError(rd.gold[p], r[0]);
          rd.priced[p] = true;
        }
        double goldVal = rd.gold[p];
        double origError = rd.origError[p];
        getFPValues({parsedNode.get()}, point, r);
        double error = sampleError(goldVal, r[0]);
        if (std::isnan(error) || std::isnan(origError))
          continue;
        for (const RegimeArm &arm : arms) {
          ArmStats &st = armStats[arm];
          if (st.count >= O.Model.MinArmSamples)
            continue;
          st.candSum += error;
          st.origSum += origError;
          ++st.count;
        }
      }
    }

    double worst = candCost;
    std::string armReport;
    raw_string_ostream os(armReport);
    for (const FPNode *ifNode : ifNodes)
      for (bool taken : {false, true}) {
        const ArmStats &st = armStats[{ifNode, taken}];
        os << " " << (taken ? "then" : "else") << ":" << st.count;
        if (st.count)
          os << ":" << format("%.3e", st.candSum / st.count) << "/"
             << format("%.3e", st.origSum / st.count);
        if (st.count < O.Model.MinArmSamples) {
          discardCandidate = true;
          why = formatv("a regime-split arm is reached by fewer than {0} "
                        "samples",
                        (unsigned)O.Model.MinArmSamples)
                    .str();
          continue;
        }
        worst =
            std::max(worst, R.origCost + (st.candSum - st.origSum) / st.count);
      }
    if (O.Diagnostics.Print)
      llvm::errs() << "[poseidon] regime-split candidate: arms (samples:cand/"
                      "orig mean error)"
                   << os.str() << "; " << drawn << " extra draws; box cost "
                   << format("%.3e", candCost) << " -> "
                   << (discardCandidate ? std::string("dropped (unpriced arm)")
                                        : formatv("{0:e}", worst).str())
                   << "; " << exprForLog.substr(0, 120) << "\n";
    if (!discardCandidate)
      candCost = worst;
  }
  if (discardCandidate)
    return std::nullopt;
  return candCost;
}

void collectInputs(const FPNode *node, const SetVector<Value *> &inputs,
                   SmallPtrSetImpl<const FPNode *> &seen,
                   SmallPtrSetImpl<Value *> &used) {
  if (!seen.insert(node).second)
    return;
  if (auto *LL = dyn_cast<FPLLValue>(node)) {
    used.insert(LL->value);
    if (inputs.contains(LL->value) && !inlinedOps(*LL, inputs))
      return;
  }
  for (const auto &operand : node->operands)
    collectInputs(operand.get(), inputs, seen, used);
}

} // namespace

void setUnifiedAccuracyCost(
    const Options &O, RewriteUnit &CO,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap) {

  CancellationPlan cancelPlan =
      buildCancellationPlan(O, *CO.subgraph, valueToNodeMap);
  SmallVector<MapVector<Value *, double>, 4> sampledPoints;

  getSampledPoints(O, CO.subgraph->inputs.getArrayRef(), valueToNodeMap,
                   symbolToValueMap, sampledPoints, &cancelPlan);

  // No usable sample means the output cannot be priced: drop its candidates and
  // keep the original body, without inventing a bound or an accuracy number.
  auto unpriceable = [&](const char *why) {
    CO.unpricedReason = why;
    llvm::errs() << "[poseidon] subgraph output cannot be priced (" << why
                 << "); discarding its " << CO.candidates.size()
                 << " Herbie candidate(s) and keeping the original body. "
                    "A rewrite may still exist here: this is an un-priced "
                    "site, not a rejected one.\n";
    CO.candidates.clear();
    CO.initialAccCost = 0.0;
  };
  if (sampledPoints.empty()) {
    unpriceable("the profiled input box is empty or unbounded");
    return;
  }

  SmallVector<RootSamples, 1> rootSamples(CO.roots.size());
  double initialAccCost = 0.0;
  for (auto [R, root, grad] : llvm::zip(rootSamples, CO.roots, CO.rootGrads)) {
    R.root = root;
    R.grad = grad;
    if (!sampleRoot(O, R, sampledPoints, valueToNodeMap)) {
      unpriceable("every sample of the original expression scored NaN");
      return;
    }
    initialAccCost += R.origCost * std::fabs(R.grad);
  }
  CO.initialAccCost = initialAccCost;
  if (!std::isfinite(CO.initialAccCost)) {
    // A non-finite baseline makes every candidate delta nan and the DP would
    // silently fall back to the no-op; drop the candidates and say so instead.
    for (const RootSamples &R : rootSamples)
      llvm::errs() << "  origCost = " << R.origCost << ", grad = " << R.grad
                   << "\n";
    unpriceable("the baseline accuracy cost is not finite; the site's profile "
                "is degenerate (a zero or null-space output shadow gives a "
                "gradient that cannot weight anything)");
    return;
  }

  // An element is spliced in at its own output, so it may only read the values
  // that output's original expression read: inputs, and what an input spelled
  // as its definition is computed from.
  SmallVector<SmallPtrSet<Value *, 8>, 1> rootInputs;
  for (Value *root : CO.roots) {
    SmallPtrSet<const FPNode *, 32> seen;
    SmallPtrSet<Value *, 8> used;
    collectInputs(valueToNodeMap[root].get(), CO.subgraph->inputs, seen, used);
    rootInputs.push_back(std::move(used));
  }

  SmallVector<RewriteCandidate, 4> newCandidates;
  for (auto &candidate : CO.candidates) {
    std::shared_ptr<FPNode> parsedNode =
        parseHerbieExpr(candidate.expr, valueToNodeMap, symbolToValueMap,
                        nullptr, &CO.subgraph->inputs);
    SmallVector<std::shared_ptr<FPNode>, 1> elements;
    if (CO.isArray()) {
      if (parsedNode->op != "array" ||
          parsedNode->operands.size() != CO.roots.size()) {
        llvm::errs() << "[poseidon] dropping array candidate with "
                     << parsedNode->operands.size() << " element(s) for "
                     << CO.roots.size() << " outputs\n";
        CO.dropped.push_back(
            {candidate.expr, "it does not have one element per output"});
        continue;
      }
      elements.append(parsedNode->operands.begin(), parsedNode->operands.end());
    } else {
      elements.push_back(parsedNode);
    }

    bool discardCandidate = false;
    double total = 0.0;
    std::string perElement;
    for (auto [k, element] : llvm::enumerate(elements)) {
      SmallPtrSet<const FPNode *, 32> seen;
      SmallPtrSet<Value *, 8> used;
      collectInputs(element.get(), CO.subgraph->inputs, seen, used);
      if (!llvm::all_of(used,
                        [&](Value *V) { return rootInputs[k].contains(V); })) {
        if (O.Diagnostics.Print)
          llvm::errs() << "[poseidon] dropping candidate: element " << k
                       << " reads a value its output does not\n";
        discardCandidate = true;
        CO.dropped.push_back(
            {candidate.expr, formatv("its element {0} reads a value that "
                                     "output does not",
                                     k)
                                 .str()});
        break;
      }
      std::string why;
      std::optional<double> cost =
          priceRoot(O, rootSamples[k], element, candidate.expr, *CO.subgraph,
                    sampledPoints, valueToNodeMap, symbolToValueMap, why);
      if (!cost) {
        discardCandidate = true;
        CO.dropped.push_back({candidate.expr, why});
        break;
      }
      total += *cost * std::fabs(rootSamples[k].grad);
      if (CO.isArray() && O.Diagnostics.Print)
        perElement +=
            formatv(" {0:e}/{1:e}", *cost * std::fabs(rootSamples[k].grad),
                    rootSamples[k].origCost * std::fabs(rootSamples[k].grad))
                .str();
    }
    if (CO.isArray() && O.Diagnostics.Print && !discardCandidate)
      llvm::errs() << "[poseidon] array candidate per element (cand/orig):"
                   << perElement << "\n";

    if (!discardCandidate) {
      candidate.accuracyCost = total;
      assert(!std::isnan(candidate.accuracyCost));
      newCandidates.push_back(std::move(candidate));
    }
  }
  CO.candidates = std::move(newCandidates);
}

double getCompCost(
    CostModelState &CM, const std::string &expr, Module *M,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap,
    const FastMathFlags &FMF, const SetVector<Value *> *subgraphInputs) {
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

  auto parsedNode = parseHerbieExpr(expr, valueToNodeMap, symbolToValueMap,
                                    nullptr, subgraphInputs);
  // An existing value a node was mapped back to is an argument no symbol names.
  SmallVector<Value *, 2> mappedArgs;
  if (subgraphInputs) {
    SmallVector<FPNode *, 16> stack{parsedNode.get()};
    SmallPtrSet<FPNode *, 16> seen;
    while (!stack.empty()) {
      FPNode *N = stack.pop_back_val();
      if (!seen.insert(N).second)
        continue;
      if (auto *L = dyn_cast<FPLLValue>(N)) {
        if (args.insert(L->value)) {
          argTypes.push_back(L->value->getType());
          mappedArgs.push_back(L->value);
        }
        continue;
      }
      for (auto &op : N->operands)
        stack.push_back(op.get());
    }
  }

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

  // An array candidate returns all of its elements, so none is dead and the
  // subterms they share are simplified into one computation before costing.
  const bool isArray = parsedNode->op == "array";
  Type *FuncRetType = ReturnType;
  if (isArray)
    FuncRetType = StructType::get(
        M->getContext(),
        SmallVector<Type *>(parsedNode->operands.size(), ReturnType));

  FunctionType *FT = FunctionType::get(FuncRetType, argTypes, false);
  Function *tempFunction =
      Function::Create(FT, Function::InternalLinkage, "tempFunc", M);

  ValueToValueMapTy VMap;
  Function::arg_iterator AI = tempFunction->arg_begin();
  for (const auto &argStr : argNames) {
    VMap[symbolToValueMap[argStr]] = &*AI;
    ++AI;
  }
  for (Value *arg : mappedArgs) {
    VMap[arg] = &*AI;
    ++AI;
  }

  BasicBlock *entry =
      BasicBlock::Create(M->getContext(), "entry", tempFunction);

  IRBuilder<> builder(entry);

  builder.setFastMathFlags(FMF);
  if (isArray) {
    SmallVector<Value *, 4> elements = emitSharedElements(
        parsedNode->operands, [&](size_t) -> IRBuilder<> & { return builder; },
        &VMap, nullptr);
    Value *Agg = UndefValue::get(FuncRetType);
    for (auto [i, V] : llvm::enumerate(elements)) {
      if (V->getType() != ReturnType)
        V = builder.CreateFPCast(V, ReturnType);
      Agg = builder.CreateInsertValue(Agg, V, i);
    }
    builder.CreateRet(Agg);
  } else {
    Value *RetVal = parsedNode->getLLValue(builder, &VMap);
    assert(RetVal && "Parsed node did not produce a value");
    if (RetVal->getType() != ReturnType)
      RetVal = builder.CreateFPCast(RetVal, ReturnType);
    builder.CreateRet(RetVal);
  }

  simplifyFunction(*tempFunction, OptimizationLevel::O3);

  double cost = getCompCost(CM, tempFunction);

  tempFunction->eraseFromParent();
  return cost;
}

namespace {

// The outputs' expressions with every subterm used more than once bound by
// let*, so Herbie reads the subgraph as the DAG it is rather than one tree per
// output.
static std::string arrayCoreBody(
    ArrayRef<Instruction *> outputs,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    const SetVector<Value *> &inputs) {
  auto isLeaf = [&](FPNode *N) {
    if (isa<FPConst>(N))
      return true;
    auto *LL = cast<FPLLValue>(N);
    return inputs.contains(LL->value);
  };
  DenseMap<FPNode *, unsigned> uses;
  SmallVector<FPNode *, 32> order;
  SmallPtrSet<FPNode *, 32> visited;
  std::function<void(FPNode *)> walk = [&](FPNode *N) {
    if (!visited.insert(N).second || isLeaf(N))
      return;
    for (auto &operand : N->operands) {
      if (!isLeaf(operand.get()))
        ++uses[operand.get()];
      walk(operand.get());
    }
    order.push_back(N);
  };
  for (Instruction *output : outputs) {
    FPNode *N = valueToNodeMap[output].get();
    ++uses[N];
    walk(N);
  }

  DenseMap<FPNode *, std::string> names;
  std::function<std::string(FPNode *)> ref;
  auto define = [&](FPNode *N) {
    std::string expr = "(" + (N->op == "neg" ? std::string("-") : N->op);
    for (auto &operand : N->operands)
      expr += " " + ref(operand.get());
    return expr + ")";
  };
  ref = [&](FPNode *N) -> std::string {
    if (isLeaf(N))
      return N->toFullExpression(valueToNodeMap, inputs);
    auto It = names.find(N);
    if (It != names.end())
      return It->second;
    return define(N);
  };

  std::string bindings;
  for (FPNode *N : order) {
    if (uses[N] < 2)
      continue;
    std::string def = define(N);
    std::string name = "t" + std::to_string(names.size() + 1);
    if (!bindings.empty())
      bindings += " ";
    bindings += "(" + name + " " + def + ")";
    names[N] = name;
  }
  std::string body = "(array";
  for (Instruction *output : outputs)
    body += " " + ref(valueToNodeMap[output].get());
  body += ")";
  return bindings.empty() ? body : "(let* (" + bindings + ") " + body + ")";
}

} // namespace

void proposeHerbieCandidates(
    Session &S, Function &F, Subgraph &subgraph, int subgraphCounter,
    int &symbolCounter,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap,
    SmallVectorImpl<RewriteUnit> &COs) {
  const Options &O = S.Opts;
  auto getNextSymbol = [&symbolCounter]() -> std::string {
    return "v" + std::to_string(symbolCounter++);
  };

  for (const auto &input : subgraph.inputs) {
    auto node = valueToNodeMap[input];
    if (node->op == "__const") {
      continue;
    }

    if (!node->hasSymbol()) {
      node->symbol = getNextSymbol();
    }
    symbolToValueMap[node->symbol] = input;
    if (O.Diagnostics.Print)
      llvm::errs() << "assigning symbol: " << node->symbol << " to " << *input
                   << "\n";
  }

  std::vector<std::string> herbieInputs;
  std::vector<RewriteUnit> newCOs;

  struct Eligible {
    Instruction *output;
    std::string expr;
    double grad;
    uint64_t executions;
  };
  SmallVector<Eligible, 8> eligible;

  auto emitPerOutput = [&](const Eligible &E, ArrayRef<const char *> precs,
                           std::vector<std::string> &inputs,
                           std::vector<RewriteUnit> &cos) {
    SmallSet<std::string, 8> args;
    getUniqueArgs(E.expr, args);

    std::string precondition =
        getPrecondition(O, args, valueToNodeMap, symbolToValueMap);

    std::string argStr;
    for (const auto &arg : args) {
      if (!argStr.empty())
        argStr += " ";
      argStr += arg;
    }

    for (const char *prec : precs) {
      std::string properties = ":herbie-conversions ([binary64 binary32])";
      properties += std::string(" :precision ") + prec;
      properties += " :pre " + precondition;

      RewriteUnit CO(S.CostModel, subgraph, E.output, E.expr, E.grad,
                     E.executions);
      CO.herbiePrecision = prec;
      properties += " :name \"" + std::to_string(cos.size()) + "\"";

      std::string herbieInput =
          "(FPCore (" + argStr + ") " + properties + " " + E.expr + ")";
      if (O.Diagnostics.Print)
        llvm::errs() << "Herbie input:\n" << herbieInput << "\n";

      inputs.push_back(herbieInput);
      cos.push_back(CO);
    }
  };

  assert(subgraph.outputs.size() > 0 && "No outputs found for subgraph");
  for (auto &output : subgraph.outputs) {
    double grad = valueToNodeMap[output]->grad;
    uint64_t executions = valueToNodeMap[output]->executions;

    if (grad == 0.) {
      llvm::errs() << "Skipping zero gradient instruction: " << *output << "\n";
      continue;
    }

    std::string expr = valueToNodeMap[output]->toFullExpression(
        valueToNodeMap, subgraph.inputs);

    if (expr.length() > O.Candidates.MaxExprLength) {
      llvm::errs() << "WARNING: Skipping Herbie optimization for " << *output
                   << " since expression length " << expr.length()
                   << " exceeds limit of " << O.Candidates.MaxExprLength
                   << "\n";
      continue;
    }

    auto parenCount = std::count(expr.begin(), expr.end(), '(');
    assert(parenCount > 0);
    if (parenCount == 1) {
      if (O.Diagnostics.Print)
        llvm::errs() << "Skipping Herbie for simple expression: " << expr
                     << "\n";
      continue;
    }

    Eligible E{output, expr, grad, executions};
    if (O.Herbie.HerbieArrays)
      eligible.push_back(std::move(E));
    else
      emitPerOutput(E, {"binary64", "binary32"}, herbieInputs, newCOs);
  }

  if (O.Herbie.HerbieArrays &&
      (!StringRef(O.Herbie.HerbieArrayGrouping).equals_insensitive("inputs") &&
       !StringRef(O.Herbie.HerbieArrayGrouping)
            .equals_insensitive("executions")))
    report_fatal_error("Poseidon: -poseidon-herbie-array-grouping must be "
                       "'inputs' or 'executions'");
  if (O.Herbie.HerbieArrays && O.Herbie.HerbieArrayWeights != "sensitivity" &&
      O.Herbie.HerbieArrayWeights != "gradient" &&
      O.Herbie.HerbieArrayWeights != "none")
    report_fatal_error("Poseidon: -poseidon-herbie-array-weights must be "
                       "'sensitivity', 'gradient' or 'none'");

  // Outputs that read the same inputs (hence the same precondition) and
  // execute equally often form one array core; the joint rewrite is priced
  // at that count.
  auto groupKey = [&](const Eligible &E) {
    std::string key = std::to_string(E.executions);
    if (O.Herbie.HerbieArrayGrouping == "inputs") {
      SmallSet<std::string, 8> args;
      getUniqueArgs(E.expr, args);
      std::vector<std::string> sorted(args.begin(), args.end());
      llvm::sort(sorted);
      for (const auto &arg : sorted)
        key += " " + arg;
    }
    return key;
  };
  SmallVector<std::pair<std::string, SmallVector<const Eligible *, 8>>, 2>
      groups;
  for (const Eligible &E : eligible) {
    std::string key = groupKey(E);
    auto It =
        llvm::find_if(groups, [&](const auto &G) { return G.first == key; });
    if (It == groups.end())
      groups.emplace_back(key, SmallVector<const Eligible *, 8>{&E});
    else
      It->second.push_back(&E);
  }

  std::vector<std::string> arrayInputs;
  std::vector<RewriteUnit> arrayCOs;
  SmallVector<const SmallVector<const Eligible *, 8> *, 4> arrayGroups;
  const char *precs[] = {"binary64", "binary32"};
  for (const auto &[key, G] : groups) {
    if (G.size() < 2)
      continue;
    SmallVector<Instruction *, 8> outputs;
    SmallVector<Value *, 8> roots;
    SmallVector<double, 8> grads;
    for (const Eligible *E : G) {
      outputs.push_back(E->output);
      roots.push_back(E->output);
      grads.push_back(E->grad);
    }
    std::string body = arrayCoreBody(outputs, valueToNodeMap, subgraph.inputs);
    SmallSet<std::string, 8> args;
    getUniqueArgs(body, args);
    std::string precondition =
        getPrecondition(O, args, valueToNodeMap, symbolToValueMap);
    std::string argStr;
    for (const auto &arg : args) {
      if (!argStr.empty())
        argStr += " ";
      argStr += arg;
    }
    std::string weights;
    if (O.Herbie.HerbieArrayWeights != "none") {
      SmallVector<double, 8> sens;
      for (Instruction *output : outputs) {
        FPNode *N = valueToNodeMap[output].get();
        if (O.Herbie.HerbieArrayWeights == "gradient") {
          double lo = N->getLowerBound(), hi = N->getUpperBound();
          double meanMag = lo >= 0 || hi <= 0
                               ? (std::fabs(lo) + std::fabs(hi)) / 2
                               : (lo * lo + hi * hi) / (2 * (hi - lo));
          sens.push_back(std::fabs(N->grad) * meanMag);
        } else {
          sens.push_back(N->sens);
        }
      }
      double total = 0.0;
      for (double v : sens)
        total += v;
      if (total > 0.0 && std::isfinite(total) &&
          llvm::all_of(sens, [](double v) { return v >= 0.0; })) {
        weights = " :herbie-weights (";
        for (auto [k, v] : llvm::enumerate(sens)) {
          char buf[32];
          snprintf(buf, sizeof(buf), "%.6e", v * sens.size() / total);
          weights += (k ? " " : "") + std::string(buf);
        }
        weights += ")";
      } else {
        llvm::errs() << "[poseidon] an array core of subgraph "
                     << subgraphCounter
                     << " is sent unweighted: its outputs' profiled "
                        "sensitivities do not sum to a positive value\n";
      }
    }
    for (const char *prec : precs) {
      std::string herbieInput =
          "(FPCore (" + argStr +
          ") :herbie-conversions ([binary64 binary32]) :precision " + prec +
          " :pre " + precondition + weights + " :name \"" +
          std::to_string(arrayCOs.size()) + "\" " + body + ")";
      if (O.Diagnostics.Print)
        llvm::errs() << "Herbie array input:\n" << herbieInput << "\n";
      arrayInputs.push_back(herbieInput);
      arrayCOs.emplace_back(S.CostModel, subgraph, ArrayRef<Value *>(roots),
                            grads, body, G.front()->executions);
      arrayCOs.back().herbiePrecision = prec;
      arrayGroups.push_back(&G);
    }
  }

  // The outputs no array covers are searched in the same invocation as the
  // arrays; per-output cores stand in for an array core that returned
  // nothing, in a second one.
  auto inArray = [&](const Eligible &E) {
    return llvm::any_of(
        arrayGroups, [&](const auto *G) { return llvm::is_contained(*G, &E); });
  };
  const size_t numArrayCores = arrayCOs.size();
  for (const Eligible &E : eligible)
    if (!inArray(E))
      emitPerOutput(E, precs, numArrayCores ? arrayInputs : herbieInputs,
                    numArrayCores ? arrayCOs : newCOs);
  if (numArrayCores && !improveViaHerbie(S, arrayInputs, arrayCOs,
                                         subgraphCounter, F.getName())) {
    if (O.Diagnostics.Print)
      llvm::errs() << "Failed to optimize expressions using Herbie!\n";
  }
  for (const Eligible &E : eligible) {
    SmallVector<const char *, 2> sendAt;
    for (size_t k = 0; k < numArrayCores; ++k)
      if (llvm::is_contained(*arrayGroups[k], &E) &&
          !arrayCOs[k].herbieAnswered)
        sendAt.push_back(precs[k % 2]);
    if (sendAt.empty())
      continue;
    llvm::errs() << "[poseidon] Herbie subgraph " << subgraphCounter
                 << ": array core returned nothing for " << *E.output
                 << "; sending it separately\n";
    emitPerOutput(E, sendAt, herbieInputs, newCOs);
  }
  COs.insert(COs.end(), arrayCOs.begin(), arrayCOs.end());

  if (!herbieInputs.empty()) {
    if (!improveViaHerbie(S, herbieInputs, newCOs, subgraphCounter,
                          F.getName())) {
      if (O.Diagnostics.Print)
        llvm::errs() << "Failed to optimize expressions using Herbie!\n";
    }

    COs.insert(COs.end(), newCOs.begin(), newCOs.end());
  }
}

void RewriteUnit::apply(
    size_t candidateIndex,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap, bool print) {
  if (candidateIndex >= candidates.size()) {
    if (print)
      llvm::errs() << "RewriteUnit::apply: candidateIndex " << candidateIndex
                   << " out of range (size " << candidates.size()
                   << "), skipping\n";
    return;
  }
  for (Value *root : roots) {
    auto *oldInst = dyn_cast<Instruction>(root);
    if (!oldInst || !oldInst->getParent()) {
      if (print)
        llvm::errs() << "RewriteUnit::apply: oldOutput "
                     << "is not in a basic block (already erased?), skipping\n";
      return;
    }
  }

  auto parsedNode =
      parseHerbieExpr(candidates[candidateIndex].expr, valueToNodeMap,
                      symbolToValueMap, nullptr, &subgraph->inputs);
  SmallVector<std::shared_ptr<FPNode>, 1> elements;
  if (isArray()) {
    if (parsedNode->op != "array" ||
        parsedNode->operands.size() != roots.size())
      report_fatal_error("Poseidon: an array rewrite does not have one element "
                         "per output of its subgraph");
    elements.append(parsedNode->operands.begin(), parsedNode->operands.end());
  } else {
    elements.push_back(parsedNode);
  }

  SmallVector<Value *, 4> newOutputs;
  if (isArray()) {
    DominatorTree DT(*cast<Instruction>(oldOutput)->getFunction());
    std::optional<IRBuilder<>> builder;
    newOutputs = emitSharedElements(
        elements,
        [&](size_t k) -> IRBuilder<> & {
          auto *oldInst = cast<Instruction>(roots[k]);
          builder.emplace(oldInst->getParent(),
                          ++BasicBlock::iterator(oldInst));
          builder->setFastMathFlags(oldInst->getFastMathFlags());
          return *builder;
        },
        nullptr, &DT);
  }

  for (auto [k, root, element] : llvm::enumerate(roots, elements)) {
    auto *oldInst = cast<Instruction>(root);
    IRBuilder<> builder(oldInst->getParent(), ++BasicBlock::iterator(oldInst));
    builder.setFastMathFlags(oldInst->getFastMathFlags());

    Value *newOutput = isArray() ? newOutputs[k] : element->getLLValue(builder);
    assert(newOutput && "Failed to get value from parsed node");
    if (isArray() && isa<Instruction>(newOutput))
      builder.SetInsertPoint(cast<Instruction>(newOutput)->getNextNode());

    if (newOutput->getType() != root->getType())
      newOutput =
          builder.CreateFPCast(newOutput, root->getType(), "herbie.fpcast");

    root->replaceAllUsesWith(newOutput);
    symbolToValueMap[valueToNodeMap[root]->symbol] = newOutput;
    valueToNodeMap[newOutput] = std::make_shared<FPLLValue>(
        newOutput, "__no", valueToNodeMap[root]->dtype);
  }

  for (auto *I : erasableInsts) {
    if (!I->use_empty())
      I->replaceAllUsesWith(UndefValue::get(I->getType()));
    I->eraseFromParent();
    subgraph->operations.remove(I); // Avoid a second removal
    cast<FPLLValue>(valueToNodeMap[I].get())->value = nullptr;
  }

  subgraph->outputs_rewritten += roots.size();
}

bool RewriteUnit::sharesRoot(const RewriteUnit &other) const {
  for (Value *root : roots)
    if (llvm::is_contained(other.roots, root))
      return true;
  return false;
}

// Lower is better
double RewriteUnit::getCompCostDelta(CostModelState &CM,
                                     size_t candidateIndex) {
  double erasableCost = 0.0;

  for (auto *I : erasableInsts) {
    erasableCost += getInstructionCompCost(CM, I);
  }

  return (candidates[candidateIndex].CompCost - erasableCost) *
         (double)executions;
}

void RewriteUnit::findErasableInstructions() {
  SmallPtrSet<Value *, 8> visited;
  SmallPtrSet<Instruction *, 8> exprInsts;
  for (Value *root : roots)
    collectExprInsts(root, subgraph->inputs, exprInsts, visited);
  visited.clear();

  // Seeded in IR order, not in the order the pointers happened to hash: the
  // topological sort below is stable, so its input order decides the order the
  // report lists these instructions in and the order their costs are summed.
  SetVector<Instruction *> instsToProcess;
  for (Instruction &I :
       instructions(*cast<Instruction>(oldOutput)->getFunction()))
    if (exprInsts.contains(&I))
      instsToProcess.insert(&I);

  SmallVector<Instruction *, 8> instsToProcessSorted;
  reverseTopoSort(instsToProcess, instsToProcessSorted);

  erasableInsts.clear();
  for (Value *root : roots)
    erasableInsts.insert(cast<Instruction>(root));

  for (auto *I : instsToProcessSorted) {
    if (erasableInsts.contains(I))
      continue;

    bool usedOutside = false;
    for (auto user : I->users()) {
      if (auto *userI = dyn_cast<Instruction>(user)) {
        if (erasableInsts.contains(userI)) {
          continue;
        }
      }
      usedOutside = true;
      break;
    }

    if (!usedOutside) {
      erasableInsts.insert(I);
    }
  }
}

// Lower is better
double RewriteUnit::getAccCostDelta(size_t candidateIndex) {
  return candidates[candidateIndex].accuracyCost - initialAccCost;
}

double RewriteUnit::costDelta(size_t i, ArrayRef<SolutionStep>) {
  return getCompCostDelta(session().CostModel, i);
}

double RewriteUnit::errorDelta(size_t i, ArrayRef<SolutionStep>) {
  return getAccCostDelta(i);
}

bool RewriteUnit::sameRewriteTargetAs(const OptimizationUnit &earlier) const {
  const auto *other = dyn_cast<RewriteUnit>(&earlier);
  return other && other->subgraph == subgraph &&
         other->roots.size() == roots.size() &&
         llvm::all_of(other->roots,
                      [&](Value *r) { return llvm::is_contained(roots, r); });
}

const SetVector<Instruction *> *RewriteUnit::footprint() const {
  return subgraph ? &subgraph->operations : nullptr;
}

std::string RewriteUnit::choiceText(size_t i) const {
  const RewriteCandidate &cand = candidates[i];
  return expr + " --(" + std::to_string(cand.ordinal) + ")-> " + cand.expr;
}

json::Object RewriteUnit::stepJSON(size_t i) {
  json::Object stepObj;
  stepObj["type"] = "rewrite";
  stepObj["original_expr"] = expr;
  auto &cand = candidates[i];
  stepObj["rewritten_expr"] = cand.expr;
  stepObj["herbie_cost"] = jsonFloat(cand.herbieCost);
  stepObj["herbie_accuracy"] = jsonFloat(cand.herbieAccuracy);
  stepObj["initial_herbie_cost"] = jsonFloat(initialHerbieCost);
  stepObj["initial_herbie_accuracy"] = jsonFloat(initialHerbieAccuracy);
  stepObj["computation_cost_delta"] = getCompCostDelta(session().CostModel, i);
  stepObj["accuracy_cost_delta"] = jsonFloat(getAccCostDelta(i));
  stepObj["gradient"] = jsonFloat(grad);
  stepObj["executions"] = static_cast<int64_t>(executions);

  SmallVector<Instruction *, 8> insts;
  if (auto *I = dyn_cast<Instruction>(oldOutput))
    insts.push_back(I);
  for (auto *I : erasableInsts)
    insts.push_back(I);
  stepObj["source_locations"] = getSourceLocationsJSON(insts);

  json::Array affectedIR;
  if (auto *I = dyn_cast<Instruction>(oldOutput))
    affectedIR.push_back(getValueStr(I));
  for (auto *I : erasableInsts) {
    if (I != oldOutput)
      affectedIR.push_back(getValueStr(I));
  }
  stepObj["affected_instructions"] = std::move(affectedIR);
  return stepObj;
}

void RewriteUnit::writeReportStep(raw_ostream &OS, size_t i, StringRef pad) {
  auto &cand = candidates[i];
  OS << pad << "[Rewrite] " << expr << "  -->  " << cand.expr << "\n";
  if (!std::isnan(initialHerbieAccuracy) && !std::isnan(cand.herbieAccuracy))
    OS << pad << "  Herbie accuracy: " << initialHerbieAccuracy << " -> "
       << cand.herbieAccuracy << " bits\n";
  OS << pad << "  Gradient: " << grad << ", Executions: " << executions << "\n";

  SmallSet<std::string, 4> seen;
  auto printLoc = [&](Instruction *I) {
    auto loc = getSourceLocationStr(I);
    if (!loc.empty() && seen.insert(loc).second)
      OS << pad << "  Source: " << loc << "\n";
  };
  if (auto *I = dyn_cast<Instruction>(oldOutput))
    printLoc(I);
  for (auto *I : erasableInsts)
    printLoc(I);

  OS << pad << "  Affected IR:\n";
  if (auto *I = dyn_cast<Instruction>(oldOutput))
    OS << pad << "    " << *I << "\n";
  for (auto *I : erasableInsts) {
    if (I != oldOutput)
      OS << pad << "    " << *I << "\n";
  }
}

void RewriteUnit::materialize(size_t i) {
  apply(i, graph().valueToNodeMap, graph().symbolToValueMap,
        session().Opts.Diagnostics.Print);
}

void RewriteUnit::price() {
  if (!herbieAnswered)
    return;
  FPGraph &st = graph();
  Module *M = st.F->getParent();
  const FastMathFlags FMF = cast<Instruction>(oldOutput)->getFastMathFlags();
  for (RewriteCandidate &c : candidates)
    c.CompCost = getCompCost(session().CostModel, c.expr, M, st.valueToNodeMap,
                             st.symbolToValueMap, FMF, &subgraph->inputs);
  setUnifiedAccuracyCost(session().Opts, *this, st.valueToNodeMap,
                         st.symbolToValueMap);
}

void RewriteUnit::scaleErrors(double factor) {
  initialAccCost *= factor;
  for (auto &cand : candidates)
    cand.accuracyCost *= factor;
}

void RewriteUnit::printPriced(raw_ostream &OS) {
  OS << "\n################################\n";
  OS << "Initial AccuracyCost: " << initialAccCost << "\n";
  OS << "Initial ComputationCost: " << initialCompCost << "\n";
  OS << "Initial HerbieCost: " << initialHerbieCost << "\n";
  OS << "Initial HerbieAccuracy: " << initialHerbieAccuracy << "\n";
  OS << "Initial Expression: " << expr << "\n";
  OS << "Grad: " << grad << "\n\n";
  OS << "Candidates:\n";
  OS << "Δ AccCost\t\tΔ "
        "CompCost\t\tHerbieCost\t\tAccuracy\t\tExpression\n";
  OS << "--------------------------------\n";
  for (size_t i = 0; i < candidates.size(); ++i) {
    auto &candidate = candidates[i];
    OS << getAccCostDelta(i) << "\t\t";
    // A compile that reads its DP table from the cache prices no candidate.
    if (candidate.CompCost == std::numeric_limits<double>::max())
      OS << "unpriced";
    else
      OS << getCompCostDelta(session().CostModel, i);
    OS << "\t\t" << candidate.herbieCost << "\t\t" << candidate.herbieAccuracy
       << "\t\t" << candidate.expr << "\n";
  }
  OS << "################################\n\n";
}

void RewriteUnit::addReportSummary(json::Array &summary) {
  json::Object co;
  co["original_expr"] = expr;
  co["num_candidates"] = static_cast<int64_t>(candidates.size());
  co["gradient"] = jsonFloat(grad);
  co["executions"] = static_cast<int64_t>(executions);
  co["initial_accuracy_cost"] = jsonFloat(initialAccCost);
  co["initial_computation_cost"] = initialCompCost;
  if (auto *I = dyn_cast<Instruction>(oldOutput)) {
    auto loc = getSourceLocationJSON(I);
    if (!loc.empty())
      co["source_location"] = std::move(loc);
  }
  summary.push_back(std::move(co));
}

void RewriteUnit::writeReportSummary(raw_ostream &textOut) {
  textOut << "Expression #" << indexInKind() << ": " << expr << "\n";
  textOut << "  Gradient: " << grad << ", Executions: " << executions << "\n";
  if (auto *I = dyn_cast<Instruction>(oldOutput)) {
    auto loc = getSourceLocationStr(I);
    if (!loc.empty())
      textOut << "  Source: " << loc << "\n";
  }
  textOut << "  Candidates: " << candidates.size() << "\n";
  for (size_t j = 0; j < candidates.size(); ++j) {
    auto &cand = candidates[j];
    textOut << "    [" << cand.ordinal << "] " << cand.expr;
    if (!std::isnan(cand.herbieAccuracy))
      textOut << "  (accuracy: " << cand.herbieAccuracy << " bits)";
    textOut << "\n";
  }
}

void RewriteUnit::addCuratedRewrites(json::Array &rewrites) {
  for (size_t ci = 0; ci < candidates.size(); ++ci) {
    auto &cand = candidates[ci];
    double accDelta = getAccCostDelta(ci);
    auto compDelta = getCompCostDelta(session().CostModel, ci);
    double compDeltaVal = compDelta;

    if (std::isnan(accDelta))
      continue;

    // comp < 0 means faster, acc < 0 means more accurate
    std::string category;
    double efficiency = 0.0;
    if (compDeltaVal <= 0 && accDelta <= 0) {
      category = "free_win";
      efficiency = std::abs(accDelta) + std::abs(compDeltaVal);
    } else if (compDeltaVal <= 0 && accDelta > 0) {
      category = "speed_for_accuracy";
      efficiency = (accDelta > 1e-30) ? std::abs(compDeltaVal) / accDelta
                                      : std::abs(compDeltaVal);
    } else if (compDeltaVal > 0 && accDelta <= 0) {
      category = "accuracy_for_speed";
      efficiency = (compDeltaVal > 0) ? std::abs(accDelta) / compDeltaVal
                                      : std::abs(accDelta);
    } else {
      continue;
    }

    json::Object rw;
    rw["id"] = candidateId(ci);
    rw["original_expr"] = expr;
    rw["rewritten_expr"] = cand.expr;
    rw["category"] = category;
    rw["efficiency"] = jsonFloat(efficiency);
    rw["computation_cost_delta"] = compDeltaVal;
    rw["accuracy_cost_delta"] = jsonFloat(accDelta);
    rw["gradient"] = jsonFloat(grad);
    rw["executions"] = static_cast<int64_t>(executions);
    rw["herbie_accuracy"] = jsonFloat(cand.herbieAccuracy);
    rw["initial_herbie_accuracy"] = jsonFloat(initialHerbieAccuracy);

    if (auto *I = dyn_cast<Instruction>(oldOutput)) {
      auto loc = getSourceLocationJSON(I);
      if (!loc.empty())
        rw["source_location"] = std::move(loc);
    }

    rewrites.push_back(std::move(rw));
  }
}

} // namespace poseidon
