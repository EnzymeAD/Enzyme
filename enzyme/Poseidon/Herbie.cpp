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
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InstructionCost.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/xxhash.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"

#include "CostModel.h"
#include "Evaluators.h"
#include "Flags.h"
#include "Herbie.h"
#include "Optimize.h"
#include "Sampling.h"
#include "Solvers.h"
#include "Types.h"
#include "Utils.h"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <map>
#include <regex>
#include <string>
#include <unordered_map>

using namespace llvm;

namespace poseidon {

// The Herbie platform this compile searches against: the cost table Herbie
// ranks its rewrite candidates by. It is the device's, so it is derived from
// the resolved cost model, and it is part of the Herbie cache key: a rewrite
// Herbie proposed against another device's costs is that device's answer.
namespace {
struct HerbiePlatform {
  // Identity, and the cache-entry component. "cuda-sm120", "cuda-sm90",
  // "default" (Herbie's own platform, for a host cost model).
  std::string Name;
  // The .rkt handed to `--platform`; empty means pass Name instead.
  std::string Path;
  // Content digest of Path, stamped beside a cache entry so a recalibrated
  // device (same Name, different prices) is reported rather than replayed
  // silently. Empty when Path is empty.
  std::string Digest;
};

std::string readFileDigest(StringRef Path) {
  auto Buf = MemoryBuffer::getFile(Path);
  if (!Buf)
    return std::string();
  return llvm::utohexstr(llvm::xxh3_64bits((*Buf)->getBuffer()));
}

// A Herbie cache entry is stamped with the digest of the binary that produced
// it: version strings do not change between upstream commits, and one
// binary's rewrites must not be replayed as another's.
static std::string herbieBinaryDigest(StringRef Program) {
  static std::map<std::string, std::string> Digests;
  auto It = Digests.find(Program.str());
  if (It == Digests.end())
    It = Digests.emplace(Program.str(), readFileDigest(Program)).first;
  return It->second;
}

const HerbiePlatform &getHerbiePlatform() {
  static HerbiePlatform P;
  static bool Resolved = false;
  if (Resolved)
    return P;
  Resolved = true;

  const std::string &NativeArch = getCostModelNativeArch();
  bool IsCUDA = StringRef(NativeArch).starts_with("sm_");
  if (IsCUDA) {
    std::string Stripped = NativeArch;
    size_t Pos;
    while ((Pos = Stripped.find('_')) != std::string::npos)
      Stripped.erase(Pos, 1);
    P.Name = "cuda-" + Stripped;
  } else {
    P.Name = "default";
  }

  if (!flags::HerbiePlatform.empty()) {
    // A value that looks like a file is the platform itself; anything else is
    // a platform name compiled into the Herbie binary. The name keeps naming
    // the device family either way, and the digest below distinguishes files.
    StringRef V(flags::HerbiePlatform);
    if (V.contains('/') || V.ends_with(".rkt"))
      P.Path = V.str();
    else
      P.Name = V.str();
  } else if (IsCUDA) {
    // Generated next to the CSV it was priced from, by
    // `poseidon-calibrate --only herbie-platform`.
    StringRef Csv(costModelPath());
    P.Path =
        (Csv.ends_with(".csv") ? Csv.drop_back(4) : Csv).str() + ".herbie.rkt";
    if (!sys::fs::exists(P.Path))
      report_fatal_error(
          Twine("Poseidon: the Herbie algebraic search needs the platform "
                "generated from this device's cost model, and ") +
          P.Path +
          " does not exist. Generate it with 'poseidon-calibrate --only "
          "herbie-platform --out " +
          costModelPath() +
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

std::shared_ptr<FPNode> parseHerbieExpr(
    const std::string &expr,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap) {
  std::string trimmedExpr = expr;
  trimmedExpr.erase(0, trimmedExpr.find_first_not_of(" "));
  trimmedExpr.erase(trimmedExpr.find_last_not_of(" ") + 1);

  if (trimmedExpr.front() != '(' && trimmedExpr.front() != '#') {
    if (auto node = valueToNodeMap[symbolToValueMap[trimmedExpr]]) {
      return node;
    }
  }

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

  // An array projection, `(ref.N.array<...> (array e0 e1 ...))` or
  // `(ref.N.array<...> (sincos.fXX x))`: the N-th element is what is wanted.
  if (fullOp.rfind("ref.", 0) == 0) {
    size_t idxEnd = fullOp.find('.', 4);
    unsigned idx = std::stoul(fullOp.substr(4, idxEnd - 4));
    std::string inner = trimmedExpr.substr(endOp + 1);
    inner.erase(0, inner.find_first_not_of(" "));
    inner.erase(inner.find_last_not_of(" ") + 1);
    if (inner.front() == '(' && inner.back() == ')') {
      std::string body = inner.substr(1, inner.size() - 2);
      std::string head = body.substr(0, body.find(' '));
      SmallVector<std::string, 4> elems;
      int depth = 0;
      size_t start = body.find_first_not_of(" ", head.size());
      for (size_t i = start; i < body.size(); ++i) {
        if (body[i] == '(')
          depth++;
        else if (body[i] == ')')
          depth--;
        else if (body[i] == ' ' && depth == 0) {
          elems.push_back(body.substr(start, i - start));
          start = i + 1;
        }
      }
      if (start < body.size())
        elems.push_back(body.substr(start));
      if (head == "array" && idx < elems.size())
        return parseHerbieExpr(elems[idx], valueToNodeMap, symbolToValueMap);
      if (head.rfind("sincos.", 0) == 0 && elems.size() == 1 && idx < 2) {
        auto node =
            std::make_shared<FPNode>(idx == 0 ? "sin" : "cos", head.substr(7));
        node->addOperand(
            parseHerbieExpr(elems[0], valueToNodeMap, symbolToValueMap));
        return node;
      }
    }
    llvm::errs() << "Unexpected array projection: " << trimmedExpr << "\n";
    assert(0 && "Failed to parse Herbie expression");
  }

  size_t pos = fullOp.find('.');
  std::string dtype;
  std::string op;
  if (pos != std::string::npos) {
    op = fullOp.substr(0, pos);
    dtype = fullOp.substr(pos + 1);
    assert(dtype == "f64" || dtype == "f32");
  } else {
    op = fullOp;
  }

  // Herbie's explicit conversions arrive without the ".f32"/".f64" suffix that
  // gives every other operator its dtype; naming the result precision here
  // makes a conversion node an identity operator carrying the destination
  // precision, which every downstream stage rounds and coerces correctly.
  if (dtype.empty()) {
    if (op == "binary64->binary32")
      dtype = "f32";
    else if (op == "binary32->binary64")
      dtype = "f64";
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
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap, int subgraphIdx,
    StringRef funcTag, StringRef cacheKey) {
  std::string Program = flags::HerbieBinary.empty() ? std::string(HERBIE_BINARY)
                                                    : flags::HerbieBinary;
  if (!flags::HerbieBinary.empty() && flags::Print)
    llvm::errs() << "Poseidon: using Herbie binary '" << Program << "'\n";
  const std::string BinaryDigest = herbieBinaryDigest(Program);
  llvm::errs() << "random seed: " << std::to_string(flags::RandomSeed) << "\n";

  // Herbie runs this subgraph's cores across flags::HerbieNumThreads workers,
  // so ceil(cores/threads) rounds of the per-core timeout is what the
  // invocation can legitimately spend; the hard kill sits one margin above
  // that.
  const unsigned subgraphBudget = flags::HerbieSubgraphTimeout;
  unsigned coreTimeout = (unsigned)flags::HerbieTimeout;
  unsigned wallBudget = 0; // 0 = ExecuteAndWait waits forever
  if (subgraphBudget > 0) {
    coreTimeout =
        std::min<unsigned>((unsigned)flags::HerbieTimeout, subgraphBudget);
    unsigned threads = std::max(1u, (unsigned)flags::HerbieNumThreads);
    unsigned rounds = (unsigned)((inputExprs.size() + threads - 1) / threads);
    if (rounds == 0)
      rounds = 1;
    wallBudget = coreTimeout * rounds + 30;
    if (coreTimeout < (unsigned)flags::HerbieTimeout)
      llvm::errs() << "[poseidon] Herbie per-core --timeout clamped from "
                   << flags::HerbieTimeout << " s to " << coreTimeout
                   << " s by -poseidon-herbie-subgraph-timeout.\n";
  }

  SmallVector<std::string> BaseArgs = {
      Program,        "report",
      "--seed",       std::to_string(flags::RandomSeed),
      "--timeout",    std::to_string(coreTimeout),
      "--threads",    std::to_string(flags::HerbieNumThreads),
      "--num-points", std::to_string(flags::HerbieNumPts),
      "--num-iters",  std::to_string(flags::HerbieNumIters),
      "--num-enodes", std::to_string(flags::HerbieNumEnodes)};

  const HerbiePlatform &Platform = getHerbiePlatform();
  // "default" is Herbie's own choice of platform, which it makes when it is
  // not told one; a host cost model leaves it there.
  if (Platform.Name != "default" || !Platform.Path.empty()) {
    BaseArgs.push_back("--platform");
    BaseArgs.push_back(Platform.Path.empty() ? Platform.Name : Platform.Path);
    if (flags::Print)
      llvm::errs() << "Poseidon: using Herbie platform '" << Platform.Name
                   << "'"
                   << (Platform.Path.empty()
                           ? std::string()
                           : " from " + Platform.Path + " (digest " +
                                 Platform.Digest + ")")
                   << "\n";
  }

  BaseArgs.push_back("--disable");
  BaseArgs.push_back("generate:proofs");

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
          bestCandidate.CompCost =
              getCompCost(bestExpr.str(), M, valueToNodeMap, symbolToValueMap,
                          cast<Instruction>(CO.oldOutput)->getFastMathFlags());
        }
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
        if (!skipEvaluation) {
          candidate.CompCost =
              getCompCost(expr.str(), M, valueToNodeMap, symbolToValueMap,
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

    std::string cacheFilePath;
    bool cached = false;

    // The cache key is positional (function + subgraph index); the input digest
    // stamped beside the output detects a decomposition change instead of
    // replaying another expression's rewrite.
    std::string inputDigest;
    {
      std::string joined;
      for (const auto &expr : inputExprs) {
        joined += expr;
        joined += '\n';
      }
      inputDigest = llvm::utohexstr(llvm::xxh3_64bits(joined));
    }
    std::string cacheStampPath, platformStampPath, timeoutMarkerPath;

    if (!flags::Cache.empty()) {
      // The platform leads the key: Herbie ranks its candidates by the
      // platform's cost table, so the same expression under another device's
      // platform is another device's answer and must not be replayed here.
      // There is no fallback to an entry without the component.
      //
      // The site's canonical-form hash is part of the key too: subgraphIdx
      // resets per function, so two functions' subgraph 0 would otherwise
      // share a cache file. The hash rather than the function name, so that
      // renaming the kernel a site was cut from does not invalidate shipped
      // results.
      std::string keyStem = Platform.Name + "_" + cacheKey.str() + "_" +
                            std::to_string(subgraphIdx) + "_" +
                            std::to_string(baseArgsIndex);
      cacheFilePath = flags::Cache + "/cachedHerbieOutput_" + keyStem + ".txt";
      cacheStampPath = cacheFilePath + ".input";
      platformStampPath = cacheFilePath + ".platform";
      // A positional collision (the same index reached twice in one compile)
      // spills into a digest-qualified sibling file; the unqualified name stays
      // the primary key so older entries are still found.
      {
        std::string stamped;
        std::ifstream probeStamp(cacheStampPath);
        if (probeStamp) {
          std::getline(probeStamp, stamped);
          probeStamp.close();
        }
        bool primaryTaken = llvm::sys::fs::exists(cacheFilePath);
        if (primaryTaken && !stamped.empty() && stamped != inputDigest) {
          cacheFilePath = flags::Cache + "/cachedHerbieOutput_" + keyStem +
                          "_" + inputDigest + ".txt";
          cacheStampPath = cacheFilePath + ".input";
          platformStampPath = cacheFilePath + ".platform";
          if (flags::Print)
            llvm::errs() << "[poseidon] Herbie cache key collision on subgraph "
                         << subgraphIdx << " of " << funcTag
                         << " (positional entry holds digest " << stamped
                         << ", this expression is " << inputDigest
                         << "); using the digest-qualified entry "
                         << cacheFilePath << "\n";
        }
      }
      // A timeout gets its own digest-qualified marker file so it never looks
      // like an empty Herbie result.
      timeoutMarkerPath = flags::Cache + "/cachedHerbieTimeout_" + keyStem +
                          "_" + inputDigest + ".txt";
      std::ifstream cacheFile(cacheFilePath);
      if (cacheFile) {
        content.assign((std::istreambuf_iterator<char>(cacheFile)),
                       std::istreambuf_iterator<char>());
        cacheFile.close();
        std::ifstream stampFile(cacheStampPath);
        if (stampFile) {
          std::string stamped;
          std::getline(stampFile, stamped);
          stampFile.close();
          if (stamped != inputDigest)
            report_fatal_error(
                Twine("Poseidon: cached Herbie output ") + cacheFilePath +
                " was produced for a DIFFERENT expression (input digest " +
                stamped + ", current " + inputDigest +
                "). The subgraph decomposition changed under a positional "
                "cache "
                "key; applying it would rewrite this subgraph with another "
                "one's result. Point --poseidon-cache at a fresh directory "
                "for this configuration; do not delete the existing Herbie "
                "outputs.");
        } else if (flags::Print) {
          llvm::errs() << "Poseidon: cached Herbie output " << cacheFilePath
                       << " predates input stamping; accepted unverified.\n";
        }
        // The entry's name already says which platform it was searched under.
        // The stamp says which COSTS: a device that was recalibrated keeps its
        // architecture and its platform name, and the rewrites Herbie ranked
        // against the previous prices are still replayed, but not silently.
        {
          std::string stampedName, stampedDigest;
          std::ifstream platformStamp(platformStampPath);
          if (platformStamp) {
            platformStamp >> stampedName >> stampedDigest;
            platformStamp.close();
          }
          if (!stampedDigest.empty() && !Platform.Digest.empty() &&
              stampedDigest != Platform.Digest)
            llvm::errs()
                << "[poseidon] WARNING: cached Herbie output " << cacheFilePath
                << " was searched under platform '" << stampedName
                << "' with cost digest " << stampedDigest
                << ", but this compile's platform " << Platform.Path
                << " digests " << Platform.Digest
                << ". Same architecture, different measured costs: the "
                   "replayed rewrites are the ones Herbie ranked against the "
                   "OTHER cost table. Point -poseidon-cache at a fresh "
                   "directory to search again under these prices.\n";
          else if (stampedDigest.empty() && flags::Print)
            llvm::errs() << "Poseidon: cached Herbie output " << cacheFilePath
                         << " predates platform stamping; the platform name in "
                            "its key is all that is checked.\n";
        }
        std::string stampedBinary;
        std::ifstream bStamp(cacheFilePath + ".herbie");
        if (bStamp)
          std::getline(bStamp, stampedBinary);
        if (!stampedBinary.empty() && stampedBinary != BinaryDigest) {
          llvm::errs() << "[poseidon] cached Herbie output " << cacheFilePath
                       << " was searched by another Herbie binary ("
                       << stampedBinary << "; this one is " << BinaryDigest
                       << "); searching again.\n";
          content.clear();
        } else {
          llvm::errs() << "Using cached Herbie output from " << cacheFilePath
                       << "\n";
          cached = true;
        }
      }
      if (!cached) {
        // A recorded timeout is a result too: without it every re-solve pays
        // the full wall-clock budget again to learn the same thing.
        std::ifstream markerFile(timeoutMarkerPath);
        if (markerFile) {
          std::string stamped, markerBinary;
          unsigned spentBudget = 0;
          markerFile >> stamped >> spentBudget >> markerBinary;
          markerFile.close();
          if (stamped == inputDigest && wallBudget > 0 &&
              spentBudget >= wallBudget &&
              (markerBinary.empty() || markerBinary == BinaryDigest)) {
            llvm::errs()
                << "[poseidon] Herbie subgraph " << subgraphIdx << " of "
                << funcTag << ": cached TIMEOUT (" << spentBudget
                << " s budget already spent on this exact input, "
                << timeoutMarkerPath
                << "). No algebraic candidates for this subgraph. Raise "
                   "-poseidon-herbie-subgraph-timeout above "
                << spentBudget << " to retry it.\n";
            continue;
          }
        }
      }
    }

    if (cached) {
      llvm::errs() << "Herbie output: " << content << "\n";
      // Keyed by this function, like the DP cache itself.
      bool skipEvaluation = dpCacheHasFunction(funcTag);
      if (processHerbieOutput(content, skipEvaluation)) {
        success = true;
      }
      continue;
    }

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

    if (flags::Print) {
      llvm::errs() << "Executing Herbie with arguments: ";
      for (const auto &arg : Args) {
        llvm::errs() << arg << " ";
      }
      llvm::errs() << "\n";
    }
    llvm::errs() << "[poseidon] Herbie subgraph " << subgraphIdx << " of "
                 << funcTag << ": " << inputExprs.size()
                 << " FPCore(s), per-core timeout " << coreTimeout
                 << " s, wall budget "
                 << (wallBudget ? std::to_string(wallBudget) + " s"
                                : std::string("unbounded"))
                 << ", input " << inputDigest << "\n";

    auto herbieStart = std::chrono::steady_clock::now();
    int RC =
        llvm::sys::ExecuteAndWait(Program, Args, /*Env=*/{},
                                  /*Redirects=*/{},
                                  /*SecondsToWait=*/wallBudget,
                                  /*MemoryLimit=*/0, &ErrMsg, &ExecutionFailed);
    double herbieSecs = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - herbieStart)
                            .count();

    std::remove(tmpin.c_str());
    if (ExecutionFailed) {
      llvm::errs() << "Execution failed: " << ErrMsg << "\n";
      if (auto EC = llvm::sys::fs::remove_directories(tmpout))
        llvm::errs() << "Warning: Failed to remove temporary output directory: "
                     << EC.message() << "\n";
      continue;
    }
    // RC == -2 means SecondsToWait elapsed and the child was killed: a
    // reportable negative result for this subgraph, not a compiler error.
    if (RC == -2) {
      llvm::errs()
          << "[poseidon] Herbie subgraph " << subgraphIdx << " of " << funcTag
          << ": KILLED after " << herbieSecs << " s (wall budget " << wallBudget
          << " s). No algebraic candidates for this subgraph; the "
             "other candidate families are unaffected. Raise "
             "-poseidon-herbie-subgraph-timeout to spend more, or set it "
             "to 0 for the historical unbounded behaviour.\n";
      if (auto EC = llvm::sys::fs::remove_directories(tmpout))
        llvm::errs() << "Warning: Failed to remove temporary output directory: "
                     << EC.message() << "\n";
      if (!timeoutMarkerPath.empty()) {
        if (auto EC = llvm::sys::fs::create_directories(flags::Cache, true))
          llvm::errs() << "Warning: Could not create cache directory: "
                       << EC.message() << "\n";
        std::ofstream markerFile(timeoutMarkerPath);
        if (markerFile) {
          markerFile << inputDigest << " " << wallBudget << " " << BinaryDigest
                     << "\n";
          markerFile.close();
          llvm::errs() << "Recorded the timeout in " << timeoutMarkerPath
                       << "\n";
        }
      }
      continue;
    }

    std::ifstream output((tmpout + "/results.json").str());
    if (!output) {
      // Herbie ran and wrote nothing. The common cause is a platform it could
      // not load (it exits 1 and reports on stderr, just above this message).
      // Continuing here would drop every algebraic candidate for this subgraph
      // and report the solve as a success, so it aborts instead.
      if (auto EC = llvm::sys::fs::remove_directories(tmpout))
        llvm::errs() << "Warning: Failed to remove temporary output directory: "
                     << EC.message() << "\n";
      std::string where = Platform.Name;
      if (!Platform.Path.empty())
        where += ", " + Platform.Path;
      report_fatal_error(Twine("Poseidon: Herbie exited with status ") +
                         Twine(RC) +
                         " and produced no results.json for subgraph " +
                         Twine(subgraphIdx) + " of " + funcTag + " (platform " +
                         where + "). Its own diagnostics are on stderr above.");
    }
    content.assign((std::istreambuf_iterator<char>(output)),
                   std::istreambuf_iterator<char>());
    output.close();
    if (auto EC = llvm::sys::fs::remove_directories(tmpout))
      llvm::errs() << "Warning: Failed to remove temporary output directory: "
                   << EC.message() << "\n";

    llvm::errs() << "Herbie output: " << content << "\n";

    if (!flags::Cache.empty()) {
      if (auto EC = llvm::sys::fs::create_directories(flags::Cache, true))
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
      // Stamp what produced it, so a later solve whose subgraph numbering has
      // shifted cannot silently reuse this entry (checked on the read path).
      std::ofstream stampFile(cacheStampPath);
      if (stampFile) {
        stampFile << inputDigest << "\n";
        stampFile.close();
      }
      // And which platform ranked it, so a later solve under recalibrated
      // prices for the same architecture is reported on the read path.
      if (!Platform.Digest.empty()) {
        std::ofstream pStamp(platformStampPath);
        if (pStamp) {
          pStamp << Platform.Name << " " << Platform.Digest << "\n";
          pStamp.close();
        }
      }
      if (!BinaryDigest.empty()) {
        std::ofstream bStamp(cacheFilePath + ".herbie");
        if (bStamp) {
          bStamp << BinaryDigest << "\n";
          bStamp.close();
        }
      }
      // A stale timeout marker under this key is left in place: the read path
      // consults it only when the output file is absent.
    }
    llvm::errs() << "[poseidon] Herbie subgraph " << subgraphIdx << " of "
                 << funcTag << ": completed in " << herbieSecs << " s\n";

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

    if (StringRef deviceMath = deviceMathName(funcName); !deviceMath.empty())
      return deviceMath.str();

    if (funcName.starts_with("llvm.")) {
      std::regex regex("llvm\\.(\\w+)\\.?.*");
      std::smatch matches;
      std::string nameStr = funcName.str();
      if (std::regex_search(nameStr, matches, regex) && matches.size() > 1) {
        std::string intrinsic = matches[1];
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
    } else {
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

void setUnifiedAccuracyCost(
    CandidateOutput &CO,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap) {

  CancellationPlan cancelPlan =
      buildCancellationPlan(*CO.subgraph, valueToNodeMap);
  SmallVector<MapVector<Value *, double>, 4> sampledPoints;

  getSampledPoints(CO.subgraph->inputs.getArrayRef(), valueToNodeMap,
                   symbolToValueMap, sampledPoints, &cancelPlan);

  // No usable sample means the output cannot be priced: drop its candidates and
  // keep the original body, without inventing a bound or an accuracy number.
  auto unpriceable = [&](const char *why) {
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

  SmallVector<double, 4> goldVals;
  goldVals.resize(flags::NumSamples);
  SmallVector<double, 4> origErrors;
  origErrors.resize(flags::NumSamples);
  SmallVector<double, 4> origVals;
  origVals.resize(flags::NumSamples);

  double origCost = 0.0;
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
    double error = sampleError(goldVal, realVal);
    origErrors[pair.index()] = error;
    origVals[pair.index()] = realVal;
    if (!std::isnan(error)) {
      sum += error;
      ++count;
    }
  }
  if (count == 0) {
    unpriceable("every sample of the original expression scored NaN");
    return;
  }
  origCost = sum / count;
  CO.initialAccCost = origCost * std::fabs(CO.grad);
  if (!std::isfinite(CO.initialAccCost)) {
    // A non-finite baseline makes every candidate delta nan and the DP would
    // silently fall back to the no-op; drop the candidates and say so instead.
    llvm::errs() << "  origCost = " << origCost << ", grad = " << CO.grad
                 << "\n";
    unpriceable("the baseline accuracy cost is not finite; the site's profile "
                "is degenerate (a zero or null-space output shadow gives a "
                "gradient that cannot weight anything)");
    return;
  }

  SmallVector<RewriteCandidate, 4> newCandidates;
  for (auto &candidate : CO.candidates) {
    bool discardCandidate = false;
    double candCost = 0.0;

    std::shared_ptr<FPNode> parsedNode =
        parseHerbieExpr(candidate.expr, valueToNodeMap, symbolToValueMap);

    SmallVector<const FPNode *, 2> ifNodes;
    if (flags::MinArmSamples > 0) {
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
      double goldVal = goldVals[pair.index()];
      if (std::isfinite(goldVal) && goldVal != 0.0) {
        double origRel =
            std::fabs(origVals[pair.index()] - goldVal) / std::fabs(goldVal);
        double candRel = std::fabs(realVal - goldVal) / std::fabs(goldVal);
        if (origRel < 1.0 && !(candRel < 1.0))
          ++broken;
      }

      if (std::isfinite(goldVal) && !std::isfinite(realVal)) {
        discardCandidate = true;
        break;
      }

      double error = sampleError(goldVal, realVal);
      if (!std::isnan(error)) {
        sum += error;
        ++count;
        double origError = origErrors[pair.index()];
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
      } else {
        candCost = sum / count;
      }
    }
    if (!discardCandidate && flags::MaxBrokenShare >= 0.0 &&
        broken > flags::MaxBrokenShare * sampledPoints.size()) {
      if (flags::Print)
        llvm::errs() << "[poseidon] dropping candidate: relative error >= 1 on "
                     << broken << " of " << sampledPoints.size()
                     << " samples: " << candidate.expr.substr(0, 120) << "\n";
      discardCandidate = true;
    }

    // Uniform marginal sampling weights each regime by its share of the
    // profiled box, which need not be its share of the workload: correlated
    // inputs can put most of the run in an arm the box almost never reaches.
    // Price every arm on its own samples and charge the worst arm's excess
    // over the original.
    if (!discardCandidate && !ifNodes.empty()) {
      auto starved = [&] {
        for (const auto &kv : armStats)
          if (kv.second.count < flags::MinArmSamples)
            return true;
        return false;
      };
      size_t budget = static_cast<size_t>(flags::ArmSearchFactor) *
                      flags::NumSamples.getValue();
      size_t drawn = 0;
      unsigned round = 0;
      std::shared_ptr<FPNode> origNode = valueToNodeMap[CO.oldOutput];
      while (starved() && drawn < budget) {
        SmallVector<MapVector<Value *, double>, 4> extra;
        getSampledPoints(CO.subgraph->inputs.getArrayRef(), valueToNodeMap,
                         symbolToValueMap, extra, nullptr,
                         flags::NumSamples.getValue(), ++round);
        if (extra.empty())
          break;
        drawn += extra.size();
        for (const auto &point : extra) {
          SmallVector<RegimeArm, 4> arms = takenArms(parsedNode.get(), point);
          bool wanted = false;
          for (const RegimeArm &arm : arms)
            if (armStats[arm].count < flags::MinArmSamples)
              wanted = true;
          if (!wanted)
            continue;
          SmallVector<double, 1> r;
          getMPFRValues({origNode.get()}, point, r, true, 53);
          double goldVal = r[0];
          getFPValues({origNode.get()}, point, r);
          double origError = sampleError(goldVal, r[0]);
          getFPValues({parsedNode.get()}, point, r);
          double error = sampleError(goldVal, r[0]);
          if (std::isnan(error) || std::isnan(origError))
            continue;
          for (const RegimeArm &arm : arms) {
            ArmStats &st = armStats[arm];
            if (st.count >= flags::MinArmSamples)
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
        if (st.count < flags::MinArmSamples) {
          discardCandidate = true;
          continue;
        }
        worst = std::max(worst, origCost + (st.candSum - st.origSum) / st.count);
        }
      if (flags::Print)
        llvm::errs() << "[poseidon] regime-split candidate: arms (samples:cand/"
                        "orig mean error)"
                     << os.str() << "; " << drawn << " extra draws; box cost "
                     << format("%.3e", candCost) << " -> "
                     << (discardCandidate ? std::string("dropped (unpriced arm)")
                                          : formatv("{0:e}", worst).str())
                     << "; " << candidate.expr.substr(0, 120) << "\n";
      if (!discardCandidate)
        candCost = worst;
    }

    if (!discardCandidate) {
      candidate.accuracyCost = candCost * std::fabs(CO.grad);
      assert(!std::isnan(candidate.accuracyCost));
      newCandidates.push_back(std::move(candidate));
    }
  }
  CO.candidates = std::move(newCandidates);
}

double getCompCost(
    const std::string &expr, Module *M,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap,
    const FastMathFlags &FMF) {
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
  if (RetVal->getType() != ReturnType)
    RetVal = builder.CreateFPCast(RetVal, ReturnType);
  builder.CreateRet(RetVal);

  simplifyFunction(*tempFunction, OptimizationLevel::O3);

  double cost = getCompCost(tempFunction);

  tempFunction->eraseFromParent();
  return cost;
}

} // namespace poseidon
