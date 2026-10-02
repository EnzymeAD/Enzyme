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

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "CostModel.h"
#include "Flags.h"
#include "Optimize.h"
#include "Solvers.h"
#include "Types.h"

#include "Herbie.h"
#include "Precision.h"
#include "Utils.h"
#include "matmul/Matmul.h"

#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <random>

using namespace llvm;

#if LLVM_VERSION_MAJOR >= 21
#define GET_INSTRUCTION_COST(cost) (cost.getValue())
#else
#define GET_INSTRUCTION_COST(cost) (cost.getValue().value())
#endif

namespace poseidon {

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

// Footprint mutual exclusion: a tensor-core raise and an elementwise rewrite
// may not claim the same instructions. Only cross-family and matmul-vs-matmul
// overlap is rejected; elementwise-vs-elementwise on one subgraph is repriced
// by CandidateSubgraph::getAdjustedCompCostDelta and must stay allowed.
static const SetVector<Instruction *> *
stepMatmulFootprint(const SolutionStep &step) {
  if (auto *const *cm = std::get_if<CandidateMatmul *>(&step.item))
    if (*cm && (*cm)->matmul)
      return &(*cm)->matmul->footprint;
  return nullptr;
}

static const SetVector<Instruction *> *
stepElementwiseOps(const SolutionStep &step) {
  if (auto *const *co = std::get_if<CandidateOutput *>(&step.item))
    if (*co && (*co)->subgraph)
      return &(*co)->subgraph->operations;
  if (auto *const *cs = std::get_if<CandidateSubgraph *>(&step.item))
    if (*cs && (*cs)->subgraph)
      return &(*cs)->subgraph->operations;
  return nullptr;
}

static bool intersects(const SetVector<Instruction *> &fp,
                       const SetVector<Instruction *> &ops) {
  if (fp.size() <= ops.size()) {
    for (Instruction *I : fp)
      if (ops.contains(I))
        return true;
    return false;
  }
  for (Instruction *I : ops)
    if (fp.contains(I))
      return true;
  return false;
}

static bool conflictsWithChosen(const SmallVectorImpl<SolutionStep> &priorSteps,
                                CandidateMatmul *cand) {
  if (!cand || !cand->matmul)
    return false;
  const auto &fp = cand->matmul->footprint;
  for (const auto &step : priorSteps) {
    if (const auto *ops = stepElementwiseOps(step))
      if (intersects(fp, *ops))
        return true;
    if (const auto *other = stepMatmulFootprint(step)) {
      if (other == &fp)
        continue;
      for (Instruction *I : *other)
        if (fp.contains(I))
          return true;
    }
  }
  return false;
}
static bool conflictsWithChosen(const SmallVectorImpl<SolutionStep> &priorSteps,
                                CandidateOutput *cand) {
  if (!cand || !cand->subgraph)
    return false;
  for (const auto &step : priorSteps)
    if (const auto *fp = stepMatmulFootprint(step))
      if (intersects(*fp, cand->subgraph->operations))
        return true;
  return false;
}
static bool conflictsWithChosen(const SmallVectorImpl<SolutionStep> &priorSteps,
                                CandidateSubgraph *cand) {
  if (!cand || !cand->subgraph)
    return false;
  for (const auto &step : priorSteps)
    if (const auto *fp = stepMatmulFootprint(step))
      if (intersects(*fp, cand->subgraph->operations))
        return true;
  return false;
}

static CandidateMatmul *stepMatmul(const SolutionStep &step) {
  if (auto *const *cm = std::get_if<CandidateMatmul *>(&step.item))
    return *cm;
  return nullptr;
}

// An elementwise candidate that overlaps already-chosen matmul steps displaces
// them and charges their deltas back: exact, because a CandidateMatmul's deltas
// depend only on the matmul. The reverse (a matmul displacing an elementwise
// step) stays a refusal, since a CandidateSubgraph's delta is prior-dependent.
// Without displacement the frontier keyed by cost evicts equal-cost prefixes
// that later subgraphs would have extended.
static bool resolveConflicts(const SmallVectorImpl<SolutionStep> &priorSteps,
                             CandidateMatmul *cand,
                             SmallVector<SolutionStep> &reduced,
                             InstructionCost &base, double &baseAcc) {
  if (conflictsWithChosen(priorSteps, cand))
    return false;
  reduced.assign(priorSteps.begin(), priorSteps.end());
  return true;
}

template <typename ElemCandT>
static bool resolveElementwiseConflicts(
    const SmallVectorImpl<SolutionStep> &priorSteps, ElemCandT *cand,
    const SetVector<Instruction *> *ops, SmallVector<SolutionStep> &reduced,
    InstructionCost &base, double &baseAcc) {
  reduced.clear();
  if (!ops) {
    reduced.assign(priorSteps.begin(), priorSteps.end());
    return true;
  }
  for (const auto &step : priorSteps) {
    CandidateMatmul *cm = stepMatmul(step);
    if (cm && cm->matmul && intersects(cm->matmul->footprint, *ops)) {
      base -= cm->getCompCostDelta(step.candidateIndex);
      baseAcc -= cm->getAccCostDelta(step.candidateIndex);
      continue;
    }
    reduced.push_back(step);
  }
  return true;
}

static bool resolveConflicts(const SmallVectorImpl<SolutionStep> &priorSteps,
                             CandidateOutput *cand,
                             SmallVector<SolutionStep> &reduced,
                             InstructionCost &base, double &baseAcc) {
  return resolveElementwiseConflicts(
      priorSteps, cand,
      cand && cand->subgraph ? &cand->subgraph->operations : nullptr, reduced,
      base, baseAcc);
}

static bool resolveConflicts(const SmallVectorImpl<SolutionStep> &priorSteps,
                             CandidateSubgraph *cand,
                             SmallVector<SolutionStep> &reduced,
                             InstructionCost &base, double &baseAcc) {
  return resolveElementwiseConflicts(
      priorSteps, cand,
      cand && cand->subgraph ? &cand->subgraph->operations : nullptr, reduced,
      base, baseAcc);
}

template <typename CandT, typename CompFn, typename AccFn>
static void
paretoIntegrate(std::map<InstructionCost, double> &cost,
                std::map<InstructionCost, SmallVector<SolutionStep>> &sol,
                CandT *cand, size_t numCandidates, CompFn compFn, AccFn accFn) {
  std::map<InstructionCost, double> newCost = cost;
  std::map<InstructionCost, SmallVector<SolutionStep>> newSol = sol;
  for (const auto &p : cost) {
    InstructionCost curr = p.first;
    double currAcc = p.second;
    const SmallVector<SolutionStep> &chosen = sol[p.first];
    SmallVector<SolutionStep> priorSteps;
    if (!resolveConflicts(chosen, cand, priorSteps, curr, currAcc))
      continue;
    for (size_t i = 0; i < numCandidates; ++i) {
      auto cd = compFn(i, priorSteps);
      auto ad = accFn(i, priorSteps);
      // Compared against the post-displacement point, so a displacement that
      // gives back more than the candidate saves is dropped.
      if (curr + cd >= p.first && currAcc + ad >= p.second)
        continue;
      InstructionCost nc = curr + cd;
      double na = currAcc + ad;
      auto it = newCost.find(nc);
      if (it == newCost.end() || it->second > na) {
        newCost[nc] = na;
        newSol[nc] = priorSteps;
        newSol[nc].emplace_back(cand, i);
      }
    }
  }
  cost = std::move(newCost);
  sol = std::move(newSol);
}

static void
paretoPrune(std::map<InstructionCost, double> &cost,
            std::map<InstructionCost, SmallVector<SolutionStep>> &sol) {
  std::map<InstructionCost, double> pruned;
  std::map<InstructionCost, SmallVector<SolutionStep>> prunedSol;
  for (const auto &l : cost) {
    InstructionCost lc = l.first;
    double la = l.second;
    bool dominated = false;
    for (const auto &r : cost) {
      InstructionCost rc = r.first;
      double ra = r.second;
      if (lc > rc && la >= ra) {
        dominated = true;
        break;
      }
    }
    if (!dominated) {
      pruned[lc] = la;
      prunedSol[lc] = sol[lc];
    }
  }
  cost = std::move(pruned);
  sol = std::move(prunedSol);
}

std::string dpCachePath() {
  if (flags::Cache.empty())
    return std::string();
  return flags::Cache + "/table.json";
}

static llvm::json::Object *
openDPCacheEntry(StringRef cacheFilePath, StringRef fnName,
                 std::optional<llvm::json::Value> &root, bool quiet) {
  if (cacheFilePath.empty() || !llvm::sys::fs::exists(cacheFilePath))
    return nullptr;
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFile(cacheFilePath);
  if (std::error_code ec = fileOrErr.getError())
    report_fatal_error("Error reading DP cache file: " + Twine(ec.message()));
  llvm::Expected<llvm::json::Value> jsonOrErr =
      llvm::json::parse(fileOrErr.get()->getBuffer());
  if (!jsonOrErr) {
    // An unreadable cache is a reason to re-solve, never to fail the build
    // (older writers emitted bare `nan` tokens); the file is left in place.
    if (!quiet)
      llvm::errs() << "[poseidon] DP cache at " << cacheFilePath
                   << " is not valid JSON ("
                   << llvm::toString(jsonOrErr.takeError())
                   << "); ignoring it and re-solving. The file is left in "
                      "place; delete it to stop this warning.\n";
    else
      llvm::consumeError(jsonOrErr.takeError());
    return nullptr;
  }
  root = std::move(*jsonOrErr);
  llvm::json::Object *jsonObj = root->getAsObject();
  if (!jsonObj) {
    if (!quiet)
      llvm::errs() << "[poseidon] DP cache at " << cacheFilePath
                   << " is not a JSON object; ignoring it and re-solving.\n";
    root.reset();
    return nullptr;
  }

  // A table written against a different cost model (or by a build predating
  // the stamp) must not be replayed: every Delta-cost in it came from prices
  // that no longer hold.
  std::string want = std::to_string(getCostModelFingerprint());
  std::optional<llvm::StringRef> got =
      jsonObj->getString("costModelFingerprint");
  if (!got || got->str() != want) {
    if (!quiet)
      llvm::errs() << "[poseidon] DP cache at " << cacheFilePath
                   << " was solved against a different cost model ("
                   << (got ? got->str() : std::string("<unstamped>"))
                   << " != " << want << "); ignoring it and re-solving.\n";
    root.reset();
    return nullptr;
  }

  llvm::json::Object *funcs = jsonObj->getObject("functions");
  if (!funcs) {
    // Pre-keying layout: one anonymous table at the top level. It cannot be
    // attributed to a function, and replaying it for the wrong site binds
    // cached steps positionally into a different candidate vector.
    if (!quiet)
      llvm::errs() << "[poseidon] DP cache at " << cacheFilePath
                   << " predates per-function keying (no \"functions\" map); "
                      "ignoring it and re-solving.\n";
    root.reset();
    return nullptr;
  }
  llvm::json::Object *entry = funcs->getObject(fnName);
  if (!entry) {
    root.reset();
    return nullptr;
  }
  return entry;
}

// Keyed on this function: a table for an earlier site says nothing about this
// one.
bool dpCacheHasFunction(StringRef fnName) {
  std::optional<llvm::json::Value> root;
  return openDPCacheEntry(dpCachePath(), fnName, root,
                          /*quiet=*/true) != nullptr;
}

// A site tolerance is a per-operation relative rounding level, so everything
// the ratio is built from has to be real and measured; a missing piece is an
// upstream defect, never something to substitute a value for.
static void checkToleranceInputs(StringRef siteName, double accScale,
                                 double baseline, bool loadedFromCache,
                                 StringRef cacheFilePath) {
  // The ratio is only dimensionless because the sampled errors are absolute.
  if (flags::RelativeError)
    report_fatal_error(
        Twine("Poseidon: -poseidon-relative-error cannot be combined with a "
              "site accuracy tolerance (") +
        siteName +
        " was given one): the tolerance is the ratio of an ABSOLUTE modelled "
        "error to the site's total sensitivity, and relative sampling makes "
        "that ratio meaningless. Drop one of the two.");

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

  if (!std::isfinite(baseline)) {
    if (loadedFromCache)
      report_fatal_error(
          Twine("Poseidon: the cached DP table for ") + siteName + " in " +
          cacheFilePath +
          " predates per-site accuracy tolerances and carries no baseline "
          "accuracy cost, and a cache hit skips the pricing that would "
          "recompute it. Delete that table.json (Herbie results are cached "
          "separately and are not affected) and recompile.");
    report_fatal_error(
        Twine("Poseidon: site ") + siteName +
        " was given an accuracy tolerance but its own baseline accuracy cost "
        "is not finite, so no candidate's modelled error can be placed on an "
        "absolute scale.");
  }
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
    baseline += CO.initialAccCost;
  }
  if (unpricedOut)
    *unpricedOut = unpriced;
  return baseline;
}

static bool loadDPCache(
    StringRef cacheFilePath, StringRef fnName,
    SmallVector<CandidateOutput, 4> &COs,
    SmallVector<CandidateSubgraph, 4> &CSs,
    SmallVector<CandidateMatmul, 4> &CMs,
    std::map<InstructionCost, double> &costToAccuracyMap,
    std::map<InstructionCost, SmallVector<SolutionStep>> &costToSolutionMap,
    std::optional<double> &cachedBaseline) {
  std::optional<llvm::json::Value> root;
  llvm::json::Object *jsonObj =
      openDPCacheEntry(cacheFilePath, fnName, root, /*quiet=*/false);
  if (!jsonObj)
    return false;

  // A cache hit skips candidate pricing entirely (dpCacheHasFunction gates
  // setUnifiedAccuracyCost), so the baseline cannot be recomputed here and has
  // to come out of the table that was written with it.
  cachedBaseline = jsonObj->getNumber("baselineAccCost");

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
      if (!stepsArray)
        report_fatal_error("Invalid steps array in DP cache file");

      for (llvm::json::Value &stepVal : *stepsArray) {
        llvm::json::Object *stepObj = stepVal.getAsObject();
        if (!stepObj)
          llvm_unreachable("Invalid step object in cache file.");

        StringRef itemType = stepObj->getString("itemType").value();
        size_t candidateIndex = stepObj->getInteger("candidateIndex").value();
        size_t itemIndex = stepObj->getInteger("itemIndex").value();

        // The table is keyed by function name, not by candidate set: the same
        // body can decompose into a different number of units in another
        // translation unit, so out-of-range indices mean a stale table, which
        // must trigger a re-solve rather than an abort.
        auto staleCache = [&](const char *what) -> bool {
          llvm::errs() << "[poseidon] DP cache for " << fnName << " is STALE ("
                       << what << ": item " << itemIndex << ", candidate "
                       << candidateIndex
                       << " out of range for this compile, which has "
                       << COs.size() << " CO / " << CSs.size() << " CS / "
                       << CMs.size()
                       << " CM). Discarding the cached table and re-solving.\n";
          costToAccuracyMap.clear();
          costToSolutionMap.clear();
          return false;
        };
        // The candidate index is as compile-specific as the item count (Herbie
        // alternatives, strict-mode filtering), so both axes are checked.
        if (itemType == "CO") {
          if (itemIndex >= COs.size())
            return staleCache("CandidateOutput");
          if (candidateIndex >= COs[itemIndex].candidates.size())
            return staleCache("CandidateOutput candidate");
          solutionSteps.emplace_back(&COs[itemIndex], candidateIndex);
        } else if (itemType == "CS") {
          if (itemIndex >= CSs.size())
            return staleCache("CandidateSubgraph");
          if (candidateIndex >= CSs[itemIndex].candidates.size())
            return staleCache("CandidateSubgraph candidate");
          solutionSteps.emplace_back(&CSs[itemIndex], candidateIndex);
        } else if (itemType == "CM") {
          if (itemIndex >= CMs.size())
            return staleCache("CandidateMatmul");
          if (candidateIndex >= CMs[itemIndex].candidates.size())
            return staleCache("CandidateMatmul candidate");
          solutionSteps.emplace_back(&CMs[itemIndex], candidateIndex);
        } else {
          llvm_unreachable("Invalid itemType in cache file.");
        }
      }

      costToSolutionMap[compCost] = solutionSteps;
    }
  } else {
    report_fatal_error("costToSolutionMap not found in DP cache file");
  }
  return true;
}

static void
writeDPCache(StringRef cacheFilePath, StringRef fnName,
             const std::map<InstructionCost, double> &costToAccuracyMap,
             const std::map<InstructionCost, SmallVector<SolutionStep>>
                 &costToSolutionMap,
             ArrayRef<CandidateOutput> COs, ArrayRef<CandidateSubgraph> CSs,
             ArrayRef<CandidateMatmul> CMs, double baseline) {
  std::unordered_map<const CandidateOutput *, size_t> coPtrToIndex;
  for (size_t i = 0; i < COs.size(); ++i)
    coPtrToIndex[&COs[i]] = i;
  std::unordered_map<const CandidateSubgraph *, size_t> csPtrToIndex;
  for (size_t i = 0; i < CSs.size(); ++i)
    csPtrToIndex[&CSs[i]] = i;
  std::unordered_map<const CandidateMatmul *, size_t> cmPtrToIndex;
  for (size_t i = 0; i < CMs.size(); ++i)
    cmPtrToIndex[&CMs[i]] = i;

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

  json::Object costAccMap;
  for (const auto &pair : costToAccuracyMap) {
    costAccMap[std::to_string(GET_INSTRUCTION_COST(pair.first))] = pair.second;
  }
  entryObj["costToAccuracyMap"] = std::move(costAccMap);

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
              stepObj["itemIndex"] = static_cast<int64_t>(coPtrToIndex[item]);
            } else if constexpr (std::is_same_v<T, CandidateSubgraph>) {
              stepObj["itemType"] = "CS";
              stepObj["itemIndex"] = static_cast<int64_t>(csPtrToIndex[item]);
            } else if constexpr (std::is_same_v<T, CandidateMatmul>) {
              stepObj["itemType"] = "CM";
              stepObj["itemIndex"] = static_cast<int64_t>(cmPtrToIndex[item]);
            }
          },
          step.item);
      stepsArray.push_back(std::move(stepObj));
    }
    costSolMap[std::to_string(GET_INSTRUCTION_COST(pair.first))] =
        std::move(stepsArray);
  }
  entryObj["costToSolutionMap"] = std::move(costSolMap);
  if (std::isfinite(baseline))
    entryObj["baselineAccCost"] = baseline;

  // Read-modify-write: the file holds one table PER FUNCTION, so a later site
  // in the same compilation unit must extend it rather than clobber the
  // earlier sites' tables. Entries written against a different cost model are
  // dropped wholesale (the fingerprint is a property of the whole file).
  json::Object functions;
  const std::string wantFP = std::to_string(getCostModelFingerprint());
  if (llvm::sys::fs::exists(cacheFilePath)) {
    if (auto fileOrErr = llvm::MemoryBuffer::getFile(cacheFilePath)) {
      if (auto prior = llvm::json::parse(fileOrErr.get()->getBuffer())) {
        if (json::Object *priorObj = prior->getAsObject()) {
          std::optional<StringRef> fp =
              priorObj->getString("costModelFingerprint");
          if (fp && fp->str() == wantFP)
            if (json::Object *priorFuncs = priorObj->getObject("functions"))
              functions = *priorFuncs;
        }
      } else {
        llvm::consumeError(prior.takeError());
      }
    }
  }
  functions[fnName] = std::move(entryObj);

  json::Object jsonObj;
  // The table is a function of every price the model supplied; without the
  // stamp a re-solve after a cost-model change would replay the previous picks.
  jsonObj["costModelFingerprint"] = wantFP;
  jsonObj["functions"] = std::move(functions);

  std::error_code EC;
  llvm::raw_fd_ostream cacheFile(cacheFilePath, EC, llvm::sys::fs::OF_Text);
  if (EC) {
    llvm::errs() << "Error writing cache file: " << EC.message() << "\n";
    return;
  }
  cacheFile << llvm::formatv("{0:2}", llvm::json::Value(std::move(jsonObj)))
            << "\n";
  cacheFile.close();
  llvm::errs() << "DP table for " << fnName << " cached to file.\n";
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
    for (const auto &step : it->second) {
      std::visit(
          [&](auto *item) {
            using T = std::decay_t<decltype(*item)>;
            if constexpr (std::is_same_v<T, CandidateOutput>) {
              llvm::errs() << "\t\t" << item->expr << " --("
                           << step.candidateIndex << ")-> "
                           << item->candidates[step.candidateIndex].expr
                           << "\n";
            } else if constexpr (std::is_same_v<T, CandidateSubgraph>) {
              llvm::errs() << "\t\tCS: "
                           << item->candidates[step.candidateIndex].desc
                           << " (#" << step.candidateIndex << ")\n";
            } else {
              llvm_unreachable(
                  "printDPTable: Unexpected type of solution step");
            }
          },
          step.item);
    }
  }
  llvm::errs() << "*** End of DP Table ***\n\n";
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

          SmallVector<Instruction *, 8> insts;
          if (auto *I = dyn_cast<Instruction>(item->oldOutput))
            insts.push_back(I);
          for (auto *I : item->erasableInsts)
            insts.push_back(I);
          stepObj["source_locations"] = getSourceLocationsJSON(insts);

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
        } else if constexpr (std::is_same_v<T, CandidateMatmul>) {
          stepObj["type"] = "matmul_rewrite";
          stepObj["candidate_index"] =
              static_cast<int64_t>(step.candidateIndex);
          stepObj["matmul_id"] = static_cast<int64_t>(item->matmul->id);
          stepObj["target"] =
              matmulOptionLabel(item->candidates[step.candidateIndex]);
          stepObj["computation_cost_delta"] =
              GET_INSTRUCTION_COST(item->getCompCostDelta(step.candidateIndex));
          stepObj["accuracy_cost_delta"] =
              jsonFloat(item->getAccCostDelta(step.candidateIndex));
          stepObj["executions"] = static_cast<int64_t>(item->executions);
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
        } else if constexpr (std::is_same_v<T, CandidateMatmul>) {
          OS << pad << "[Matmul] matmul[" << item->matmul->id << "] -> "
             << matmulOptionLabel(item->candidates[step.candidateIndex])
             << " (#" << step.candidateIndex << ")\n";
        }
      },
      step.item);
}

static void
emitReport(StringRef funcName,
           const std::map<InstructionCost, double> &costToAccuracyMap,
           const std::map<InstructionCost, SmallVector<SolutionStep>>
               &costToSolutionMap,
           SmallVector<CandidateOutput, 4> &COs,
           SmallVector<CandidateSubgraph, 4> &CSs) {
  if (flags::ReportPath.empty())
    return;

  std::error_code EC;
  if (!llvm::sys::fs::exists(flags::ReportPath)) {
    EC = llvm::sys::fs::create_directories(flags::ReportPath);
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

  json::Array coSummary;
  for (const auto &CO : COs) {
    json::Object co;
    co["original_expr"] = CO.expr;
    co["num_candidates"] = static_cast<int64_t>(CO.candidates.size());
    co["gradient"] = jsonFloat(CO.grad);
    co["executions"] = static_cast<int64_t>(CO.executions);
    co["initial_accuracy_cost"] = jsonFloat(CO.initialAccCost);
    co["initial_computation_cost"] = CO.initialCompCost;
    if (auto *I = dyn_cast<Instruction>(CO.oldOutput)) {
      auto loc = getSourceLocationJSON(I);
      if (!loc.empty())
        co["source_location"] = std::move(loc);
    }
    coSummary.push_back(std::move(co));
  }
  report["candidate_outputs"] = std::move(coSummary);

  json::Array csSummary;
  for (size_t csIdx = 0; csIdx < CSs.size(); ++csIdx) {
    auto &CS = CSs[csIdx];
    json::Object cs;
    cs["num_candidates"] = static_cast<int64_t>(CS.candidates.size());
    cs["initial_accuracy_cost"] = jsonFloat(CS.initialAccCost);
    cs["initial_computation_cost"] = CS.initialCompCost;
    // Top-N per-candidate deltas keep an all-dominated frontier auditable (the
    // table then collapses to the no-op point).
    constexpr size_t kMaxCandRows = 40;
    SmallVector<size_t, 40> order(CS.candidates.size());
    for (size_t ci = 0; ci < order.size(); ++ci)
      order[ci] = ci;
    llvm::sort(order, [&](size_t a, size_t b) {
      return GET_INSTRUCTION_COST(CS.getCompCostDelta(a)) <
             GET_INSTRUCTION_COST(CS.getCompCostDelta(b));
    });
    json::Array cands;
    for (size_t k = 0; k < order.size() && k < kMaxCandRows; ++k) {
      size_t ci = order[k];
      json::Object c;
      c["id"] = "PT" + std::to_string(csIdx) + "_" + std::to_string(ci);
      c["description"] = CS.candidates[ci].desc;
      c["computation_cost_delta"] =
          GET_INSTRUCTION_COST(CS.getCompCostDelta(ci));
      c["accuracy_cost_delta"] = jsonFloat(CS.getAccCostDelta(ci));
      cands.push_back(std::move(c));
    }
    cs["candidates"] = std::move(cands);
    csSummary.push_back(std::move(cs));
  }
  report["candidate_subgraphs"] = std::move(csSummary);

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

  std::string jsonFile =
      (Twine(flags::ReportPath) + "/" + funcName + ".json").str();
  raw_fd_ostream jsonOut(jsonFile, EC, sys::fs::OF_Text);
  if (EC) {
    llvm::errs() << "Error writing JSON report: " << EC.message() << "\n";
  } else {
    jsonOut << formatv("{0:2}", json::Value(std::move(report))) << "\n";
    llvm::errs() << "Poseidon JSON report written to " << jsonFile << "\n";
  }

  std::string textFile =
      (Twine(flags::ReportPath) + "/" + funcName + ".txt").str();
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

  for (size_t csIdx = 0; csIdx < CSs.size(); ++csIdx) {
    auto &CS = CSs[csIdx];
    textOut << "Subgraph #" << csIdx << ": " << CS.candidates.size()
            << " candidates (Δcost, Δacc; sorted by Δcost)\n";
    constexpr size_t kMaxCandRows = 40;
    SmallVector<size_t, 40> order(CS.candidates.size());
    for (size_t ci = 0; ci < order.size(); ++ci)
      order[ci] = ci;
    llvm::sort(order, [&](size_t a, size_t b) {
      return GET_INSTRUCTION_COST(CS.getCompCostDelta(a)) <
             GET_INSTRUCTION_COST(CS.getCompCostDelta(b));
    });
    for (size_t k = 0; k < order.size() && k < kMaxCandRows; ++k) {
      size_t ci = order[k];
      textOut << "  [PT" << csIdx << "_" << ci << "] " << CS.candidates[ci].desc
              << ": Δcost=" << GET_INSTRUCTION_COST(CS.getCompCostDelta(ci))
              << " Δacc=" << format("%.6e", CS.getAccCostDelta(ci)) << "\n";
    }
  }
  textOut << "\n";

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

  {
    std::string configFile =
        (Twine(flags::ReportPath) + "/validate_config.json").str();
    {
      json::Object cfg;
      cfg["function"] = funcName.str();
      cfg["profile_path"] = flags::ProfileUse.getValue();
      cfg["cache_path"] = flags::Cache.getValue();
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
    for (size_t coIdx = 0; coIdx < COs.size(); ++coIdx) {
      auto &CO = COs[coIdx];
      for (size_t ci = 0; ci < CO.candidates.size(); ++ci) {
        auto &cand = CO.candidates[ci];
        double accDelta = CO.getAccCostDelta(ci);
        auto compDelta = CO.getCompCostDelta(ci);
        int64_t compDeltaVal = GET_INSTRUCTION_COST(compDelta);

        if (std::isnan(accDelta))
          continue;

        // comp < 0 means faster, acc < 0 means more accurate
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

        if (auto *I = dyn_cast<Instruction>(CO.oldOutput)) {
          auto loc = getSourceLocationJSON(I);
          if (!loc.empty())
            rw["source_location"] = std::move(loc);
        }

        rewrites.push_back(std::move(rw));
      }
    }

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
          // Emitted even though dominated: an all-dominated frontier must
          // leave a record of why each candidate lost.
          category = "dominated";
          efficiency = 0.0;
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
        (Twine(flags::ReportPath) + "/" + funcName + "_rewrites.json").str();
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

bool stepConflictsWithMatmulSteps(const SolutionStep &elem,
                                  ArrayRef<SolutionStep> matmulSteps) {
  const SetVector<Instruction *> *ops = stepElementwiseOps(elem);
  if (!ops)
    return false;
  for (const SolutionStep &m : matmulSteps)
    if (const auto *fp = stepMatmulFootprint(m))
      if (intersects(*fp, *ops))
        return true;
  return false;
}

SmallVector<SolutionStep>
errorBudgetSelector(SmallVector<CandidateMatmul, 4> &CMs, double budget,
                    double confidence) {
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
    // Inside the calibration noise band (-poseidon-cost-tie-band-rel) two costs
    // are a tie: among qualifying candidates within band of the cheapest, the
    // winner is the smallest modelled domainError, then the smallest compCost,
    // then the lowest index. The guard never widens the qualifying set; 0 keeps
    // strictly-cheapest-wins.
    if (best >= 0 && flags::CostTieBandRel > 0.0) {
      const double band =
          bestComp + flags::CostTieBandRel * std::fabs(bestComp);
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
      if (tieBest != best && flags::Print)
        llvm::errs() << "Poseidon error-budget: cost tie inside "
                     << (flags::CostTieBandRel * 100.0)
                     << "% calibration band ("
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
      if (flags::Print)
        llvm::errs() << "Poseidon error-budget: matmul -> baseline F64 (no "
                        "rewrite clears "
                     << budget << " at confidence " << confidence
                     << " faster than F64)\n";
      continue;
    }
    if (flags::Print)
      llvm::errs() << "Poseidon error-budget: matmul -> "
                   << matmulOptionLabel(cm.candidates[best]) << " (domain err "
                   << cm.candidates[best].domainError << " <= " << budget
                   << " at confidence " << confidence << ")\n";
    steps.emplace_back(&cm, (size_t)best);
  }
  return steps;
}

// Write the DP table's cost breakpoints to <cache>/budgets.txt for the Pareto-
// variant workflow. Written only if absent so parallel variant builds reusing a
// cached table don't clobber it.
static void writeBudgetsFileOnce(
    const std::map<InstructionCost, double> &costToAccuracyMap) {
  if (flags::Cache.empty())
    return;
  std::string budgetsFile = flags::Cache + "/budgets.txt";
  if (llvm::sys::fs::exists(budgetsFile))
    return;
  std::string budgetsStr;
  for (const auto &pair : costToAccuracyMap)
    budgetsStr += std::to_string(GET_INSTRUCTION_COST(pair.first)) + ",";
  if (!budgetsStr.empty())
    budgetsStr.pop_back();
  std::error_code EC;
  llvm::raw_fd_ostream Out(budgetsFile, EC, llvm::sys::fs::OF_Text);
  if (EC)
    llvm::errs() << "Error opening " << budgetsFile << ": " << EC.message()
                 << "\n";
  else
    Out << budgetsStr;
}

SmallVector<SolutionStep> accuracyDPSolver(
    Function &F, SmallVector<CandidateOutput, 4> &COs,
    SmallVector<CandidateSubgraph, 4> &CSs,
    SmallVector<CandidateMatmul, 4> &CMs,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap, double errorTol,
    double accScale) {
  llvm::errs() << "Starting accuracy DP solver with computation budget: "
               << flags::CompCostBudget << "\n";
  if (errorTol > 0.0) {
    llvm::errs() << "Per-operation relative error tolerance: " << errorTol
                 << "\n";
  }

  using CostMap = std::map<InstructionCost, double>;
  using SolutionMap = std::map<InstructionCost, SmallVector<SolutionStep>>;

  CostMap costToAccuracyMap;
  SolutionMap costToSolutionMap;

  std::string cacheFilePath = dpCachePath();
  bool loadedFromCache = false;
  std::optional<double> cachedBaseline;

  if (!cacheFilePath.empty() && llvm::sys::fs::exists(cacheFilePath) &&
      loadDPCache(cacheFilePath, F.getName(), COs, CSs, CMs, costToAccuracyMap,
                  costToSolutionMap, cachedBaseline)) {
    llvm::errs() << "Cache entry for " << F.getName()
                 << " found. Loaded DP tables from cache.\n";
    loadedFromCache = true;

  } else {
    llvm::errs() << "No cache entry for " << F.getName()
                 << ". Proceeding to solve DP.\n";

    costToAccuracyMap[0] = 0;
    costToSolutionMap[0] = {};

    int COCounter = 0;
    for (auto &CO : COs) {
      paretoIntegrate(
          costToAccuracyMap, costToSolutionMap, &CO, CO.candidates.size(),
          [&](size_t i, const auto &) { return CO.getCompCostDelta(i); },
          [&](size_t i, const auto &) { return CO.getAccCostDelta(i); });
      // TODO: Do not prune CO parts of the DP table since COs influence CSs
      if (flags::EarlyPrune)
        paretoPrune(costToAccuracyMap, costToSolutionMap);
      llvm::errs() << "##### Finished processing " << ++COCounter << " of "
                   << COs.size() << " COs #####\n";
      llvm::errs() << "Current DP table sizes: " << costToAccuracyMap.size()
                   << "\n";
    }

    int CMCounter = 0;
    for (auto &CM : CMs) {
      paretoIntegrate(
          costToAccuracyMap, costToSolutionMap, &CM, CM.candidates.size(),
          [&](size_t i, const auto &) { return CM.getCompCostDelta(i); },
          [&](size_t i, const auto &) { return CM.getAccCostDelta(i); });
      if (flags::EarlyPrune)
        paretoPrune(costToAccuracyMap, costToSolutionMap);
      llvm::errs() << "##### Finished processing " << ++CMCounter << " of "
                   << CMs.size() << " CMs #####\n";
      llvm::errs() << "Current DP table sizes: " << costToAccuracyMap.size()
                   << "\n";
    }

    int CSCounter = 0;
    for (auto &CS : CSs) {
      paretoIntegrate(
          costToAccuracyMap, costToSolutionMap, &CS, CS.candidates.size(),
          [&](size_t i, const auto &prior) {
            return CS.getAdjustedCompCostDelta(i, prior);
          },
          [&](size_t i, const auto &prior) {
            return CS.getAdjustedAccCostDelta(i, prior, valueToNodeMap,
                                              symbolToValueMap);
          });
      // CS always prunes (subsequent CSs see fewer dominated alternatives).
      paretoPrune(costToAccuracyMap, costToSolutionMap);
      llvm::errs() << "##### Finished processing " << ++CSCounter << " of "
                   << CSs.size() << " CSs #####\n";
      llvm::errs() << "Current DP table sizes: " << costToAccuracyMap.size()
                   << "\n";
    }

    // Inserted before the cache write so sweep compiles reusing the cached
    // table inherit these entries.

    if (!cacheFilePath.empty())
      writeDPCache(cacheFilePath, F.getName(), costToAccuracyMap,
                   costToSolutionMap, COs, CSs, CMs, baselineAccCost(COs, CSs));
  }

  if (flags::Print && flags::ShowTable)
    printDPTable(costToAccuracyMap, costToSolutionMap);

  writeBudgetsFileOnce(costToAccuracyMap);

  if (!loadedFromCache)
    emitReport(F.getName(), costToAccuracyMap, costToSolutionMap, COs, CSs);

  llvm::errs() << "Critical computation cost range: ["
               << costToAccuracyMap.begin()->first << ", "
               << costToAccuracyMap.rbegin()->first << "]\n";

  llvm::errs() << "DP table contains " << costToAccuracyMap.size()
               << " entries.\n";

  if (costToAccuracyMap.size() == 1) {
    size_t totalCands = 0;
    for (const auto &CO : COs)
      totalCands += CO.candidates.size();
    for (const auto &CS : CSs)
      totalCands += CS.candidates.size();
    for (const auto &CM : CMs)
      totalCands += CM.candidates.size();
    if (totalCands > 0)
      llvm::errs() << "[poseidon] NOTE: " << totalCands
                   << " candidate(s) synthesized but NONE lands on the Pareto "
                      "frontier (all dominated by the baseline under this "
                      "device's cost model); the no-op solution is genuinely "
                      "optimal here.\n";
  }

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
    checkToleranceInputs(F.getName(), accScale, baseline, loadedFromCache,
                         cacheFilePath);

    bool foundSolution = false;
    double bestRel = 0.0;

    for (const auto &pair : costToAccuracyMap) {
      InstructionCost compCost = pair.first;
      double accCost = pair.second;
      double rel = (baseline + accCost) / accScale;

      if (rel <= errorTol) {
        const bool better =
            flags::TauCheapest
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

      if (compCost <= flags::CompCostBudget && accCost < minAccCost) {
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

  return SmallVector<SolutionStep>(costToSolutionMap[bestCompCost].begin(),
                                   costToSolutionMap[bestCompCost].end());
}

SmallVector<SolutionStep>
jointAccuracyDPSolver(ArrayRef<FunctionFPState *> states, double errorTol) {
  llvm::errs() << "Starting JOINT accuracy DP solver over " << states.size()
               << " function(s) with shared computation budget: "
               << flags::CompCostBudget << "\n";

  using CostMap = std::map<InstructionCost, double>;
  using SolutionMap = std::map<InstructionCost, SmallVector<SolutionStep>>;
  CostMap costToAccuracyMap;
  SolutionMap costToSolutionMap;
  costToAccuracyMap[0] = 0;
  costToSolutionMap[0] = {};

  // Same integration order as accuracyDPSolver. The joint table is not cached
  // (table.json is per candidate set); budgets.txt is still written.
  for (auto *st : states)
    for (auto &CO : st->COs) {
      paretoIntegrate(
          costToAccuracyMap, costToSolutionMap, &CO, CO.candidates.size(),
          [&](size_t i, const auto &) { return CO.getCompCostDelta(i); },
          [&](size_t i, const auto &) { return CO.getAccCostDelta(i); });
      if (flags::EarlyPrune)
        paretoPrune(costToAccuracyMap, costToSolutionMap);
    }
  for (auto *st : states)
    for (auto &CM : st->CMs) {
      paretoIntegrate(
          costToAccuracyMap, costToSolutionMap, &CM, CM.candidates.size(),
          [&](size_t i, const auto &) { return CM.getCompCostDelta(i); },
          [&](size_t i, const auto &) { return CM.getAccCostDelta(i); });
      if (flags::EarlyPrune)
        paretoPrune(costToAccuracyMap, costToSolutionMap);
    }
  for (auto *st : states) {
    auto &valueToNodeMap = st->valueToNodeMap;
    auto &symbolToValueMap = st->symbolToValueMap;
    for (auto &CS : st->CSs) {
      paretoIntegrate(
          costToAccuracyMap, costToSolutionMap, &CS, CS.candidates.size(),
          [&](size_t i, const auto &prior) {
            return CS.getAdjustedCompCostDelta(i, prior);
          },
          [&](size_t i, const auto &prior) {
            return CS.getAdjustedAccCostDelta(i, prior, valueToNodeMap,
                                              symbolToValueMap);
          });
      paretoPrune(costToAccuracyMap, costToSolutionMap);
    }
  }

  llvm::errs() << "JOINT DP table contains " << costToAccuracyMap.size()
               << " entries; cost range [" << costToAccuracyMap.begin()->first
               << ", " << costToAccuracyMap.rbegin()->first << "]\n";

  writeBudgetsFileOnce(costToAccuracyMap);

  if (costToSolutionMap.count(0) && costToSolutionMap[0].empty())
    llvm::errs() << "WARNING: No-op solution (budget = 0) is Pareto-optimal.\n";

  if (costToAccuracyMap.size() == 1) {
    size_t totalCands = 0;
    for (auto *st : states) {
      for (const auto &CO : st->COs)
        totalCands += CO.candidates.size();
      for (const auto &CS : st->CSs)
        totalCands += CS.candidates.size();
      for (const auto &CM : st->CMs)
        totalCands += CM.candidates.size();
    }
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
    checkToleranceInputs(names, accScale, baseline,
                         /*loadedFromCache=*/false, StringRef());

    bool found = false;
    double bestRel = 0.0;
    for (const auto &pair : costToAccuracyMap) {
      double rel = (baseline + pair.second) / accScale;
      if (rel > errorTol)
        continue;
      const bool better =
          flags::TauCheapest
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
      if (pair.first <= flags::CompCostBudget && pair.second < minAccCost) {
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
  return SmallVector<SolutionStep>(costToSolutionMap[bestCompCost].begin(),
                                   costToSolutionMap[bestCompCost].end());
}

SmallVector<SolutionStep>
parseManualRewrites(StringRef spec, SmallVector<CandidateOutput, 4> &COs,
                    SmallVector<CandidateSubgraph, 4> &CSs,
                    SmallVector<CandidateMatmul, 4> &CMs) {
  SmallVector<SolutionStep> steps;
  SmallVector<StringRef> ids;
  spec.split(ids, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);

  SmallDenseSet<size_t> appliedCOs, appliedCSs, appliedCMs;
  for (auto id : ids) {
    id = id.trim();
    if (id.starts_with("R")) {
      auto rest = id.drop_front(1);
      auto [coStr, candStr] = rest.split('_');
      size_t coIdx, candIdx;
      if (coStr.getAsInteger(10, coIdx) || candStr.getAsInteger(10, candIdx))
        report_fatal_error("[poseidon] invalid rewrite ID '" + Twine(id) + "'");
      if (coIdx >= COs.size())
        report_fatal_error("[poseidon] CO index " + Twine(coIdx) +
                           " out of range (" + Twine(COs.size()) + " COs)");
      if (candIdx >= COs[coIdx].candidates.size())
        report_fatal_error("[poseidon] candidate index " + Twine(candIdx) +
                           " out of range for CO " + Twine(coIdx) + " (" +
                           Twine(COs[coIdx].candidates.size()) +
                           " candidates)");
      if (!appliedCOs.insert(coIdx).second)
        report_fatal_error("[poseidon] CO " + Twine(coIdx) +
                           " already has a rewrite applied (duplicate ID '" +
                           Twine(id) + "')");
      llvm::errs() << "[poseidon] Selecting " << id << ": " << COs[coIdx].expr
                   << " -> " << COs[coIdx].candidates[candIdx].expr << "\n";
      steps.emplace_back(&COs[coIdx], candIdx);
    } else if (id.starts_with("PT")) {
      auto rest = id.drop_front(2);
      auto [csStr, candStr] = rest.split('_');
      size_t csIdx, candIdx;
      if (csStr.getAsInteger(10, csIdx) || candStr.getAsInteger(10, candIdx))
        report_fatal_error("[poseidon] invalid PT ID '" + Twine(id) + "'");
      if (csIdx >= CSs.size())
        report_fatal_error("[poseidon] CS index " + Twine(csIdx) +
                           " out of range (" + Twine(CSs.size()) + " CSs)");
      if (candIdx >= CSs[csIdx].candidates.size())
        report_fatal_error("[poseidon] candidate index " + Twine(candIdx) +
                           " out of range for CS " + Twine(csIdx) + " (" +
                           Twine(CSs[csIdx].candidates.size()) +
                           " candidates)");
      if (!appliedCSs.insert(csIdx).second)
        report_fatal_error("[poseidon] CS " + Twine(csIdx) +
                           " already has a PT applied (duplicate ID '" +
                           Twine(id) + "')");
      llvm::errs() << "[poseidon] Selecting " << id << ": "
                   << CSs[csIdx].candidates[candIdx].desc << "\n";
      steps.emplace_back(&CSs[csIdx], candIdx);
    } else if (id.starts_with("M")) {
      auto rest = id.drop_front(1);
      auto [cmStr, candStr] = rest.split('_');
      size_t cmIdx, candIdx;
      if (cmStr.getAsInteger(10, cmIdx) || candStr.getAsInteger(10, candIdx))
        report_fatal_error("[poseidon] invalid matmul ID '" + Twine(id) + "'");
      if (cmIdx >= CMs.size())
        report_fatal_error("[poseidon] CM index " + Twine(cmIdx) +
                           " out of range (" + Twine(CMs.size()) + " CMs)");
      if (candIdx >= CMs[cmIdx].candidates.size())
        report_fatal_error("[poseidon] candidate index " + Twine(candIdx) +
                           " out of range for CM " + Twine(cmIdx) + " (" +
                           Twine(CMs[cmIdx].candidates.size()) +
                           " candidates)");
      if (!appliedCMs.insert(cmIdx).second)
        report_fatal_error("[poseidon] CM " + Twine(cmIdx) +
                           " already has a rewrite applied (duplicate ID '" +
                           Twine(id) + "')");
      llvm::errs() << "[poseidon] Selecting " << id << ": matmul["
                   << CMs[cmIdx].matmul->id << "] -> "
                   << matmulOptionLabel(CMs[cmIdx].candidates[candIdx]) << "\n";
      steps.emplace_back(&CMs[cmIdx], candIdx);
    } else {
      report_fatal_error("[poseidon] unknown ID prefix in '" + Twine(id) +
                         "' (expected R, PT, or M)");
    }
  }
  return steps;
}

bool applySolution(
    ArrayRef<SolutionStep> steps,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap) {
  if (steps.empty())
    return false;
  llvm::errs() << "\n!!! Applying solution (" << steps.size()
               << " step(s)) "
                  "... !!!\n";
  for (const auto &step : steps) {
    std::visit(
        [&](auto *item) {
          using T = std::decay_t<decltype(*item)>;
          if constexpr (std::is_same_v<T, CandidateOutput>) {
            llvm::errs() << "Applying solution for " << item->expr << " --("
                         << step.candidateIndex << ")-> "
                         << item->candidates[step.candidateIndex].expr << "\n";
            item->apply(step.candidateIndex, valueToNodeMap, symbolToValueMap);
          } else if constexpr (std::is_same_v<T, CandidateSubgraph>) {
            llvm::errs() << "Applying solution for CS: "
                         << item->candidates[step.candidateIndex].desc << " (#"
                         << step.candidateIndex << ")\n";
            item->apply(step.candidateIndex);
          } else if constexpr (std::is_same_v<T, CandidateMatmul>) {
            llvm::errs() << "Applying solution for matmul[" << item->matmul->id
                         << "] -> "
                         << matmulOptionLabel(
                                item->candidates[step.candidateIndex])
                         << " (#" << step.candidateIndex << ")\n";
            item->apply(step.candidateIndex);
          } else {
            llvm_unreachable("applySolution: unexpected SolutionStep variant");
          }
        },
        step.item);
  }
  llvm::errs() << "!!! Solution applied !!!\n\n";
  return true;
}

} // namespace poseidon
