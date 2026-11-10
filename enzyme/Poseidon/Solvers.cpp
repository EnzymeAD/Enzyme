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
#include "Flags.h"
#include "Optimize.h"
#include "Solvers.h"
#include "Types.h"

#include "Herbie.h"
#include "Matmul.h"
#include "Precision.h"
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

#if LLVM_VERSION_MAJOR >= 21
#define GET_INSTRUCTION_COST(cost) (cost.getValue())
#else
#define GET_INSTRUCTION_COST(cost) (cost.getValue().value())
#endif

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

template <typename CandT, typename TextFn>
static std::optional<size_t> findCandidate(ArrayRef<CandT> cands, size_t hint,
                                           StringRef key, TextFn text) {
  if (hint < cands.size() && candidateKey(text(cands[hint])) == key)
    return hint;
  std::optional<size_t> found;
  for (size_t i = 0; i < cands.size(); ++i) {
    if (candidateKey(text(cands[i])) != key)
      continue;
    if (found)
      return std::nullopt;
    found = i;
  }
  return found;
}

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
  // The binary64 and binary32 searches of one output are separate items, but
  // only one rewrite of that output can be materialized.
  for (const auto &step : priorSteps)
    if (auto *const *co = std::get_if<CandidateOutput *>(&step.item))
      if (cand && (*co)->sharesRoot(*cand))
        return false;
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

namespace {
struct DPUnit {
  decltype(SolutionStep::item) item;
  size_t numCandidates;
  std::function<InstructionCost(size_t, const SmallVectorImpl<SolutionStep> &)>
      comp;
  std::function<double(size_t, const SmallVectorImpl<SolutionStep> &)> acc;
};
} // namespace

static SolutionStep unitStep(const DPUnit &unit, size_t i) {
  return std::visit([&](auto *p) { return SolutionStep(p, i); }, unit.item);
}

static const void *itemKey(const decltype(SolutionStep::item) &item) {
  return std::visit([](auto *p) -> const void * { return p; }, item);
}

// One rewrite per output, and no instruction claimed both by a matrix-product
// raise and by another candidate.
static bool excludes(const SolutionStep &a, const SolutionStep &b) {
  auto *coA = std::get_if<CandidateOutput *>(&a.item);
  auto *coB = std::get_if<CandidateOutput *>(&b.item);
  if (coA && coB)
    return (*coA)->sharesRoot(**coB);
  SmallVector<SolutionStep, 1> other;
  CandidateMatmul *cm = stepMatmul(a);
  other.push_back(b);
  if (!cm) {
    cm = stepMatmul(b);
    other.front() = a;
  }
  return cm && conflictsWithChosen(other, cm);
}

// Whether taking `later` is admitted or priced differently depending on
// whether `earlier` was taken. Neither reads which candidate `earlier` took.
static bool reads(const SolutionStep &later, const SolutionStep &earlier) {
  if (excludes(later, earlier))
    return true;
  auto *cs = std::get_if<CandidateSubgraph *>(&later.item);
  auto *co = std::get_if<CandidateOutput *>(&earlier.item);
  return cs && co && (*cs)->subgraph == (*co)->subgraph;
}

// The binary64 and binary32 searches of one output, and any two cores with the
// same outputs, are read alike by every later unit.
static bool sameRewriteTarget(const CandidateOutput &a,
                              const CandidateOutput &b) {
  return a.subgraph == b.subgraph && a.roots.size() == b.roots.size() &&
         llvm::all_of(a.roots,
                      [&](Value *r) { return llvm::is_contained(b.roots, r); });
}

// -poseidon-early-prune=false. A partial solution is keyed by its cost and by
// its signature: the taken units some later unit still reads. Two with one
// signature are admitted and priced alike by every later unit, so one that is
// cheaper and no less accurate than the other stays so under every completion,
// and only such a one is dropped. After the last unit every signature is
// empty and the one remaining bucket is the frontier of all
// footprint-compatible strategies.
static void
exactParetoSolve(ArrayRef<DPUnit> units, std::map<InstructionCost, double> &cost,
                 std::map<InstructionCost, SmallVector<SolutionStep>> &sol) {
  const int n = units.size();
  SmallVector<int> lastRead(n, -1), canon(n);
  DenseMap<const void *, int> unitIndex;
  for (int u = 0; u < n; ++u) {
    unitIndex[itemKey(units[u].item)] = u;
    canon[u] = u;
    if (auto *co = std::get_if<CandidateOutput *>(&units[u].item))
      for (int w = 0; w < u; ++w)
        if (auto *other = std::get_if<CandidateOutput *>(&units[w].item))
          if (sameRewriteTarget(**other, **co)) {
            canon[u] = canon[w];
            break;
          }
    for (int v = u + 1; v < n; ++v)
      if (reads(unitStep(units[v], 0), unitStep(units[u], 0)))
        lastRead[canon[u]] = std::max(lastRead[canon[u]], v);
  }

  using Signature = std::vector<int>;
  struct Entry {
    double acc;
    SmallVector<SolutionStep> steps;
  };
  using Bucket = std::map<InstructionCost, Entry>;
  auto insert = [](Bucket &b, InstructionCost c, double a,
                   const SmallVectorImpl<SolutionStep> &steps) {
    auto [it, fresh] = b.try_emplace(c);
    if (!fresh && !(it->second.acc > a))
      return;
    it->second.acc = a;
    it->second.steps.assign(steps.begin(), steps.end());
  };
  // paretoPrune's rule: an entry goes when a cheaper one is no less accurate.
  auto prune = [](Bucket &b) {
    std::optional<double> best;
    for (auto it = b.begin(); it != b.end();) {
      const double a = it->second.acc;
      if (best && a >= *best) {
        it = b.erase(it);
        continue;
      }
      if (!std::isnan(a))
        best = best ? std::min(*best, a) : a;
      ++it;
    }
  };

  std::map<Signature, Bucket> table;
  insert(table[Signature()], cost.begin()->first, cost.begin()->second,
         sol.begin()->second);
  size_t peak = 1, peakSignatures = 1, pricings = 0;
  const auto start = std::chrono::steady_clock::now();
  for (int k = 0; k < n; ++k) {
    const DPUnit &unit = units[k];
    const SolutionStep step = unitStep(unit, 0);
    auto stillRead = [&](const Signature &sig) {
      Signature kept;
      for (int u : sig)
        if (lastRead[u] > k)
          kept.push_back(u);
      return kept;
    };
    auto *CS = std::get_if<CandidateSubgraph *>(&unit.item);
    const size_t memoBefore = CS ? (*CS)->compCostDeltaCache.size() : 0;

    std::map<Signature, Bucket> next;
    for (const auto &[sig, bucket] : table) {
      Bucket &into = next[stillRead(sig)];
      for (const auto &[c, e] : bucket)
        insert(into, c, e.acc, e.steps);
    }
    for (const auto &[sig, bucket] : table) {
      if (unit.numCandidates == 0)
        break;
      // Everything a unit reads is in the signature, so one entry stands for
      // the bucket.
      const SmallVector<SolutionStep> &rep = bucket.begin()->second.steps;
      if (llvm::any_of(rep,
                       [&](const SolutionStep &s) { return excludes(step, s); }))
        continue;
      SmallVector<SolutionStep> priced;
      for (const SolutionStep &s : rep) {
        int u = canon[unitIndex.find(itemKey(s.item))->second];
        priced.push_back(unitStep(units[u], s.candidateIndex));
      }
      SmallVector<InstructionCost> dc;
      SmallVector<double> da;
      for (size_t i = 0; i < unit.numCandidates; ++i) {
        dc.push_back(unit.comp(i, priced));
        da.push_back(unit.acc(i, priced));
      }
      Signature taken = stillRead(sig);
      if (lastRead[canon[k]] > k) {
        taken.push_back(canon[k]);
        llvm::sort(taken);
      }
      Bucket &into = next[taken];
      SmallVector<SolutionStep> steps;
      for (const auto &[c, e] : bucket) {
        steps.assign(e.steps.begin(), e.steps.end());
        steps.push_back(step);
        for (size_t i = 0; i < unit.numCandidates; ++i) {
          steps.back().candidateIndex = i;
          insert(into, c + dc[i], e.acc + da[i], steps);
        }
      }
    }
    size_t entries = 0;
    for (auto &[sig, bucket] : next) {
      prune(bucket);
      entries += bucket.size();
    }
    table = std::move(next);
    if (CS)
      pricings += (*CS)->compCostDeltaCache.size() - memoBefore;
    if (entries > peak) {
      peak = entries;
      peakSignatures = table.size();
    }
    llvm::errs() << "##### Exact DP: finished unit " << k + 1 << " of " << n
                 << ": " << entries << " entries in " << table.size()
                 << " signatures #####\n";
  }
  assert(table.size() == 1 && table.begin()->first.empty() &&
         "exact DP: a signature outlived the last unit");

  cost.clear();
  sol.clear();
  for (auto &[c, e] : table.begin()->second) {
    cost[c] = e.acc;
    sol[c] = std::move(e.steps);
  }
  const double secs =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  llvm::errs() << "Exact DP: " << cost.size() << " frontier points; peak "
               << peak << " entries in " << peakSignatures << " signatures; "
               << pricings << " precision changes priced; " << secs << " s\n";
}

static void appendMatmulUnits(SmallVectorImpl<DPUnit> &units,
                              MutableArrayRef<CandidateMatmul> CMs) {
  for (auto &CM : CMs)
    units.push_back(
        {&CM, CM.candidates.size(),
         [&CM](size_t i, const auto &) { return CM.getCompCostDelta(i); },
         [&CM](size_t i, const auto &) { return CM.getAccCostDelta(i); }});
}

// Each CS right after the COs on its subgraph, the only steps that price it,
// so they leave the signature there. The CMs come after every elementwise
// unit, so a footprint conflict is always a CM's refusal.
static void appendElementwiseUnits(
    SmallVectorImpl<DPUnit> &units, MutableArrayRef<CandidateOutput> COs,
    MutableArrayRef<CandidateSubgraph> CSs,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap) {
  auto appendCO = [&](CandidateOutput &CO) {
    units.push_back(
        {&CO, CO.candidates.size(),
         [&CO](size_t i, const auto &) { return CO.getCompCostDelta(i); },
         [&CO](size_t i, const auto &) { return CO.getAccCostDelta(i); }});
  };
  SmallPtrSet<const Subgraph *, 8> placed;
  for (auto &CS : CSs) {
    if (placed.insert(CS.subgraph).second)
      for (auto &CO : COs)
        if (CO.subgraph == CS.subgraph)
          appendCO(CO);
    units.push_back({&CS, CS.candidates.size(),
                     [&CS](size_t i, const auto &prior) {
                       return CS.getAdjustedCompCostDelta(i, prior);
                     },
                     [&CS, &valueToNodeMap,
                      &symbolToValueMap](size_t i, const auto &prior) {
                       return CS.getAdjustedAccCostDelta(
                           i, prior, valueToNodeMap, symbolToValueMap);
                     }});
  }
  for (auto &CO : COs)
    if (!placed.contains(CO.subgraph))
      appendCO(CO);
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

static std::string tablePath() { return flags::Cache + "/table.json"; }

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
static CacheKey solveKey(const FunctionFPState &st,
                         ArrayRef<CandidateMatmul> CMs, StringRef profilePath) {
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
  key.add("device", getCostModelNativeArch());
  key.add("cost model", utohexstr(getCostModelFingerprint()));

  SmallVector<std::string, 4> profilePaths = {profilePath.str()};
  for (const std::string &dir : flags::OutputProfiles) {
    SmallString<128> path(dir);
    sys::path::append(path, siteProfileStem(F) + ".fpprofile");
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

  for (const auto &[name, value] : solveFlagValues())
    key.add(("-" + name).str(), value);
  // Only a site that sent Herbie cores depends on which Herbie answered them.
  if (!st.COs.empty())
    addHerbieKeyParts(key);

  std::string candidates;
  raw_string_ostream os(candidates);
  for (const CandidateOutput &CO : st.COs) {
    os << "CO " << CO.herbiePrecision << " " << CO.expr << "\n";
    for (const RewriteCandidate &c : CO.candidates)
      os << "  " << c.expr << "\n";
  }
  for (const CandidateSubgraph &CS : st.CSs) {
    os << "CS\n";
    for (const PTCandidate &c : CS.candidates)
      os << "  " << c.desc << "\n";
  }
  for (const CandidateMatmul &CM : CMs) {
    os << "CM " << CM.matmul->id << "\n";
    for (const CandidateMatmul::Option &o : CM.candidates)
      os << "  " << matmulOptionLabel(o) << "\n";
  }
  key.addContent("candidate set", candidates);
  return key;
}

// Fills `table` from the cache when it holds a table for `fnName` under
// table.key, and otherwise says why not unless there is none at all.
static bool readSiteTable(StringRef fnName, SiteTable &table,
                          MutableArrayRef<CandidateOutput> COs,
                          MutableArrayRef<CandidateSubgraph> CSs,
                          MutableArrayRef<CandidateMatmul> CMs) {
  const std::string path = tablePath();
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
    if (s.itemType == "CO" && s.itemIndex < COs.size())
      if (auto i = findCandidate(
              ArrayRef(COs[s.itemIndex].candidates), s.candidateIndex,
              s.candidateKey,
              [](const RewriteCandidate &c) -> StringRef { return c.expr; }))
        return SolutionStep(&COs[s.itemIndex], *i);
    if (s.itemType == "CS" && s.itemIndex < CSs.size())
      if (auto i = findCandidate(
              ArrayRef(CSs[s.itemIndex].candidates), s.candidateIndex,
              s.candidateKey,
              [](const PTCandidate &c) -> StringRef { return c.desc; }))
        return SolutionStep(&CSs[s.itemIndex], *i);
    if (s.itemType == "CM" && s.itemIndex < CMs.size() &&
        s.candidateIndex < CMs[s.itemIndex].candidates.size())
      return SolutionStep(&CMs[s.itemIndex], s.candidateIndex);
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
writeSiteTable(StringRef fnName, const CacheKey &key,
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
      json::Object stepObj;
      stepObj["candidateIndex"] = static_cast<int64_t>(step.candidateIndex);

      std::visit(
          [&](auto *item) {
            using T = std::decay_t<decltype(*item)>;
            if constexpr (std::is_same_v<T, CandidateOutput>) {
              stepObj["itemType"] = "CO";
              stepObj["itemIndex"] = static_cast<int64_t>(coPtrToIndex[item]);
              stepObj["candidateKey"] =
                  candidateKey(item->candidates[step.candidateIndex].expr);
            } else if constexpr (std::is_same_v<T, CandidateSubgraph>) {
              stepObj["itemType"] = "CS";
              stepObj["itemIndex"] = static_cast<int64_t>(csPtrToIndex[item]);
              stepObj["candidateKey"] =
                  candidateKey(item->candidates[step.candidateIndex].desc);
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

  // Read-modify-write: the file holds one table per function, so a later site
  // in the same compilation unit must extend it rather than clobber the
  // earlier sites' tables. A file that does not parse is replaced.
  const std::string path = tablePath();
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
  Module *M = st.F->getParent();
  for (CandidateOutput &CO : st.COs) {
    if (!CO.herbieAnswered)
      continue;
    const FastMathFlags FMF =
        cast<Instruction>(CO.oldOutput)->getFastMathFlags();
    for (RewriteCandidate &c : CO.candidates)
      c.CompCost = getCompCost(c.expr, M, st.valueToNodeMap,
                               st.symbolToValueMap, FMF, &CO.subgraph->inputs);
    setUnifiedAccuracyCost(CO, st.valueToNodeMap, st.symbolToValueMap);
  }
  for (CandidateSubgraph &CS : st.CSs) {
    for (PTCandidate &c : CS.candidates)
      c.CompCost = getCompCost(*CS.subgraph, c);
    setUnifiedAccuracyCost(CS, st.valueToNodeMap, st.symbolToValueMap);
  }
}

void priceUnlessCached(FunctionFPState &st, StringRef profilePath) {
  // A joint or a manual solve reads no table, and an accuracy target leaves
  // the matrix products to errorBudgetSelector (fpOptimize).
  if (!flags::Cache.empty() && !flags::JointDP &&
      flags::ApplyRewrites.empty()) {
    const bool target = st.errTol > 0.0 || flags::Tau > 0.0;
    MutableArrayRef<CandidateMatmul> CMs;
    if (!target)
      CMs = st.CMs;
    st.table = std::make_unique<SiteTable>();
    st.table->key = solveKey(st, CMs, profilePath);
    if (readSiteTable(st.F->getName(), *st.table, st.COs, st.CSs, CMs))
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
    for (const auto &step : it->second) {
      std::visit(
          [&](auto *item) {
            using T = std::decay_t<decltype(*item)>;
            if constexpr (std::is_same_v<T, CandidateOutput>) {
              const RewriteCandidate &cand =
                  item->candidates[step.candidateIndex];
              llvm::errs() << "\t\t" << item->expr << " --(" << cand.ordinal
                           << ")-> " << cand.expr << "\n";
            } else if constexpr (std::is_same_v<T, CandidateSubgraph>) {
              const PTCandidate &cand = item->candidates[step.candidateIndex];
              llvm::errs() << "\t\tCS: " << cand.desc << " (#" << cand.ordinal
                           << ")\n";
            } else if constexpr (std::is_same_v<T, CandidateMatmul>) {
              llvm::errs() << "\t\tmatmul[" << item->matmul->id << "] -> "
                           << matmulOptionLabel(
                                  item->candidates[step.candidateIndex])
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
          stepObj["candidate_index"] = static_cast<int64_t>(cand.ordinal);

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
          OS << pad << "[Precision] " << cand.desc << " (#" << cand.ordinal
             << ")\n";
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
      c["id"] = "PT" + std::to_string(csIdx) + "_" +
                std::to_string(CS.candidates[ci].ordinal);
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
      textOut << "    [" << cand.ordinal << "] " << cand.expr;
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
      textOut << "  [PT" << csIdx << "_" << CS.candidates[ci].ordinal << "] "
              << CS.candidates[ci].desc
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
        std::string id =
            "R" + std::to_string(coIdx) + "_" + std::to_string(cand.ordinal);
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
            "PT" + std::to_string(csIdx) + "_" + std::to_string(pt.ordinal);
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
      if (tieBest != best && flags::Print)
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
// variant workflow, on every solve and replay. Parallel variant builds reusing
// a cached table write the same breakpoints, and the replace is atomic, so
// none of them reads a partial file.
static void
writeBudgetsFile(const std::map<InstructionCost, double> &costToAccuracyMap) {
  if (flags::Cache.empty())
    return;
  std::string budgetsFile = flags::Cache + "/budgets.txt";
  std::string budgetsStr;
  for (const auto &pair : costToAccuracyMap)
    budgetsStr += std::to_string(GET_INSTRUCTION_COST(pair.first)) + ",";
  if (!budgetsStr.empty())
    budgetsStr.pop_back();
  if (Error E = writeFileAtomically(budgetsFile, budgetsStr))
    llvm::errs() << "Error writing " << budgetsFile << ": "
                 << toString(std::move(E)) << "\n";
}

SmallVector<SolutionStep> accuracyDPSolver(
    Function &F, SmallVector<CandidateOutput, 4> &COs,
    SmallVector<CandidateSubgraph, 4> &CSs,
    SmallVector<CandidateMatmul, 4> &CMs,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap, double errorTol,
    double accScale, SolveRecord *record, SiteTable *table) {
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

    if (!flags::EarlyPrune) {
      SmallVector<DPUnit> units;
      appendElementwiseUnits(units, COs, CSs, valueToNodeMap, symbolToValueMap);
      appendMatmulUnits(units, CMs);
      exactParetoSolve(units, costToAccuracyMap, costToSolutionMap);
    } else {
      // Each prune also compares partial solutions a later unit treats
      // differently (a CS is priced against the COs chosen on its subgraph),
      // so the frontier can miss points the exact solve finds.
      int COCounter = 0;
      for (auto &CO : COs) {
        paretoIntegrate(
            costToAccuracyMap, costToSolutionMap, &CO, CO.candidates.size(),
            [&](size_t i, const auto &) { return CO.getCompCostDelta(i); },
            [&](size_t i, const auto &) { return CO.getAccCostDelta(i); });
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
        paretoPrune(costToAccuracyMap, costToSolutionMap);
        llvm::errs() << "##### Finished processing " << ++CSCounter << " of "
                     << CSs.size() << " CSs #####\n";
        llvm::errs() << "Current DP table sizes: " << costToAccuracyMap.size()
                     << "\n";
      }
    }

    if (table)
      writeSiteTable(F.getName(), table->key, costToAccuracyMap,
                     costToSolutionMap, COs, CSs, CMs,
                     baselineAccCost(COs, CSs));
  }

  if (flags::Print && flags::ShowTable)
    printDPTable(costToAccuracyMap, costToSolutionMap);

  writeBudgetsFile(costToAccuracyMap);

  if (!loadedFromCache)
    emitReport(F.getName(), costToAccuracyMap, costToSolutionMap, COs, CSs);

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
    checkToleranceInputs(F.getName(), accScale, baseline);

    bool foundSolution = false;
    double bestRel = 0.0;

    for (const auto &pair : costToAccuracyMap) {
      InstructionCost compCost = pair.first;
      double accCost = pair.second;
      double rel = (baseline + accCost) / accScale;

      if (std::isfinite(accCost) && accCost < flags::ExponentPenalty &&
          rel <= errorTol) {
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

      if (compCost <= flags::CompCostBudget && std::isfinite(accCost) &&
          accCost < flags::ExponentPenalty && accCost < minAccCost) {
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
jointAccuracyDPSolver(ArrayRef<FunctionFPState *> states, double errorTol,
                      SolveRecord *record) {
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
  if (!flags::EarlyPrune) {
    SmallVector<DPUnit> units;
    for (auto *st : states)
      appendElementwiseUnits(units, st->COs, st->CSs, st->valueToNodeMap,
                             st->symbolToValueMap);
    for (auto *st : states)
      appendMatmulUnits(units, st->CMs);
    exactParetoSolve(units, costToAccuracyMap, costToSolutionMap);
  } else {
    for (auto *st : states)
      for (auto &CO : st->COs) {
        paretoIntegrate(
            costToAccuracyMap, costToSolutionMap, &CO, CO.candidates.size(),
            [&](size_t i, const auto &) { return CO.getCompCostDelta(i); },
            [&](size_t i, const auto &) { return CO.getAccCostDelta(i); });
        paretoPrune(costToAccuracyMap, costToSolutionMap);
      }
    for (auto *st : states)
      for (auto &CM : st->CMs) {
        paretoIntegrate(
            costToAccuracyMap, costToSolutionMap, &CM, CM.candidates.size(),
            [&](size_t i, const auto &) { return CM.getCompCostDelta(i); },
            [&](size_t i, const auto &) { return CM.getAccCostDelta(i); });
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
  }

  llvm::errs() << "JOINT DP table contains " << costToAccuracyMap.size()
               << " entries; cost range [" << costToAccuracyMap.begin()->first
               << ", " << costToAccuracyMap.rbegin()->first << "]\n";

  writeBudgetsFile(costToAccuracyMap);

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
    checkToleranceInputs(names, accScale, baseline);

    bool found = false;
    double bestRel = 0.0;
    for (const auto &pair : costToAccuracyMap) {
      double rel = (baseline + pair.second) / accScale;
      if (!std::isfinite(pair.second) ||
          pair.second >= flags::ExponentPenalty || rel > errorTol)
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
  if (record)
    record->selected = GET_INSTRUCTION_COST(bestCompCost);
  return SmallVector<SolutionStep>(costToSolutionMap[bestCompCost].begin(),
                                   costToSolutionMap[bestCompCost].end());
}

// The position in `cands` of the candidate an id's ordinal names; pricing may
// have dropped it.
template <typename CandT>
static std::optional<size_t> findOrdinal(ArrayRef<CandT> cands,
                                         size_t ordinal) {
  for (size_t i = 0; i < cands.size(); ++i)
    if (cands[i].ordinal == ordinal)
      return i;
  return std::nullopt;
}

namespace {
enum class UnitKind { R, PT, M };

// One item of -poseidon-apply-rewrites: [<site>:]<unit>_<ordinal> or
// [<site>:]<unit>=<name>.
struct ApplyItem {
  std::string text;
  std::string site;
  UnitKind kind = UnitKind::R;
  size_t unit = 0;
  std::optional<size_t> ordinal;
  std::string name;
  // The keys of the module's sites `site` named. Sites are solved as the pass
  // reaches them, so that it named none is only known at the end of the
  // module.
  std::set<std::string> named;
};

struct PinState {
  std::string spec;
  std::vector<ApplyItem> items;
  std::vector<std::string> sites;
  // Two sites cut from one body share a key; printing their pin twice would
  // give a value that applies each item twice.
  std::set<std::string> printed;
};
} // namespace

static std::optional<ApplyItem> parseUnitRef(StringRef s) {
  ApplyItem item;
  if (s.consume_front("PT"))
    item.kind = UnitKind::PT;
  else if (s.consume_front("R"))
    item.kind = UnitKind::R;
  else if (s.consume_front("M"))
    item.kind = UnitKind::M;
  else
    return std::nullopt;
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

static PinState &pinState() {
  static PinState state;
  return state;
}

static std::vector<ApplyItem> &applyItems() {
  PinState &P = pinState();
  if (P.spec != flags::ApplyRewrites) {
    P.spec = flags::ApplyRewrites;
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
void noteSiteForPins(const Function &body) {
  if (flags::ApplyRewrites.empty())
    return;
  PinState &P = pinState();
  std::string key = siteKeyOfBody(body);
  if (is_contained(P.sites, key))
    return;
  P.sites.push_back(key);
  SmallVector<std::string, 2> aliases = siteAliases(key);
  for (ApplyItem &item : applyItems()) {
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

void checkPinnedSites() {
  PinState &P = pinState();
  // The host half of a CUDA or HIP unit is given the items too, and the sites
  // they name are the device half's.
  if (!P.sites.empty())
    for (const ApplyItem &item : applyItems())
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
template <typename CandT, typename NameFn>
static size_t findNamed(const ApplyItem &item, StringRef site, StringRef unit,
                        ArrayRef<CandT> cands,
                        ArrayRef<std::pair<std::string, std::string>> dropped,
                        NameFn nameOf) {
  std::optional<size_t> found;
  for (size_t i = 0; i < cands.size(); ++i) {
    if (nameOf(cands[i]) != item.name)
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
  for (const auto &[name, why] : dropped)
    if (name == item.name)
      report_fatal_error("[poseidon] " + Twine(site) + ": " + unit +
                             " has no candidate named '" + item.name +
                             "': pricing dropped it, because " + why,
                         /*gen_crash_diag=*/false);
  std::string names;
  for (const CandT &c : cands)
    names += "\n  " + nameOf(c);
  if (names.empty())
    names = " none";
  report_fatal_error("[poseidon] " + Twine(site) + ": " + unit +
                         " has no candidate named '" + item.name +
                         "'. Its candidates are:" + names,
                     /*gen_crash_diag=*/false);
}

SmallVector<SolutionStep> parseManualRewrites(FunctionFPState &st) {
  auto &COs = st.COs;
  auto &CSs = st.CSs;
  auto &CMs = st.CMs;
  const std::string key = siteKey(*st.F);
  const SmallVector<std::string, 2> aliases = siteAliases(key);
  SmallVector<SolutionStep> steps;
  SmallDenseSet<size_t> appliedCOs, appliedCSs, appliedCMs;
  for (const ApplyItem &item : applyItems()) {
    if (!namesSite(item, key, aliases))
      continue;
    const std::string &id = item.text;
    switch (item.kind) {
    case UnitKind::R: {
      const size_t coIdx = item.unit;
      if (coIdx >= COs.size())
        report_fatal_error("[poseidon] CO index " + Twine(coIdx) +
                           " out of range (" + Twine(COs.size()) + " COs)");
      CandidateOutput &CO = COs[coIdx];
      size_t pos;
      if (item.ordinal) {
        std::optional<size_t> found =
            findOrdinal(ArrayRef(CO.candidates), *item.ordinal);
        if (!found)
          report_fatal_error("[poseidon] CO " + Twine(coIdx) +
                             " has no candidate " + Twine(*item.ordinal) +
                             " that pricing kept ('" + Twine(id) + "')");
        pos = *found;
      } else {
        pos = findNamed(item, key, "R" + std::to_string(coIdx),
                        ArrayRef(CO.candidates), CO.dropped,
                        [](const RewriteCandidate &c) { return c.expr; });
      }
      if (!appliedCOs.insert(coIdx).second)
        report_fatal_error("[poseidon] CO " + Twine(coIdx) +
                           " already has a rewrite applied (duplicate ID '" +
                           Twine(id) + "')");
      llvm::errs() << "[poseidon] Selecting " << id << ": " << CO.expr << " -> "
                   << CO.candidates[pos].expr << "\n";
      steps.emplace_back(&CO, pos);
      break;
    }
    case UnitKind::PT: {
      const size_t csIdx = item.unit;
      if (csIdx >= CSs.size())
        report_fatal_error("[poseidon] CS index " + Twine(csIdx) +
                           " out of range (" + Twine(CSs.size()) + " CSs)");
      CandidateSubgraph &CS = CSs[csIdx];
      size_t pos;
      if (item.ordinal) {
        std::optional<size_t> found =
            findOrdinal(ArrayRef(CS.candidates), *item.ordinal);
        if (!found)
          report_fatal_error("[poseidon] CS " + Twine(csIdx) +
                             " has no candidate " + Twine(*item.ordinal) +
                             " that pricing kept ('" + Twine(id) + "')");
        pos = *found;
      } else {
        pos = findNamed(item, key, "PT" + std::to_string(csIdx),
                        ArrayRef(CS.candidates), CS.dropped,
                        [](const PTCandidate &c) { return c.desc; });
      }
      if (!appliedCSs.insert(csIdx).second)
        report_fatal_error("[poseidon] CS " + Twine(csIdx) +
                           " already has a PT applied (duplicate ID '" +
                           Twine(id) + "')");
      llvm::errs() << "[poseidon] Selecting " << id << ": "
                   << CS.candidates[pos].desc << "\n";
      steps.emplace_back(&CS, pos);
      break;
    }
    case UnitKind::M: {
      const size_t cmIdx = item.unit;
      if (cmIdx >= CMs.size())
        report_fatal_error("[poseidon] CM index " + Twine(cmIdx) +
                           " out of range (" + Twine(CMs.size()) + " CMs)");
      CandidateMatmul &CM = CMs[cmIdx];
      size_t pos;
      if (item.ordinal) {
        pos = *item.ordinal;
        if (pos >= CM.candidates.size())
          report_fatal_error("[poseidon] candidate index " + Twine(pos) +
                             " out of range for CM " + Twine(cmIdx) + " (" +
                             Twine(CM.candidates.size()) + " candidates)");
      } else {
        pos = findNamed(item, key, "M" + std::to_string(cmIdx),
                        ArrayRef(CM.candidates), {},
                        [](const CandidateMatmul::Option &o) {
                          return matmulOptionLabel(o);
                        });
      }
      if (!appliedCMs.insert(cmIdx).second)
        report_fatal_error("[poseidon] CM " + Twine(cmIdx) +
                           " already has a rewrite applied (duplicate ID '" +
                           Twine(id) + "')");
      llvm::errs() << "[poseidon] Selecting " << id << ": matmul["
                   << CM.matmul->id << "] -> "
                   << matmulOptionLabel(CM.candidates[pos]) << "\n";
      steps.emplace_back(&CM, pos);
      break;
    }
    }
  }
  return steps;
}

void printPin(const FunctionFPState &st, ArrayRef<SolutionStep> applied) {
  const std::string key = siteKey(*st.F);
  std::string value;
  for (const SolutionStep &step : applied) {
    std::string item = std::visit(
        [&](auto *unit) -> std::string {
          using T = std::decay_t<decltype(*unit)>;
          if constexpr (std::is_same_v<T, CandidateOutput>)
            return "R" + std::to_string(unit - st.COs.data()) + "=" +
                   unit->candidates[step.candidateIndex].expr;
          else if constexpr (std::is_same_v<T, CandidateSubgraph>)
            return "PT" + std::to_string(unit - st.CSs.data()) + "=" +
                   unit->candidates[step.candidateIndex].desc;
          else
            return "M" + std::to_string(unit - st.CMs.data()) + "=" +
                   matmulOptionLabel(unit->candidates[step.candidateIndex]);
        },
        step.item);
    if (!value.empty())
      value += ",";
    value += key + ":" + item;
  }
  if (!value.empty() && !pinState().printed.insert(value).second)
    return;
  llvm::errs() << "[poseidon] pin: " << value << "\n";
}

bool applySolution(
    ArrayRef<SolutionStep> steps,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap,
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
    std::visit(
        [&](auto *item) {
          using T = std::decay_t<decltype(*item)>;
          if constexpr (std::is_same_v<T, CandidateOutput>) {
            const RewriteCandidate &cand =
                item->candidates[step.candidateIndex];
            llvm::errs() << "Applying solution for " << item->expr << " --("
                         << cand.ordinal << ")-> " << cand.expr << "\n";
            if (llvm::any_of(item->roots, [&](Value *root) {
                  return rewrittenOutputs.contains(root);
                })) {
              llvm::errs() << "[poseidon] output already rewritten by an "
                              "earlier step; skipping this one\n";
              return;
            }
            rewrittenOutputs.insert(item->roots.begin(), item->roots.end());
            item->apply(step.candidateIndex, valueToNodeMap, symbolToValueMap);
          } else if constexpr (std::is_same_v<T, CandidateSubgraph>) {
            const PTCandidate &cand = item->candidates[step.candidateIndex];
            llvm::errs() << "Applying solution for CS: " << cand.desc << " (#"
                         << cand.ordinal << ")\n";
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
          applied.push_back(step);
        },
        step.item);
  }
  llvm::errs() << "!!! Solution applied !!!\n\n";
  return true;
}

} // namespace poseidon
