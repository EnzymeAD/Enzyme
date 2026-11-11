#include <llvm/Config/llvm-config.h>

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Dominators.h"

#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Verifier.h"

#include "llvm/Passes/PassBuilder.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "CostModel.h"
#include "Evaluators.h"
#include "Herbie.h"
#include "Optimize.h"
#include "Pass/Explain.h"
#include "Pass/Session.h"
#include "Profile/ProfileRead.h"
#include "Sites/Canonicalize.h"
#include "Sites/SiteIdentity.h"
#include "Solvers.h"
#include "Subgraphs.h"
#include "Targets/DeviceMath.h"
#include "Targets/Target.h"
#include "Techniques/Expansion.h"
#include "Techniques/Matmul/HostDispatch.h"
#include "Techniques/Matmul/Matmul.h"
#include "Techniques/Matmul/RaiseWMMA.h"
#include "Techniques/Precision.h"
#include "Techniques/Staging.h"
#include "Types.h"
#include "Utils.h"

using namespace llvm;
#ifdef DEBUG_TYPE
#undef DEBUG_TYPE
#endif
#define DEBUG_TYPE "poseidon"

namespace poseidon {

namespace {

// What collectFPCandidates reads from the site's profile.
struct SiteProfileData {
  std::string path;
  ProfileMap map;
  FunctionProfileHeader header;
  std::unordered_set<size_t> zeroGradIdx;
};

} // namespace

// Reads the site's profile, checked against the canonical form its slots were
// numbered against.
static void readSiteProfile(Session &S, Function &F, SiteProfileData &P) {
  const Options &O = S.Opts;
  assert(!O.Profiling.ProfileUse.empty());
  SmallString<128> profilePathBuf(O.Profiling.ProfileUse);
  llvm::sys::path::append(profilePathBuf, siteProfileStem(S, F) + ".fpprofile");
  P.path = profilePathBuf.str().str();
  const std::string &profilePath = P.path;

  if (!O.Herbie.Cache.empty()) {
    if (auto EC = llvm::sys::fs::create_directories(O.Herbie.Cache, true))
      llvm::errs() << "Warning: Could not create cache directory: "
                   << EC.message() << "\n";
  }

  // Names the site by what it computes, not by the kernel it was cut from: the
  // profile is only readable against the canonical form it was numbered
  // against.
  const std::string canonicalHash = poseidon::canonicalFormHash(F);
  if (O.Diagnostics.Print)
    llvm::errs() << "[poseidon] canonical form hash of " << F.getName()
                 << " is " << canonicalHash << "\n";

  auto &profileMap = P.map;
  auto &profileHeader = P.header;
  auto &zeroGradIdx = P.zeroGradIdx;
  if (!profilePath.empty()) {
    parseProfileFile(profilePath, profileMap, &profileHeader);
    if (profileMap.empty()) {
      llvm::errs() << "Warning: No profile data found in " << profilePath
                   << "\n";
    }
    if (!O.Profiling.OutputProfiles.empty())
      mergeOutputProfiles(profileMap, O.Profiling.OutputProfiles,
                          siteProfileStem(S, F) + ".fpprofile", canonicalHash,
                          F.getName());
    // Before the floor below, which would hide the exact zeros.
    if (O.Solver.AggressiveDCE)
      for (const auto &kv : profileMap) {
        const ProfileInfo &p = kv.second;
        if ((p.sumAbsGrad >= 0.0 ? p.sumAbsGrad : p.sumGrad) == 0.0)
          zeroGradIdx.insert(kv.first);
      }
    // Degenerate-adjoint guard, before anything reads a gradient: the floor is
    // written back into the profile records so the node table, the Herbie
    // candidate pricing, the precision-tuning per-output costs and the
    // scalar-reduction matmul gradD cells all see one consistent weight.
    applySensitivityFloor(profileMap, F.getName());
    requireCanonicalHash(profileHeader, canonicalHash, profilePath,
                         F.getName());
    if (O.Diagnostics.Print && profileHeader.launchCount > 0) {
      llvm::errs() << "[poseidon] profile header for " << F.getName()
                   << ": MaxBlockDims=(" << profileHeader.maxBlockDim[0] << ","
                   << profileHeader.maxBlockDim[1] << ","
                   << profileHeader.maxBlockDim[2]
                   << ") LaunchCount=" << profileHeader.launchCount << "\n";
    }
  }
}

// The site's matrix products, with the analyses their raise keeps in `st`.
static void recognizeMatrixProducts(Session &S, Function &F,
                                    const SiteProfileData &P,
                                    FunctionFPState &st) {
  const Options &O = S.Opts;
  const Target &T = Target::of(F);
  const auto &profileMap = P.map;
  const auto &profileHeader = P.header;
  auto &abstractMatmuls = st.abstractMatmuls;

  // These stash a ScalarEvolution* (and const SCEV* pointers it owns) into each
  // ScalarLoopHandle for the materializer's SCEVExpander, dereferenced during
  // materialization, so they must outlive it and live in `st`. Declared SE-last
  // so it is destroyed first (it references the others).
  auto &raiseDT = st.raiseDT;
  auto &raiseLI = st.raiseLI;
  auto &raiseAC = st.raiseAC;
  auto &raiseTLII = st.raiseTLII;
  auto &raiseTLI = st.raiseTLI;
  auto &raiseSE = st.raiseSE;
  if (T.isDevice() && O.Candidates.RaiseWMMA) {
    raiseDT.emplace(F);
    raiseLI.emplace(*raiseDT);
    raiseAC.emplace(F);
    raiseTLII.emplace(Triple(F.getParent()->getTargetTriple()));
    raiseTLI.emplace(*raiseTLII, &F);
    raiseSE.emplace(F, *raiseTLI, *raiseAC, *raiseDT, *raiseLI);
    // The constant-trip products are the in-kernel raise's.
    if (T.hasInKernelMatrixRaise())
      findScalarLoopMatmuls(O, F, *raiseSE, *raiseLI, profileHeader,
                            abstractMatmuls);
    // Runtime-shape dense GEMMs (finite-element partial-assembly reduces). Runs
    // after the constant-trip recognizer and skips whatever that one claimed,
    // so previously recognized sites keep their exact candidate sets.
    findHostGemmLoopNests(O, F, *raiseSE, *raiseLI, profileHeader, profileMap,
                          abstractMatmuls);
  }

  if (O.Diagnostics.Print && !abstractMatmuls.empty()) {
    llvm::errs() << "[poseidon] Found " << abstractMatmuls.size()
                 << " AbstractMatmul(s) for " << F.getName() << "\n";
    auto dimStr = [](unsigned d) -> std::string {
      return d == ~0u ? std::string("?") : std::to_string(d);
    };
    for (const auto &m : abstractMatmuls) {
      llvm::errs() << "  Matmul[" << m.id << "]: " << dimStr(m.M) << "x"
                   << dimStr(m.N) << "x" << dimStr(m.K)
                   << " a=" << fpKindName(m.aType)
                   << " b=" << fpKindName(m.bType)
                   << " acc=" << fpKindName(m.accType)
                   << " d=" << fpKindName(m.dType) << " origin="
                   << (m.origin == AbstractMatmul::Origin::HostGemmLoopNest
                           ? "HostGemmLoopNest"
                           : "ScalarLoopReduction")
                   << "\n";
      if (m.origin == AbstractMatmul::Origin::HostGemmLoopNest)
        llvm::errs() << "    (profile-scale shape; runtime descriptor printed "
                        "by the [hostgemm] recognizer above)\n";
      else
        llvm::errs() << "    (profile dump skipped for ScalarLoopReduction)\n";
    }
  }
}

// Each technique's candidates for every FP subgraph, one subgraph at a time:
// Herbie's rewrites into st.COs, precision assignments into st.CSs.
static void
proposeSubgraphCandidates(Session &S, Function &F,
                          const FunctionProfileHeader &profileHeader,
                          bool expansionSafe, FunctionFPState &st) {
  const Options &O = S.Opts;
  auto &subgraphs = st.subgraphs;
  int symbolCounter = 0;
  int subgraphCounter = 0;

  for (auto &subgraph : subgraphs) {
    assert(subgraph.inputs.size() > 0 && "No inputs found for subgraph");

    if (O.Candidates.EnableHerbie)
      proposeHerbieCandidates(S, F, subgraph, subgraphCounter, symbolCounter,
                              st.valueToNodeMap, st.symbolToValueMap, st.COs);

    if (O.Candidates.EnablePT)
      proposePrecisionCandidates(S, F, subgraph, st.valueToNodeMap,
                                 profileHeader, expansionSafe, st.CSs);
    llvm::errs() << "##### Finished synthesizing candidates for "
                 << ++subgraphCounter << " of " << subgraphs.size()
                 << " subgraphs! #####\n";
  }
}

// Put this site's error on the application's scale. kappa is the measured
// response of the declared quantity of interest to computing this site with a
// relative error eps on every operation, and the profile's own first-order
// prediction for the same noise is eps times the site's total sensitivity, in
// the units the accuracy costs are already in; dividing the two cancels the
// within-site propagation that both carry and leaves the downstream condition
// per unit of accuracy cost. A site whose probe run diverged gets a large
// finite factor instead, so no candidate of it is ever traded against
// precision.
static void putOnApplicationScale(Session &S, Function &F,
                                  const SiteProfileData &P,
                                  FunctionFPState &st) {
  const Options &O = S.Opts;
  const auto &profileMap = P.map;
  const auto &profileHeader = P.header;

  double siteSumSens = 0.0;
  for (const auto &kv : profileMap)
    siteSumSens += kv.second.sumSens;
  // Per-output weights are already divided by each output's own total.
  if (!O.Profiling.OutputProfiles.empty()) {
    if (profileHeader.hasKappa)
      report_fatal_error(
          Twine("Poseidon: the profile of ") + F.getName() +
          " carries a measured condition number, which scales one site-wide "
          "error; it cannot be combined with -poseidon-output-profiles.");
    siteSumSens = 1.0;
  }

  double factor = 1.0;
  if (profileHeader.hasKappa) {
    double sumSens = siteSumSens;
    if (sumSens == 0.0)
      report_fatal_error(
          Twine("Poseidon: the profile of ") + F.getName() +
          " carries a measured condition number but a zero total "
          "sensitivity, "
          "so the accuracy costs cannot be put on the scale the condition "
          "number was measured against; re-run profile generation.");
    factor = profileHeader.kappaMark == "diverged"
                 ? 1e30
                 : profileHeader.kappa / sumSens;
    llvm::errs() << "[poseidon] site " << F.getName()
                 << ": kappa=" << profileHeader.kappa << " ("
                 << profileHeader.kappaMark << ") sumsens=" << sumSens
                 << " factor=" << factor << "\n";
  }
  // The accuracy costs below are scaled by `factor`, so the level a tolerance
  // is compared against has to be scaled the same way; with a kappa this
  // reduces to kappa itself, without one to the site's total sensitivity.
  st.accScale = factor * siteSumSens;
  if (factor != 1.0)
    for (SiteUnit *unit : st.units)
      unit->scaleErrors(factor);
}

// What the profiled run spent in `F`: every operation the profile records, its
// execution count times its price.
static double profiledCost(CostModelState &CM, const Function &F,
                           const ProfileMap &profile) {
  double cost = 0.0;
  for (const Instruction &I : instructions(F)) {
    size_t slot;
    if (!tryReadProfIdxMetadata(&I, slot))
      continue;
    auto it = profile.find(slot);
    if (it != profile.end())
      cost += (double)it->second.exec * getInstructionCompCost(CM, &I);
  }
  return cost;
}

// Prints every collected candidate with its priced deltas.
static void printCandidates(Session &S, FunctionFPState &st) {
  if (S.Opts.Diagnostics.Print)
    for (SiteUnit *unit : st.units)
      unit->printPriced(llvm::errs());
}

bool collectFPCandidates(Session &S, Function &F, double errorTol,
                         unsigned sampleLogBits, FunctionFPState &st) {
  const Options &O = S.Opts;
  st.F = &F;
  st.errTol = errorTol;
  st.sampleLogBits = sampleLogBits;

  const Target &T = Target::of(F);
  if (T.isDevice())
    llvm::errs() << "[poseidon] GPU mode active for " << F.getName() << "\n";

  SiteProfileData profile;
  readSiteProfile(S, F, profile);
  recognizeMatrixProducts(S, F, profile, st);

  // Nothing above prices: a compile for a device no model has been measured
  // on yet still reports what it recognized before it refuses.
  requireCostModel(S.CostModel, F);
  st.profiledCost = profiledCost(S.CostModel, F, profile.map);

  buildSubgraphs(O, F, st.abstractMatmuls, st.valueToNodeMap, st.subgraphs);

  // Profile read must happen before aggressive DCE as it requires gradients
  attachProfile(O, profile.map, st.valueToNodeMap, st.subgraphs);
  shapeSubgraphs(O, F, profile.map, profile.zeroGradIdx, st.abstractMatmuls,
                 st.valueToNodeMap, st.subgraphs);

  if (st.subgraphs.empty() && st.abstractMatmuls.empty()) {
    if (O.Diagnostics.Print)
      llvm::errs() << "No subgraphs or matmuls found\n";
    return false;
  }

  const bool expansionSafe =
      expansionsRoundedAsWritten(O, F, !st.subgraphs.empty());
  proposeSubgraphCandidates(S, F, profile.header, expansionSafe, st);

  generateMatmulCandidates(S, st.abstractMatmuls, profile.map,
                           kDefaultConfidence, st.sampleLogBits, st.CMs);
  st.sealUnits(S);

  priceUnlessCached(S, st, profile.path);

  putOnApplicationScale(S, F, profile, st);
  printCandidates(S, st);

  return true;
}

// Split out of fpOptimize so the joint solver can collect across functions,
// solve once, then materialize each function with its own steps.
bool materializeFPSolution(Session &S, Function &F, FunctionFPState &st,
                           ArrayRef<SolutionStep> steps) {
  const Options &O = S.Opts;
  auto &subgraphs = st.subgraphs;
  SmallVector<SolutionStep> applied;
  bool changed = applySolution(steps, applied);
  printPin(S, st, applied);

  llvm::errs() << "[poseidon] Finished optimizing " << F.getName() << "\n";

  if (changed) {
    for (auto &subgraph : subgraphs) {
      if (subgraph.outputs_rewritten != subgraph.outputs.size()) {
        if (O.Diagnostics.Print)
          llvm::errs() << "Skip erasing a subgraph: only rewrote "
                       << subgraph.outputs_rewritten << " of "
                       << subgraph.outputs.size() << " outputs\n";
        continue; // Intermediate operations cannot be erased safely
      }
      for (auto *I : subgraph.operations) {
        if (O.Diagnostics.Print)
          llvm::errs() << "Erasing: " << *I << "\n";
        if (!I->use_empty()) {
          I->replaceAllUsesWith(UndefValue::get(I->getType()));
        }
        I->eraseFromParent();
      }
    }

    llvm::errs() << "[poseidon] Finished cleaning up " << F.getName() << "\n";
  }

  if (changed)
    applyStagingNarrowing(S, F, /*announce=*/true,
                          O.Materialization.NarrowParamStaging ? ParamArm::Both
                                                               : ParamArm::None,
                          &F);
  if (changed && demoteFPCastPHIs(F)) {
    if (O.Diagnostics.Print)
      llvm::errs() << "[poseidon] demoted fpcast-sandwiched FP64 PHIs in "
                   << F.getName() << "\n";
  }

  simplifyFunction(F, OptimizationLevel::O3);

  // Mark the Poseidon-transformed function `alwaysinline` so the downstream
  // inliner folds it back into the caller. The source-level `noinline` on the
  // user's matmul body is needed UPSTREAM so the function survives as a
  // distinct symbol Poseidon can clone, but the call boundary on the
  // transformed clone costs ~1.5x wallclock on GPU (NVPTX passes pointer args
  // via per-thread R-regs across calls, forcing R2UR broadcasts and tanking
  // uniformity analysis).
  //
  // The two `remove` calls are required by the IR verifier: `alwaysinline`
  // conflicts with `noinline` / `optimizenone`.
  if (changed) {
    F.removeFnAttr(Attribute::NoInline);
    F.removeFnAttr(Attribute::OptimizeNone);
    F.addFnAttr(Attribute::AlwaysInline);
  }

  if (O.Diagnostics.Print) {
    llvm::errs() << "[poseidon] Finished Optimization\n";
    F.print(llvm::errs());
  }

  return changed;
}

bool fpOptimize(Session &S, Function &F, double errorTol) {
  const Options &O = S.Opts;
  // A target written AT THE SITE wins over -poseidon-tau, which is the target
  // of every site that carries none. Either way the number means the same: one
  // path below solves the site under it.
  const double target = errorTol > 0.0 ? errorTol : (double)O.Solver.Tau;

  // Matrix-product selection is only trustworthy with dynamic-range-aware
  // sampling (uniform [min,max] under-prices fixed-point Ozaki on
  // wide-dynamic-range matrices), so a target defaults the sampling knob to a
  // wide log-uniform range; an explicit -poseidon-sample-log-bits still
  // overrides. Read only by the matmul accuracy model, so a site with no matrix
  // product is unaffected.
  unsigned sampleLogBits = O.Model.SampleLogBits;
  if (target > 0.0 && sampleLogBits == 0)
    sampleLogBits = 40;

  FunctionFPState st;
  if (!collectFPCandidates(S, F, errorTol, sampleLogBits, st)) {
    explainNothingToOptimize(S, F);
    return false;
  }

  if (errorTol > 0.0 && O.Solver.Tau > 0.0)
    llvm::errs() << "[poseidon] " << F.getName()
                 << ": the site's own accuracy target " << errorTol
                 << " overrides -poseidon-tau=" << O.Solver.Tau << "\n";

  SolveRecord record;
  SolveRecord *rec = explainWanted(O, F) ? &record : nullptr;
  record.accScale = st.accScale;
  record.profiledCost = st.profiledCost;
  record.tau = target;
  record.siteTarget = errorTol > 0.0;
  SmallVector<SolutionStep> steps;
  if (!O.Solver.ApplyRewrites.empty()) {
    steps = parseManualRewrites(S, st);
    record.selector = "manual";
  } else if (target > 0.0 && !st.CMs.empty()) {
    // The target is a RELATIVE error, and for a matrix product that is what
    // errorBudgetSelector compares against. Whatever elementwise work the
    // products do not cover is then solved by the normalized DP under the same
    // number.
    steps = errorBudgetSelector(O, st.CMs, target, kDefaultConfidence);
    if (!st.COs.empty() || !st.CSs.empty()) {
      SmallVector<SolutionStep> elem =
          accuracyDPSolver(S, F, st, solvedUnits(st, /*target=*/true), target,
                           st.accScale, rec, st.table.get());
      for (const SolutionStep &e : elem) {
        // The two selectors ran independently, so the footprint rule the DP
        // applies internally is applied between their results here. Never
        // silent: dropping a rewrite the DP priced changes what the site emits.
        if (overlapsConsumed(e, steps)) {
          llvm::errs() << "[poseidon] " << F.getName()
                       << ": an elementwise rewrite the DP selected edits "
                          "instructions a selected matrix-product raise "
                          "consumes; keeping the raise and dropping the "
                          "rewrite\n";
          continue;
        }
        steps.push_back(e);
      }
    }
    record.selector = "error-budget";
  } else {
    steps = accuracyDPSolver(S, F, st, st.units, target, st.accScale, rec,
                             st.table.get());
  }
  if (rec)
    explainSolve(S, F, st, steps, record);
  return materializeFPSolution(S, F, st, steps);
}

bool solveJointly(Session &S, Module &M, FunctionAnalysisManager &FAM) {
  const Options &O = S.Opts;
  SmallVector<Function *, 4> marked;
  for (Function &F : M)
    if (!F.empty() && F.hasFnAttribute("poseidon-joint-errtol"))
      marked.push_back(&F);
  if (marked.empty())
    return false;

  llvm::errs() << "Poseidon JOINT: collecting candidates from " << marked.size()
               << " marked function(s)\n";

  // One shared budget is spent against one target, so every site in the
  // solve has to mean the same one: its own if it carries one, else
  // -poseidon-tau, else none.
  SmallVector<std::pair<Function *, double>, 4> targets;
  bool allFromSite = true;
  for (Function *F : marked) {
    double errTol = 0.0;
    F->getFnAttribute("poseidon-joint-errtol")
        .getValueAsString()
        .getAsDouble(errTol);
    F->removeFnAttr("poseidon-joint-errtol");
    allFromSite &= errTol > 0.0;
    targets.push_back({F, errTol > 0.0 ? errTol : (double)O.Solver.Tau});
  }
  const double jointErrTol = targets.front().second;
  if (llvm::any_of(targets,
                   [&](const auto &t) { return t.second != jointErrTol; })) {
    std::string msg;
    raw_string_ostream os(msg);
    os << "Poseidon: the sites of one -poseidon-joint-dp solve have different "
          "accuracy targets:";
    for (auto &[F, t] : targets) {
      const Function *origin = siteOriginOf(S, *F);
      os << " "
         << llvm::demangle((origin ? origin->getName() : F->getName()).str())
         << " (";
      if (t > 0.0)
        os << t;
      else
        os << "none";
      os << ")";
    }
    os << "; give them one target";
    M.getContext().diagnose(DiagnosticInfoUnsupported(
        *targets.front().first, os.str(), DiagnosticLocation(), DS_Error));
    for (auto &[F, t] : targets)
      redirectNoopSite(S, F);
    return false;
  }

  std::vector<std::unique_ptr<FunctionFPState>> states;
  for (auto &[F, errTol] : targets) {
    auto st = std::make_unique<FunctionFPState>();
    if (collectFPCandidates(S, *F, errTol, O.Model.SampleLogBits, *st)) {
      states.push_back(std::move(st));
    } else {
      llvm::errs() << "Poseidon JOINT: nothing to optimize in " << F->getName()
                   << "\n";
      redirectNoopSite(S, F);
    }
  }
  if (states.empty())
    return false;

  SmallVector<FunctionFPState *, 4> raw;
  for (auto &s : states)
    raw.push_back(s.get());

  SolveRecord record;
  const bool explain = llvm::any_of(
      raw, [&](FunctionFPState *st) { return explainWanted(O, *st->F); });
  record.tau = jointErrTol;
  record.siteTarget = allFromSite;
  for (FunctionFPState *st : raw)
    record.profiledCost += st->profiledCost;
  SmallVector<SolutionStep> steps;
  if (!O.Solver.ApplyRewrites.empty()) {
    for (auto *st : raw) {
      auto s = parseManualRewrites(S, *st);
      steps.append(s.begin(), s.end());
    }
    record.selector = "manual";
  } else {
    steps =
        jointAccuracyDPSolver(S, raw, jointErrTol, explain ? &record : nullptr);
  }

  std::unordered_map<FunctionFPState *, SmallVector<SolutionStep>> perState;
  for (auto &step : steps)
    perState[&unitOf(step).site()].push_back(step);

  // A site whose step list is empty gets its wrapper call restored to the
  // original body: the preprocessed clone is not guaranteed to lower
  // bit-identically, and an unrewritten site must be numerically transparent.
  if (explain)
    for (auto *st : raw)
      explainSolve(S, *st->F, *st, perState[st], record);
  bool changed = false;
  for (auto *st : raw) {
    bool ch = materializeFPSolution(S, *st->F, *st, perState[st]);
    if (!ch)
      redirectNoopSite(S, st->F);
    changed |= ch;
  }
  // The joint materialize has stashed the host-dispatch notes; emit the
  // deferred descriptors now.
  flushPendingGemmDispatches(S);
  return changed;
}

} // namespace poseidon
