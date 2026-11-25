//===- Optimize.cpp - a site's candidates, solve and materialization ------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Optimize.h"
#include "Explain.h"
#include "IR/Utils.h"
#include "Model/CostModel.h"
#include "Pass/Session.h"
#include "Profile/ProfileRead.h"
#include "Sites/Canonicalize.h"
#include "Sites/SiteIdentity.h"
#include "Solvers.h"
#include "Targets/Target.h"
#include "Techniques/Expansion.h"
#include "Techniques/Herbie.h"
#include "Techniques/Matmul/HostDispatch.h"
#include "Techniques/Matmul/Matmul.h"
#include "Techniques/Precision.h"
#include "Techniques/Staging.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace llvm;

namespace poseidon {

std::string siteProfilePath(const Session &S, const Function &clone,
                            StringRef dir) {
  SmallString<128> path(dir);
  sys::path::append(path, siteProfileFile(S.Sites, clone));
  return path.str().str();
}

// Reads the site's profile, checked against the canonical form its slots were
// numbered against.
static void readSiteProfile(Session &S, Function &F, SiteProfileData &P) {
  const Options &O = S.Opts;
  assert(!O.Profiling.ProfileUse.empty() &&
         "a site is optimized only under profile use");

  // Names the site by what it computes, not by the kernel it was cut from: the
  // profile is only readable against the canonical form it was numbered
  // against.
  const std::string canonicalHash = poseidon::canonicalFormHash(F);
  if (O.Diagnostics.Print)
    llvm::errs() << "[poseidon] canonical form hash of " << F.getName()
                 << " is " << canonicalHash << '\n';

  P.path = siteProfilePath(S, F, O.Profiling.ProfileUse);
  P.canonicalHash = canonicalHash;
  parseProfileFile(P.path, F, P.map, P.header);
  requireCanonicalHash(P.header, canonicalHash, P.path, F);
  auto &profileMap = P.map;
  auto &profileHeader = P.header;
  const std::string &profilePath = P.path;
  if (profileMap.empty())
    diagnose(F, F.getSubprogram(),
             "the profile " + profilePath + " records no operation of the site",
             DS_Warning);
  if (!O.Profiling.OutputProfiles.empty())
    mergeOutputProfiles(profileMap, O.Profiling.OutputProfiles,
                        siteProfileFile(S.Sites, F), canonicalHash, F,
                        O.Diagnostics.Print);
  // Before anything reads a gradient: the floor is written back into the
  // profile records so that every reader sees one weight.
  applySensitivityFloor(profileMap, F.getName(), O.Diagnostics.Print);
  // Per-output accounting weighs each output on its own scale and takes no
  // condition number.
  if (O.Profiling.OutputProfiles.empty())
    requireMeasuredKappa(profileHeader, profilePath, F);
  if (O.Diagnostics.Print && profileHeader.launchCount > 0) {
    llvm::errs() << "[poseidon] profile header for " << F.getName()
                 << ": MaxBlockDims=(" << profileHeader.maxBlockDim[0] << ','
                 << profileHeader.maxBlockDim[1] << ','
                 << profileHeader.maxBlockDim[2]
                 << ") LaunchCount=" << profileHeader.launchCount << '\n';
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
    // Runtime-shape dense GEMMs. Runs after the constant-trip recognizer and
    // skips whatever that one claimed, so a product both recognize keeps the
    // constant-trip candidate set.
    findHostGemmLoopNests(O, F, *raiseSE, *raiseLI, profileHeader, profileMap,
                          abstractMatmuls);
  }

  if (O.Diagnostics.Print && !abstractMatmuls.empty()) {
    llvm::errs() << "[poseidon] Found " << abstractMatmuls.size()
                 << " matrix product(s) in " << F.getName() << '\n';
    auto dimStr = [](unsigned d) -> std::string {
      return d == ~0u ? std::string("?") : std::to_string(d);
    };
    for (const auto &m : abstractMatmuls) {
      llvm::errs() << "  Matmul[" << m.id << "]: " << dimStr(m.M) << 'x'
                   << dimStr(m.N) << 'x' << dimStr(m.K)
                   << " type=" << fpKindName(m.elemType) << " origin="
                   << (m.origin == AbstractMatmul::Origin::HostGemmLoopNest
                           ? "HostGemmLoopNest"
                           : "ScalarLoopReduction")
                   << '\n';
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
    ++subgraphCounter;
  }
}

// Puts this site's error on the application's scale. kappa is the measured
// response of the quantity of interest to a relative error eps on every
// operation of the site; the profile predicts eps times the site's total
// sensitivity for the same noise. Their ratio cancels the propagation within
// the site and leaves the downstream condition per unit of accuracy cost. A
// site whose probe run diverged gets a large finite factor, so none of its
// candidates is traded against precision.
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
      reportUserError(
          F, F.getSubprogram(),
          Twine("the profile of ") + F.getName() +
              " carries a measured condition number, which scales one "
              "site-wide error; it cannot be combined with "
              "-poseidon-output-profiles. Leave out one or the other.");
    siteSumSens = 1.0;
  }

  double factor = 1.0;
  // A condition number the profiling runs did not resolve (the metric moved
  // within their noise, or not at all) is no measurement, and scaling the
  // site's errors by it would let any rewrite through. The site is solved
  // against its own target, as a site of a run without a metric is.
  const bool unresolved = profileHeader.hasKappa &&
                          (profileHeader.kappaMark == KappaMark::BelowNoise ||
                           !(profileHeader.kappa > 0.0));
  if (unresolved)
    diagnose(F, F.getSubprogram(),
             Twine("the profiling runs did not resolve the condition "
                   "number of ") +
                 F.getName() +
                 " against the declared metric from their "
                 "noise (Kappa = " +
                 shortDouble(profileHeader.kappa) + ", " +
                 kappaMarkName(profileHeader.kappaMark) +
                 "), so the site is solved against its own target; probe "
                 "it with larger perturbations (POSEIDON_PROBE_LADDER) to "
                 "measure it",
             DS_Warning);
  if (profileHeader.hasKappa && !unresolved) {
    double sumSens = siteSumSens;
    if (sumSens == 0.0)
      reportUserError(F, F.getSubprogram(),
                      Twine("the profile of ") + F.getName() +
                          " carries a measured condition number but a zero "
                          "total sensitivity, so the accuracy costs cannot "
                          "be put on the scale the condition number was "
                          "measured against; re-run profile generation.");
    factor = profileHeader.kappaMark == KappaMark::Diverged
                 ? 1e30
                 : profileHeader.kappa / sumSens;
    st.kappa = profileHeader.kappa;
    st.kappaMark = profileHeader.kappaMark;
    if (O.Diagnostics.Print)
      llvm::errs() << "[poseidon] " << F.getName()
                   << ": measured condition number " << profileHeader.kappa
                   << " (" << kappaMarkName(profileHeader.kappaMark)
                   << "), profiled sensitivity " << sumSens << ", error scale "
                   << factor << '\n';
  }
  // The accuracy costs below are scaled by `factor`, so the level a tolerance
  // is compared against has to be scaled the same way; with a kappa this
  // reduces to kappa itself, without one to the site's total sensitivity.
  st.accScale = factor * siteSumSens;
  st.scaleFactor = factor;
  if (factor != 1.0)
    for (SiteUnit *unit : st.units)
      unit->scaleErrors(factor);
}

// Prints every collected candidate with its priced deltas.
static void printCandidates(Session &S, FunctionFPState &st) {
  if (S.Opts.Diagnostics.Print)
    for (SiteUnit *unit : st.units)
      unit->printPriced(llvm::errs());
}

static void priceCandidates(FunctionFPState &st) {
  for (SiteUnit *unit : st.units)
    unit->price();
  for (Subgraph &subgraph : st.subgraphs)
    subgraph.samples.reset();
}

// Candidates are priced, and repriced by the solve, on scratch copies of the
// site, and what the copies declare is not the module's: a replay prices
// nothing, and its module is the solve's.
static StringSet<> declaredFunctions(const Module &M) {
  StringSet<> names;
  for (const Function &G : M)
    names.insert(G.getName());
  return names;
}

static void eraseScratchDeclarations(Module &M, const StringSet<> &before) {
  for (Function &G : make_early_inc_range(M))
    if (G.isDeclaration() && G.use_empty() && !before.contains(G.getName()))
      G.eraseFromParent();
}

// The units the DP solves: every unit of the site, or under an accuracy
// target every unit but the matrix products, which errorBudgetSelector picks.
static SmallVector<SiteUnit *> solvedUnits(const FunctionFPState &st,
                                           bool target) {
  SmallVector<SiteUnit *> units;
  for (SiteUnit *unit : st.units)
    if (!target || !isa<MatmulUnit>(unit))
      units.push_back(unit);
  return units;
}

void FunctionFPState::sealUnits(Session &S) {
  assert(!unitsSealed && "a site's units are sealed once");
  unitsSealed = true;
  for (size_t i = 0; i < COs.size(); ++i) {
    COs[i].bind(S, *this, *this, i);
    units.push_back(&COs[i]);
  }
  for (size_t i = 0; i < CSs.size(); ++i) {
    CSs[i].bind(S, *this, *this, i);
    units.push_back(&CSs[i]);
  }
  for (size_t i = 0; i < CMs.size(); ++i) {
    CMs[i].bind(S, *this, *this, i);
    units.push_back(&CMs[i]);
  }
}

// Prices the collected candidates, unless the site's DP table is read back
// from the cache: the table already holds every price its solve used.
static void priceUnlessCached(Session &S, FunctionFPState &st,
                              StringRef profilePath) {
  const Options &O = S.Opts;
  // A joint or a manual solve reads no table, and an accuracy target leaves
  // the matrix products to errorBudgetSelector (fpOptimize).
  if (!O.Herbie.Cache.empty() && !O.Solver.JointDP &&
      O.Solver.ApplyRewrites.empty()) {
    const bool target = st.errTol > 0.0 || O.Solver.Tau > 0.0;
    const SmallVector<SiteUnit *> units = solvedUnits(st, target);
    st.table = lookupSiteTable(S, st, units, profilePath);
    if (st.table->hit)
      return;
  }
  priceCandidates(st);
}

// Collects the candidates of the site F into `st` and prices them unless its
// DP table is cached; false when F has no FP subgraph and no matrix product.
static bool collectFPCandidates(Session &S, Function &F, double errorTol,
                                unsigned sampleLogBits, FunctionFPState &st) {
  const Options &O = S.Opts;
  st.F = &F;
  st.errTol = errorTol;

  const Target &T = Target::of(F);
  if (T.isDevice() && O.Diagnostics.Print)
    llvm::errs() << "[poseidon] GPU mode active for " << F.getName() << '\n';

  SiteProfileData profile;
  readSiteProfile(S, F, profile);
  st.profilePath = profile.path;
  st.canonicalHash = profile.canonicalHash;
  recognizeMatrixProducts(S, F, profile, st);

  // Nothing above prices: a compile for a device no model has been measured
  // on yet still reports what it recognized before it refuses.
  requireCostModel(S.CostModel, F);
  st.profiledCost = profiledCost(S.CostModel, F, profile.map,
                                 [](const Instruction &I, size_t &slot) {
                                   return tryReadProfIdxMetadata(&I, slot);
                                 });

  buildSubgraphs(O, F, profile.map, st.abstractMatmuls, st.valueToNodeMap,
                 st.subgraphs);

  attachProfile(O, profile.map, st.valueToNodeMap, st.subgraphs);
  shapeSubgraphs(O, F, st.subgraphs);
  if (O.Candidates.EnablePT && !O.Candidates.MergeSharedStaging)
    hintSharedStaging(S, F, st.subgraphs);

  if (st.subgraphs.empty() && st.abstractMatmuls.empty())
    return false;

  const bool expansionSafe =
      expansionsRoundedAsWritten(O, F, !st.subgraphs.empty());
  proposeSubgraphCandidates(S, F, profile.header, expansionSafe, st);

  generateMatmulCandidates(S, st.abstractMatmuls, profile.map, sampleLogBits,
                           st.CMs);
  st.sealUnits(S);

  priceUnlessCached(S, st, profile.path);

  putOnApplicationScale(S, F, profile, st);
  if (st.table && st.table->hit) {
    if (O.Diagnostics.Print)
      llvm::errs() << "[poseidon] " << F.getName()
                   << ": candidates not priced; the DP table read from the "
                      "cache holds the prices its solve used\n";
  } else {
    printCandidates(S, st);
  }

  return true;
}

SmallVector<SolutionStep> stepsToApply(ArrayRef<SolutionStep> steps) {
  SmallVector<SolutionStep> applied;
  SmallPtrSet<Value *, 8> rewrittenOutputs;
  for (const SolutionStep &step : steps) {
    ArrayRef<Value *> outputs = unitOf(step).rewrittenOutputs();
    if (llvm::any_of(outputs,
                     [&](Value *V) { return rewrittenOutputs.contains(V); }))
      continue;
    rewrittenOutputs.insert(outputs.begin(), outputs.end());
    applied.push_back(step);
  }
  return applied;
}

// Materializes the steps of `steps` that stepsToApply keeps, which it
// returns.
static SmallVector<SolutionStep> applySolution(const Options &O,
                                               ArrayRef<SolutionStep> steps) {
  SmallVector<SolutionStep> applied = stepsToApply(steps);
  if (steps.empty())
    return applied;
  if (O.Diagnostics.Print)
    llvm::errs() << "[poseidon] Applying " << steps.size() << " step(s)\n";
  const SolutionStep *next = applied.begin();
  for (const SolutionStep &step : steps) {
    SiteUnit &unit = unitOf(step);
    if (O.Diagnostics.Print)
      llvm::errs() << "[poseidon] Applying solution for "
                   << unit.choiceText(step.candidateIndex) << '\n';
    if (next == applied.end() || next->unit != step.unit ||
        next->candidateIndex != step.candidateIndex) {
      if (O.Diagnostics.Print)
        llvm::errs() << "[poseidon] output already rewritten by an "
                        "earlier step; skipping this one\n";
      continue;
    }
    ++next;
    unit.materialize(step.candidateIndex);
  }
  return applied;
}

// Applies `steps` to F and cleans F up; true when a step was applied.
static bool materializeFPSolution(Session &S, Function &F, FunctionFPState &st,
                                  ArrayRef<SolutionStep> steps) {
  const Options &O = S.Opts;
  auto &subgraphs = st.subgraphs;
  SmallVector<SolutionStep> applied = applySolution(O, steps);
  const bool changed = !steps.empty();
  printPin(S, st, applied);

  if (O.Diagnostics.Print)
    llvm::errs() << "[poseidon] Finished optimizing " << F.getName() << '\n';

  if (changed) {
    for (auto &subgraph : subgraphs) {
      // Intermediate operations cannot be erased safely.
      if (subgraph.outputs_rewritten != subgraph.outputs.size())
        continue;
      for (auto *I : subgraph.operations) {
        if (!I->use_empty()) {
          I->replaceAllUsesWith(UndefValue::get(I->getType()));
        }
        I->eraseFromParent();
      }
    }
  }

  if (changed)
    narrowSharedStaging(S, F, /*announce=*/true, ParamArm::Both, F);
  if (changed)
    demoteFPCastPHIs(F);

  // The emitted site is cleaned up without the target, which versions a loop
  // on a loop-invariant condition: that keeps a comparison of an operand left
  // in FP64 out of the loop of a rewrite to FP32, where the target's pipeline
  // leaves one per iteration on the FP64 pipe. Pricing walks the target's
  // clean-up of the site instead, which keeps one copy of each loop; a thread
  // runs one version of a versioned loop.
  simplifyFunction(F, OptimizationLevel::O3);

  // The rewritten site is inlined back into its caller: on a GPU a call passes
  // pointer arguments in per-thread registers, which defeats uniformity
  // analysis. The verifier rejects `alwaysinline` with `noinline` or
  // `optnone`.
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

// Selects the steps of `sites`, pinned or solved in one DP under `tau`, then
// explains and materializes each site; true when a site changed.
static bool solveAndApply(Session &S, Module &M, const StringSet<> &declared,
                          ArrayRef<FunctionFPState *> sites, double tau,
                          bool siteTarget, bool joint) {
  const Options &O = S.Opts;
  SolveRecord record;
  record.tau = tau;
  record.siteTarget = siteTarget;
  for (FunctionFPState *st : sites) {
    record.profiledCost += st->profiledCost;
    record.accScale += st->accScale;
    if (joint)
      record.sites.push_back(st->F->getName().str());
  }
  const bool explain = llvm::any_of(
      sites, [&](FunctionFPState *st) { return explainWanted(O, *st->F); });
  SolveRecord *rec = explain ? &record : nullptr;

  SmallVector<SolutionStep> steps;
  if (!O.Solver.ApplyRewrites.empty()) {
    for (FunctionFPState *st : sites)
      append_range(steps, parseManualRewrites(S, *st));
    record.selector = "manual";
  } else if (!joint && tau > 0.0 && !sites.front()->CMs.empty()) {
    FunctionFPState &st = *sites.front();
    // The target is a relative error, which is what errorBudgetSelector
    // compares a matrix product against. The elementwise units are solved by
    // the normalized DP under the same number.
    steps = errorBudgetSelector(O, st.CMs, tau);
    const SmallVector<SiteUnit *> elemUnits = solvedUnits(st, /*target=*/true);
    if (!elemUnits.empty())
      for (const SolutionStep &e :
           solveSites(S, sites, ArrayRef<SiteUnit *>(elemUnits), tau,
                      /*joint=*/false, rec)) {
        // The two selectors run independently, so the DP's footprint rule is
        // applied between their results here, loudly: dropping a priced
        // rewrite changes what the site emits.
        if (overlapsConsumed(e, steps)) {
          diagnose(*st.F, st.F->getSubprogram(),
                   "a selected rewrite edits instructions a selected matrix "
                   "product consumes; the rewrite is dropped",
                   DS_Warning);
          continue;
        }
        steps.push_back(e);
      }
    record.selector = "error-budget";
  } else {
    SmallVector<ArrayRef<SiteUnit *>, 4> units;
    for (FunctionFPState *st : sites)
      units.push_back(st->units);
    steps = solveSites(S, sites, units, tau, joint, rec);
  }

  std::unordered_map<FunctionFPState *, SmallVector<SolutionStep>> perSite;
  for (const SolutionStep &step : steps)
    perSite[&unitOf(step).site()].push_back(step);
  if (explain)
    for (FunctionFPState *st : sites)
      explainSolve(S, *st->F, *st, perSite[st], record);
  eraseScratchDeclarations(M, declared);

  bool changed = false;
  for (FunctionFPState *st : sites) {
    const bool siteChanged = materializeFPSolution(S, *st->F, *st, perSite[st]);
    // An unrewritten site calls the original body again: the preprocessed
    // clone need not lower bit-identically.
    if (joint && !siteChanged)
      redirectNoopSite(S.Sites, st->F, S.Opts.Diagnostics.Print);
    changed |= siteChanged;
  }
  // The host-dispatch descriptors of a joint solve are known only once every
  // site is materialized.
  if (joint)
    flushPendingGemmDispatches(S);
  return changed;
}

bool fpOptimize(Session &S, Function &F, double errorTol) {
  const Options &O = S.Opts;
  // A target written at the site wins over -poseidon-tau, the target of every
  // site that carries none; both mean the same.
  const double target = errorTol > 0.0 ? errorTol : (double)O.Solver.Tau;

  // Uniform sampling over [min, max] under-prices fixed-point Ozaki on matrices
  // of wide dynamic range, so a target defaults to a wide log-uniform range
  // unless -poseidon-sample-log-bits is given. Only the matrix accuracy model
  // reads it.
  unsigned sampleLogBits = O.Model.SampleLogBits;
  if (target > 0.0 && sampleLogBits == 0)
    sampleLogBits = 40;

  const StringSet<> declared = declaredFunctions(*F.getParent());
  FunctionFPState st;
  if (!collectFPCandidates(S, F, errorTol, sampleLogBits, st)) {
    explainNothingToOptimize(S, F, target, errorTol > 0.0);
    return false;
  }

  if (errorTol > 0.0 && O.Solver.Tau > 0.0 && O.Diagnostics.Print)
    llvm::errs() << "[poseidon] " << F.getName()
                 << ": the site's own accuracy target " << errorTol
                 << " overrides -poseidon-tau=" << O.Solver.Tau << '\n';

  return solveAndApply(S, *F.getParent(), declared, {&st}, target,
                       errorTol > 0.0, /*joint=*/false);
}

bool solveJointly(Session &S, Module &M) {
  const Options &O = S.Opts;
  SmallVector<Function *, 4> marked;
  for (Function &F : M)
    if (!F.empty() && F.hasFnAttribute(kJointErrTol))
      marked.push_back(&F);
  if (marked.empty())
    return false;

  if (O.Diagnostics.Print)
    llvm::errs() << "[poseidon] joint solve: collecting candidates from "
                 << marked.size() << " marked function(s)\n";

  // One shared budget is spent against one target, so every site in the
  // solve has to mean the same one: its own if it carries one, else
  // -poseidon-tau, else none.
  SmallVector<std::pair<Function *, double>, 4> targets;
  bool allFromSite = true;
  for (Function *F : marked) {
    double errTol = 0.0;
    F->getFnAttribute(kJointErrTol).getValueAsString().getAsDouble(errTol);
    F->removeFnAttr(kJointErrTol);
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
      const Function *origin = siteOriginOf(S.Sites, *F);
      os << ' '
         << llvm::demangle((origin ? origin->getName() : F->getName()).str())
         << " (";
      if (t > 0.0)
        os << t;
      else
        os << "none";
      os << ')';
    }
    os << "; give them one target";
    M.getContext().diagnose(DiagnosticInfoUnsupported(
        *targets.front().first, os.str(), DiagnosticLocation(), DS_Error));
    for (auto &[F, t] : targets)
      redirectNoopSite(S.Sites, F, S.Opts.Diagnostics.Print);
    return false;
  }

  const StringSet<> declared = declaredFunctions(M);
  std::vector<std::unique_ptr<FunctionFPState>> states;
  for (auto &[F, errTol] : targets) {
    auto st = std::make_unique<FunctionFPState>();
    if (collectFPCandidates(S, *F, errTol, O.Model.SampleLogBits, *st)) {
      states.push_back(std::move(st));
    } else {
      if (O.Diagnostics.Print)
        llvm::errs() << "[poseidon] joint solve: nothing to optimize in "
                     << F->getName() << '\n';
      explainNothingToOptimize(S, *F, errTol, allFromSite);
      redirectNoopSite(S.Sites, F, S.Opts.Diagnostics.Print);
    }
  }
  if (states.empty())
    return false;

  SmallVector<FunctionFPState *, 4> sites;
  for (auto &st : states)
    sites.push_back(st.get());
  return solveAndApply(S, M, declared, sites, jointErrTol, allFromSite,
                       /*joint=*/true);
}

} // namespace poseidon
