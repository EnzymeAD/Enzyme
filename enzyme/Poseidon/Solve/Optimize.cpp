//===- Optimize.cpp - a site's candidates, solve and materialization ------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Core/Session.h"
#include "Core/Technique.h"
#include "Explain.h"
#include "Model/CostModel.h"
#include "Pins.h"
#include "Profile/ProfileReader.h"
#include "Sites/Canonicalize.h"
#include "Sites/Site.h"
#include "Sites/SiteIdentity.h"
#include "Solve.h"
#include "Support/Diagnostics.h"
#include "Support/Utils.h"
#include "Targets/Target.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cmath>
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
  sys::path::append(path, siteProfileFile(S.get<SiteRecords>(), clone));
  return path.str().str();
}

// Reads the site's profile, checked against the canonical form its slots were
// numbered against.
static Error readSiteProfile(Session &S, Function &F, SiteProfileData &P) {
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
  if (Error E = parseProfileFile(P.path, F, P.map, P.header))
    return E;
  if (Error E = requireCanonicalHash(P.header, canonicalHash, F))
    return E;
  if (P.map.empty())
    diagnose(F, "the profile " + P.path + " records no operation of the site",
             DS_Warning);
  if (!O.Profiling.OutputProfiles.empty())
    if (Error E = mergeOutputProfiles(P.map, O.Profiling.OutputProfiles,
                                      siteProfileFile(S.get<SiteRecords>(), F),
                                      canonicalHash, F, O.Diagnostics.Print))
      return E;
  // Before anything reads a gradient: the floor is written back into the
  // profile records so that every reader sees one weight.
  applySensitivityFloor(P.map, F.getName(), O.Diagnostics.Print);
  if (!P.map.empty() && P.map.begin()->second.hasSecondMoments)
    P.moment = ErrorMoment::Second;
  // Per-output accounting weighs each output on its own scale and takes no
  // condition number.
  if (O.Profiling.OutputProfiles.empty())
    if (Error E = requireMeasuredKappa(P.header, P.path, F))
      return E;
  if (O.Diagnostics.Print && P.header.launchCount > 0) {
    llvm::errs() << "[poseidon] profile header for " << F.getName()
                 << ": MaxBlockDims=(" << P.header.maxBlockDim[0] << ','
                 << P.header.maxBlockDim[1] << ',' << P.header.maxBlockDim[2]
                 << ") LaunchCount=" << P.header.launchCount << '\n';
  }
  return Error::success();
}

// Puts this site's error on the application's scale. kappa is the measured
// response of the quantity of interest to a relative error eps on every
// operation of the site; the profile predicts eps times the site's total
// sensitivity for the same noise (in the second moment, eps times the root of
// the summed squared sensitivity). Their ratio cancels the propagation within
// the site and leaves the downstream condition per unit of accuracy cost. A
// site whose kappa run diverged has no finite condition number: it is solved
// with no candidate (solveAndApply), under a target and under a budget.
static Error putOnApplicationScale(Session &S, Function &F,
                                   const SiteProfileData &P, SiteState &st) {
  const Options &O = S.Opts;
  const auto &profileMap = P.map;
  const auto &profileHeader = P.header;

  const bool second = st.moment == ErrorMoment::Second;
  double siteSumSens = 0.0;
  for (const auto &kv : profileMap)
    siteSumSens += second ? kv.second.sumSqSens : kv.second.sumSens;
  // Per-output weights are already divided by each output's own total.
  if (!O.Profiling.OutputProfiles.empty()) {
    if (profileHeader.kappa)
      return userError(
          F, Twine("the profile of ") + F.getName() +
                 " carries a measured condition number, which scales one "
                 "site-wide error and so cannot be combined with "
                 "-poseidon-output-profiles; leave out one or the other");
    siteSumSens = 1.0;
  }

  double factor = 1.0;
  // A condition number the profiling runs did not resolve (the metric moved
  // within their noise, or not at all) is no measurement, and scaling the
  // site's errors by it would let any rewrite through. The site is solved
  // against its own target, as a site of a run without a metric is.
  const bool unresolved = profileHeader.kappa &&
                          (profileHeader.kappaMark == KappaMark::BelowNoise ||
                           !(*profileHeader.kappa > 0.0));
  if (unresolved)
    diagnose(F,
             Twine("the profiling runs did not resolve the condition "
                   "number of ") +
                 F.getName() +
                 " against the declared metric from their "
                 "noise (Kappa = " +
                 shortDouble(*profileHeader.kappa) + ", " +
                 kappaMarkName(profileHeader.kappaMark) +
                 "), so the site is solved against its own target; measure it "
                 "with larger perturbation steps in POSEIDON_KAPPA_LADDER",
             DS_Warning);
  if (profileHeader.kappa && !unresolved) {
    double sumSens = siteSumSens;
    if (sumSens == 0.0)
      return userError(F, Twine("the profile of ") + F.getName() +
                              " carries a measured condition number but a zero "
                              "total sensitivity, so the accuracy costs cannot "
                              "be put on the scale the condition number was "
                              "measured against; re-run profile generation");
    if (profileHeader.kappaMark == KappaMark::Diverged)
      diagnose(F,
               Twine("the profiling runs of ") + F.getName() +
                   " diverged under the smallest perturbation, so no "
                   "candidate can be traded against its accuracy and the "
                   "site is left as written",
               DS_Warning);
    else
      factor = *profileHeader.kappa / (second ? std::sqrt(sumSens) : sumSens);
    // A second-moment cost is a variance: it scales by the factor's square.
    if (second)
      factor *= factor;
    SolvedFrom &from = st.get<SolvedFrom>();
    from.kappa = profileHeader.kappa;
    from.kappaMark = profileHeader.kappaMark;
    if (O.Diagnostics.Print)
      llvm::errs() << "[poseidon] " << F.getName()
                   << ": measured condition number " << *profileHeader.kappa
                   << " (" << kappaMarkName(profileHeader.kappaMark)
                   << "), profiled sensitivity " << sumSens << ", error scale "
                   << factor << '\n';
  }
  // The accuracy costs below are scaled by `factor`, so the level a tolerance
  // is compared against has to be scaled the same way; with a kappa this
  // reduces to kappa itself (its square in the second moment), without one to
  // the site's total sensitivity.
  st.accScale = factor * siteSumSens;
  st.scaleFactor = factor;
  if (factor != 1.0)
    for (SiteUnit *unit : st.units)
      unit->scaleErrors(factor);
  return Error::success();
}

static Error priceCandidates(SiteState &st) {
  for (SiteUnit *unit : st.units)
    if (Error E = unit->price())
      return E;
  for (Subgraph &subgraph : st.subgraphs)
    subgraph.samples.reset();
  return Error::success();
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

void sealUnits(Session &S, SiteState &st) {
  assert(!st.unitsSealed && "a site's units are sealed once");
  st.unitsSealed = true;
  for (const Technique *T : techniques())
    T->bindUnits(S, st);
}

// The cost change of each candidate of `units` that pricing priced, against
// the site's own axis: an error at the site for one the axis cannot hold. A
// joint solve's axis is no finer, its profiled cost being at least the site's.
static Error requireCostsOnAxis(SiteState &st, ArrayRef<SiteUnit *> units) {
  if (st.profiledCost == 0.0)
    return Error::success();
  const CostScale scale(st.profiledCost);
  for (SiteUnit *unit : units)
    for (size_t i = 0; i < unit->numCandidates(); ++i) {
      std::optional<double> priced = unit->pricedCostDelta(i);
      if (!priced)
        continue;
      const double delta = *priced;
      if (scale.toDPCost(delta))
        continue;
      std::string Text;
      raw_string_ostream(Text) << delta;
      return userError(*st.F,
                       unit->choiceText(i) + " changes the cost of " +
                           st.F->getName() + " by " + Text +
                           ", which does not fit the DP's cost axis; a "
                           "cost-model row prices it beyond 2^30 times the "
                           "profiled cost");
    }
  return Error::success();
}

// Prices the collected candidates, unless the site's DP table is read back
// from the cache: the table already holds every price its solve used.
static Error priceUnlessCached(Session &S, SiteState &st,
                               StringRef profilePath) {
  const Options &O = S.Opts;
  // A joint or a manual solve reads no table.
  if (!O.Cache.Dir.empty() && !O.Solver.SolveJointly && O.Solver.Pin.empty()) {
    Expected<std::unique_ptr<DPTable>> table =
        lookupDPTable(S, st, st.units, profilePath);
    if (!table)
      return table.takeError();
    std::unique_ptr<DPTable> &cached = st.get<SolvedFrom>().table;
    cached = std::move(*table);
    if (cached->hit)
      return Error::success();
  }
  if (Error E = priceCandidates(st))
    return E;
  // A manual solve runs no DP.
  if (!O.Solver.Pin.empty())
    return Error::success();
  return requireCostsOnAxis(st, st.units);
}

// Collects the candidates of the site F, solved against `target`, into `st`
// and prices them unless its DP table is cached; false when F has no FP
// subgraph and no technique recognized anything in it.
static Expected<bool> collectFPCandidates(Session &S, Function &F,
                                          double target, SiteState &st) {
  const Options &O = S.Opts;
  st.F = &F;
  st.target = target;

  const Target &T = Target::of(F);
  if (T.isDevice() && O.Diagnostics.Print)
    llvm::errs() << "[poseidon] GPU mode active for " << F.getName() << '\n';

  SiteProfileData profile;
  if (Error E = readSiteProfile(S, F, profile))
    return std::move(E);
  SolvedFrom &from = st.get<SolvedFrom>();
  from.profilePath = profile.path;
  from.canonicalHash = profile.canonicalHash;
  bool recognized = false;
  for (const Technique *T : techniques())
    if (T->enabled(O)) {
      Expected<bool> Found = T->recognize(S, F, profile, st);
      if (!Found)
        return Found.takeError();
      recognized |= *Found;
    }

  // Nothing above prices: a compile for a device no model has been measured
  // on yet still reports what it recognized before it refuses.
  if (Error E = requireCostModel(S, F))
    return std::move(E);
  Expected<double> cost =
      profiledCost(S.get<CostModelState>(), F, profile.map, slotOf);
  if (!cost)
    return cost.takeError();
  st.profiledCost = *cost;

  buildSubgraphs(O, profile.map, st);

  if (Error E = attachProfile(O, profile.map, st, st.subgraphs))
    return std::move(E);
  shapeSubgraphs(O, F, st.subgraphs);
  st.moment = profile.moment;
  for (Subgraph &subgraph : st.subgraphs)
    subgraph.moment = st.moment;

  if (st.subgraphs.empty() && !recognized)
    return false;

  for (const Technique *T : techniques())
    if (T->enabled(O))
      if (Error E = T->beginSite(S, F, st))
        return std::move(E);
  if (!st.subgraphs.empty())
    for (const Technique *T : techniques())
      if (!T->enabled(O))
        T->remarkUnavailable(S, F);
  // One subgraph at a time, every technique's units for it.
  for (auto [index, subgraph] : enumerate(st.subgraphs)) {
    assert(subgraph.inputs.size() > 0 && "No inputs found for subgraph");
    for (const Technique *T : techniques())
      if (T->enabled(O))
        if (Error E = T->proposeForSubgraph(S, F, st, subgraph, index, profile))
          return std::move(E);
  }
  for (const Technique *T : techniques())
    if (T->enabled(O))
      if (Error E = T->proposeForSite(S, st, profile))
        return std::move(E);
  sealUnits(S, st);

  if (Error E = priceUnlessCached(S, st, profile.path))
    return std::move(E);

  if (Error E = putOnApplicationScale(S, F, profile, st))
    return std::move(E);
  if (O.Diagnostics.Print) {
    if (from.table && from.table->hit)
      llvm::errs() << "[poseidon] " << F.getName()
                   << ": candidates not priced; the DP table read from the "
                      "cache holds the prices its solve used\n";
    else
      for (SiteUnit *unit : st.units)
        unit->printPriced(llvm::errs());
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
static Expected<SmallVector<SolutionStep>>
applySolution(const Options &O, ArrayRef<SolutionStep> steps) {
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
    if (Error E = unit.materialize(step.candidateIndex))
      return std::move(E);
  }
  return applied;
}

// Applies `steps` to F and cleans F up; true when a step was applied.
static Expected<bool> materializeSolution(Session &S, Function &F,
                                          SiteState &st,
                                          ArrayRef<SolutionStep> steps) {
  const Options &O = S.Opts;
  Expected<SmallVector<SolutionStep>> applied = applySolution(O, steps);
  if (!applied)
    return applied.takeError();
  const bool changed = !steps.empty();
  printPin(S, st, *applied);

  if (O.Diagnostics.Print)
    llvm::errs() << "[poseidon] Finished optimizing " << F.getName() << '\n';

  if (changed) {
    for (Subgraph &subgraph : st.subgraphs) {
      // Intermediate operations cannot be erased safely.
      if (subgraph.outputsRewritten != subgraph.outputs.size())
        continue;
      for (Instruction *I : subgraph.operations) {
        if (!I->use_empty())
          I->replaceAllUsesWith(PoisonValue::get(I->getType()));
        I->eraseFromParent();
      }
    }
    for (const Technique *T : techniques())
      T->siteApplied(S, F);
  }

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
/// Calls the body `clone` was cut from in place of `clone`, a clone the solve
/// applied nothing to; `print` says so on stderr.
static void redirectNoopSite(SiteRecords &T, Function *clone, bool print) {
  auto it = T.Clones.find(clone);
  if (it == T.Clones.end() || !it->second.Origin)
    return;
  Function *orig = it->second.Origin;
  if (orig == clone)
    return;
  SmallVector<CallBase *, 4> callers = callsOf(*clone);
  for (CallBase *CB : callers)
    CB->setCalledFunction(orig);
  if (!callers.empty() && print)
    llvm::errs() << "[poseidon] no rewrite applied for " << clone->getName()
                 << "; wrapper call restored to original body "
                 << orig->getName() << " (bit-transparent no-op)\n";
}

static Expected<bool> solveAndApply(Session &S, Module &M,
                                    const StringSet<> &declared,
                                    ArrayRef<SiteState *> sites, double tau,
                                    bool siteTarget, bool joint) {
  const Options &O = S.Opts;
  // One table sums the sites' accuracy costs, which add up in one moment only.
  for (SiteState *st : sites)
    if (st->moment != sites.front()->moment)
      return userError(
          *st->F,
          Twine("the profile of ") + st->F->getName() +
              (st->moment == ErrorMoment::Second ? " records"
                                                 : " does not record") +
              " the squared sums SumSqGrad and SumSqSens, and the profile of " +
              sites.front()->F->getName() +
              (st->moment == ErrorMoment::Second ? " does not" : " does") +
              ", so the two cannot be solved jointly; profile both with one "
              "profiler");
  // A site whose condition number diverged offers no candidate.
  SmallVector<SiteState *, 4> solved;
  SmallVector<ArrayRef<SiteUnit *>, 4> units;
  for (SiteState *st : sites)
    if (st->get<SolvedFrom>().kappaMark != KappaMark::Diverged) {
      solved.push_back(st);
      units.push_back(st->units);
    }
  SolveRecord record;
  record.tau = tau;
  record.siteTarget = siteTarget;
  record.moment = sites.front()->moment;
  const ArrayRef<SiteState *> scaled =
      solved.empty() ? sites : ArrayRef<SiteState *>(solved);
  for (SiteState *st : scaled) {
    record.profiledCost += st->profiledCost;
    record.accScale += st->accScale;
  }
  if (joint)
    for (SiteState *st : sites)
      record.sites.push_back(st->F->getName().str());
  const bool explain = llvm::any_of(
      sites, [&](SiteState *st) { return explainWanted(O, *st->F); });
  SolveRecord *rec = explain ? &record : nullptr;

  SmallVector<SolutionStep> steps;
  if (!O.Solver.Pin.empty()) {
    for (SiteState *st : sites) {
      Expected<SmallVector<SolutionStep>> pinned = parsePins(S, *st);
      if (!pinned)
        return pinned.takeError();
      append_range(steps, *pinned);
    }
    record.selector = SolveSelector::Pinned;
  } else if (!solved.empty()) {
    Expected<SmallVector<SolutionStep>> selected =
        solveSites(S, solved, units, CostScale(record.profiledCost),
                   record.accScale, tau, joint, rec);
    if (!selected)
      return selected.takeError();
    steps = std::move(*selected);
  }

  std::unordered_map<SiteState *, SmallVector<SolutionStep>> perSite;
  for (const SolutionStep &step : steps)
    perSite[&unitOf(step).site()].push_back(step);
  if (explain)
    for (SiteState *st : sites)
      if (Error E = explainSolve(S, *st->F, *st, perSite[st], record))
        return std::move(E);
  eraseScratchDeclarations(M, declared);

  bool changed = false;
  for (SiteState *st : sites) {
    Expected<bool> siteChanged =
        materializeSolution(S, *st->F, *st, perSite[st]);
    if (!siteChanged)
      return siteChanged.takeError();
    // An unrewritten site calls the original body again: the preprocessed
    // clone need not lower bit-identically.
    if (joint && !*siteChanged)
      redirectNoopSite(S.get<SiteRecords>(), st->F, S.Opts.Diagnostics.Print);
    changed |= *siteChanged;
  }
  return changed;
}

Expected<bool> optimizeSite(Session &S, Function &F, double tau) {
  const Options &O = S.Opts;
  // A target written at the site wins over -poseidon-tau, the target of every
  // site that carries none; both mean the same.
  const double target = tau > 0.0 ? tau : (double)O.Solver.Tau;

  const StringSet<> declared = declaredFunctions(*F.getParent());
  SiteState st;
  Expected<bool> collected = collectFPCandidates(S, F, target, st);
  if (!collected)
    return collected.takeError();
  if (!*collected) {
    explainNothingToOptimize(S, F, target, tau > 0.0);
    return false;
  }

  if (tau > 0.0 && O.Solver.Tau > 0.0 && O.Diagnostics.Print)
    llvm::errs() << "[poseidon] " << F.getName()
                 << ": the site's own accuracy target " << tau
                 << " overrides -poseidon-tau=" << O.Solver.Tau << '\n';

  return solveAndApply(S, *F.getParent(), declared, {&st}, target, tau > 0.0,
                       /*joint=*/false);
}

Expected<bool> solveJointly(Session &S, Module &M) {
  const Options &O = S.Opts;
  SmallVector<Function *, 4> marked;
  for (Function &F : M)
    if (!F.empty() && F.hasFnAttribute(kJointTau))
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
    double tau = 0.0;
    [[maybe_unused]] const bool unparsable =
        F->getFnAttribute(kJointTau).getValueAsString().getAsDouble(tau);
    assert(!unparsable && "kJointTau holds the target the pass wrote");
    F->removeFnAttr(kJointTau);
    allFromSite &= tau > 0.0;
    targets.push_back({F, tau > 0.0 ? tau : (double)O.Solver.Tau});
  }
  const double jointTau = targets.front().second;
  if (llvm::any_of(targets,
                   [&](const auto &t) { return t.second != jointTau; })) {
    std::string msg;
    raw_string_ostream os(msg);
    os << "the sites of one -poseidon-solve-jointly solve have different "
          "accuracy targets:";
    for (auto &[F, t] : targets) {
      const Function *origin = siteOriginOf(S.get<SiteRecords>(), *F);
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
    for (auto &[F, t] : targets)
      redirectNoopSite(S.get<SiteRecords>(), F, S.Opts.Diagnostics.Print);
    return userError(*targets.front().first, msg);
  }

  const StringSet<> declared = declaredFunctions(M);
  std::vector<std::unique_ptr<SiteState>> states;
  for (auto &[F, tau] : targets) {
    auto st = std::make_unique<SiteState>();
    Expected<bool> collected = collectFPCandidates(S, *F, tau, *st);
    if (!collected)
      return collected.takeError();
    if (*collected) {
      states.push_back(std::move(st));
    } else {
      if (O.Diagnostics.Print)
        llvm::errs() << "[poseidon] joint solve: nothing to optimize in "
                     << F->getName() << '\n';
      explainNothingToOptimize(S, *F, tau, allFromSite);
      redirectNoopSite(S.get<SiteRecords>(), F, S.Opts.Diagnostics.Print);
    }
  }
  if (states.empty())
    return false;

  SmallVector<SiteState *, 4> sites;
  for (auto &st : states)
    sites.push_back(st.get());
  return solveAndApply(S, M, declared, sites, jointTau, allFromSite,
                       /*joint=*/true);
}

} // namespace poseidon
