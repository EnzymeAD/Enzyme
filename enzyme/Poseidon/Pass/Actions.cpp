//===- Actions.cpp - the site actions of the Poseidon passes --------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Core/Technique.h"
#include "PassInternal.h"

#include "Model/CostModel.h"
#include "Profile/Instrument.h"
#include "Profile/ProfileReader.h"
#include "Sites/Annotations.h"
#include "Sites/Canonicalize.h"
#include "Sites/Markers.h"
#include "Sites/Site.h"
#include "Sites/SiteIdentity.h"
#include "Solve/Explain.h"
#include "Solve/Optimize.h"
#include "Solve/Session.h"
#include "Support/Diagnostics.h"
#include "Support/Utils.h"
#include "runtime/profile/ProfileFormat.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>
#include <system_error>

using namespace llvm;

namespace poseidon {

static void noteRewritten(Session &S, Function &F) {
  SmallVector<Function *, 4> work{&F};
  SmallPtrSet<Function *, 4> seen;
  while (!work.empty()) {
    Function *G = work.pop_back_val();
    if (!seen.insert(G).second)
      continue;
    S.RewrittenFunctions.push_back(G);
    for (User *U : G->users())
      if (auto *CB = dyn_cast<CallBase>(U))
        work.push_back(CB->getFunction());
  }
}

// Redirect the site's call to `F`: the rewritten clone, or the original body
// when nothing was applied and the site has to stay bit-transparent.
static void emitSiteCall(CallBase *CI, Function *F,
                         SmallVectorImpl<Value *> &primalArgs,
                         SmallVectorImpl<CallBase *> &calls) {
  IRBuilder<> Builder(CI);
  auto *II = dyn_cast<InvokeInst>(CI);
  CallBase *optCall =
      II ? (CallBase *)Builder.CreateInvoke(F->getFunctionType(), F,
                                            II->getNormalDest(),
                                            II->getUnwindDest(), primalArgs)
         : Builder.CreateCall(F->getFunctionType(), F, primalArgs);
  optCall->setCallingConv(F->getCallingConv());
  optCall->setDebugLoc(CI->getDebugLoc());

  CI->replaceAllUsesWith(optCall);
  CI->eraseFromParent();

  calls.push_back(optCall);
}

// A profile-use compile pointed at a directory no profiling run wrote into
// would otherwise leave every site as written and succeed.
static Error requireProfileDirectory(CallBase *CI, StringRef dir,
                                     StringRef flag) {
  const bool isDir = sys::fs::is_directory(dir);
  std::error_code EC;
  if (isDir)
    for (sys::fs::directory_iterator it(dir, EC), end; !EC && it != end;
         it.increment(EC))
      if (sys::path::extension(it->path()) == kProfileFileSuffix)
        return Error::success();
  return userError(*CI,
                   Twine("the profile directory ") + dir + " (" + flag + ") " +
                       (isDir ? "holds no .fpprofile file" : "does not exist") +
                       "; run the program built with "
                       "-poseidon-profile-generate first, or pass the "
                       "directory it wrote");
}

namespace {
// Why a site has no records to solve from, read from every directory that
// holds a profile of it.
struct Unprofiled {
  bool neverExecuted;
  std::string path;
};
} // namespace

static Expected<std::optional<Unprofiled>>
unprofiledSite(Session &S, CallBase *CI, Function &clone) {
  std::optional<Unprofiled> absent, neverExecuted;
  auto read = [&](StringRef dir, StringRef flag) -> Error {
    const std::string path = siteProfilePath(S, clone, dir);
    if (!sys::fs::exists(path)) {
      if (Error E = requireProfileDirectory(CI, dir, flag))
        return E;
      if (!absent)
        absent = Unprofiled{false, path};
      return Error::success();
    }
    ProfileMap records;
    SiteProfileHeader header;
    if (Error E = parseProfileFile(path, clone, records, header))
      return E;
    if (header.executed)
      return Error::success();
    if (Error E =
            requireCanonicalHash(header, canonicalFormHash(clone), path, clone))
      return E;
    if (!neverExecuted)
      neverExecuted = Unprofiled{true, path};
    return Error::success();
  };
  if (Error E = read(S.Opts.Profiling.ProfileUse, "-poseidon-profile-use"))
    return std::move(E);
  for (const std::string &dir : S.Opts.Profiling.OutputProfiles)
    if (Error E = read(dir, "-poseidon-output-profiles"))
      return std::move(E);
  return absent ? absent : neverExecuted;
}

// `tau` > 0 is the site's accuracy target, its own when `siteTarget`.
static Error reportUnprofiled(Session &S, CallBase *CI, const Function &body,
                              Function &clone, const Unprofiled &site,
                              double tau, bool siteTarget) {
  if (site.neverExecuted) {
    const char *cause =
        "the profiling run instrumented this site and never executed it";
    if (S.Opts.Diagnostics.Print)
      llvm::errs() << "[poseidon] " << body.getName() << ": " << cause << " ("
                   << site.path << "); left unchanged\n";
    explainUnprofiled(S, clone, tau, siteTarget, "never_executed", cause,
                      site.path);
    return Error::success();
  }
  const char *cause =
      "the profiled program did not contain this site (it was annotated "
      "after the profiling run, is in a translation unit the profiled "
      "program did not link, or was renamed)";
  // Strict mode leaves a site unchanged only for a cause that says the site
  // did not run.
  if (S.Opts.Model.StrictMode)
    return userError(*CI, Twine("no profile of ") + body.getName() + " at " +
                              site.path + ": " + cause +
                              ", and -poseidon-strict-mode does not leave a "
                              "site unchanged");
  warn(CI, Twine("no profile at ") + site.path + ": " + cause +
               "; leaving the site unchanged");
  explainUnprofiled(S, clone, tau, siteTarget, "unprofiled", cause, site.path);
  return Error::success();
}

// Tells every technique that the host launches the solved site `body` through
// the kernel `CI` is in, with `primalArgs`.
static void noteSiteLaunch(Session &S, CallBase *CI, const Function &body,
                           ArrayRef<Value *> primalArgs) {
  for (const Technique &T : techniques())
    if (T.SiteLaunched)
      T.SiteLaunched(S, body, *CI->getFunction(), primalArgs);
}

// Profile use, shared by both annotation forms: `CI` is the call the site's
// body is reached through (the marker call, or the wrapper kernel's call of its
// own outlined body), `F` the body and `primalArgs` the caller values its
// parameters take.
static Error optimizeSiteBody(Session &S, CallBase *CI, Function *F,
                              SmallVectorImpl<Value *> &primalArgs, double tau,
                              SmallVectorImpl<CallBase *> &calls) {
  const Options &O = S.Opts;
  const bool profileUse = O.Profiling.usesProfile();
  if (!profileUse) {
    warn(CI, kNoProfileAction);
  } else {
    if (Error E = noteSiteForPins(S, *F))
      return E;
    auto key = std::make_pair((const Function *)F, tau);
    auto prior = S.SolvedSites.find(key);
    if (!O.Solver.SolveJointly && prior != S.SolvedSites.end())
      if (auto *done = dyn_cast_or_null<Function>(prior->second)) {
        if (done != F)
          noteRewritten(S, *CI->getFunction());
        noteSiteLaunch(S, CI, *done, primalArgs);
        if (O.Diagnostics.Print)
          llvm::errs() << "[poseidon] " << F->getName()
                       << ": same body and target as an earlier site; reusing "
                          "its solve ("
                       << done->getName() << ")\n";
        emitSiteCall(CI, done, primalArgs, calls);
        return Error::success();
      }

    // A site the solve applies nothing to calls the original body: the
    // canonical clone's lowering is not guaranteed bit-identical to it.
    Function *sourceBody = F;
    F = canonicalize(*F);
    S.SiteClones.push_back(F);
    setSlotMetadata(*F);
    noteSiteClone(S.Sites, F, sourceBody, primalArgs);
    S.SiteCalls[F] = CI->getDebugLoc();

    // Profile generation recorded nothing for a site without an operation to
    // rewrite, so the absent profile says nothing about whether it ran.
    const double target = tau > 0.0 ? tau : (double)O.Solver.Tau;
    if (none_of(instructions(*F),
                [](const Instruction &I) { return isOptimizable(I); })) {
      warn(CI, Twine(sourceBody->getName()) +
                   " holds no floating-point operation the pass rewrites; "
                   "leaving the site unchanged");
      explainLeftAlone(S, *F, target, tau > 0.0, "no_fp_operation",
                       "the site holds no floating-point operation the pass "
                       "rewrites");
      markOutlineTransparent(sourceBody);
      emitSiteCall(CI, sourceBody, primalArgs, calls);
      return Error::success();
    }

    // A site with nothing to solve from still has to build and run, so it keeps
    // the original body rather than failing the compile.
    Expected<std::optional<Unprofiled>> site = unprofiledSite(S, CI, *F);
    if (!site)
      return site.takeError();
    if (*site) {
      if (Error E = reportUnprofiled(S, CI, *sourceBody, *F, **site, target,
                                     tau > 0.0))
        return E;
      markOutlineTransparent(sourceBody);
      emitSiteCall(CI, sourceBody, primalArgs, calls);
      return Error::success();
    }

    if (O.Diagnostics.Print)
      llvm::errs() << "[poseidon] Optimizing " << F->getName()
                   << " with relative error tolerance: " << target << '\n';

    if (O.Solver.SolveJointly) {
      F->addFnAttr(kJointTau, exactDouble(tau));
      // The joint solve materializes F later; the wrapper's arguments are
      // recorded now, while primalArgs holds them.
      noteSiteLaunch(S, CI, *F, primalArgs);
    } else {
      Expected<bool> optimized = optimizeSite(S, *F, tau);
      if (!optimized)
        return optimized.takeError();
      if (*optimized)
        noteRewritten(S, *CI->getFunction());

      noteSiteLaunch(S, CI, *F, primalArgs);

      // The canonical clone's lowering may differ from the original body's at
      // ulp level, so a site with no rewrite calls the original body.
      if (!*optimized) {
        if (O.Diagnostics.Print)
          llvm::errs() << "[poseidon] no rewrite applied for " << F->getName()
                       << "; calling original body " << sourceBody->getName()
                       << " (bit-transparent no-op)\n";
        F = sourceBody;
      }
      S.SolvedSites[key] = F;
    }
  }

  if (profileUse)
    markOutlineTransparent(F);
  emitSiteCall(CI, F, primalArgs, calls);
  return Error::success();
}

static Expected<bool> optimizeMarker(Session &S, CallInst *CI,
                                     SmallVectorImpl<CallBase *> &calls) {
  Expected<Function *> F = markerTarget(CI);
  if (!F)
    return F.takeError();
  Expected<MarkerArgs> marker = parseMarker(CI, *F);
  if (!marker)
    return marker.takeError();
  if (Error E =
          optimizeSiteBody(S, CI, *F, marker->primalArgs, marker->tau, calls))
    return std::move(E);
  return true;
}

// Profile use of a host site: the body is solved once, and every call of it,
// the wrapper's and the ones the inliner placed in F's callers, is redirected
// to what the solve chose; a caller redirected to a rewrite is a function the
// rewrite reached.
static Expected<bool> optimizeHostBody(Session &S, const Site &site) {
  Function &B = *site.fn;
  releaseHostBody(B);
  markOutlineTransparent(&B);
  SmallVector<CallBase *, 4> calls = callsOf(B);
  CallBase *rep = calls.empty() ? nullptr : calls.front();
  for (CallBase *CB : calls)
    if (isa<CallInst>(CB) && (!isa<CallInst>(rep) ||
                              CB->getFunction()->hasFnAttribute(kSiteWrapper)))
      rep = CB;
  if (!rep)
    return false;
  llvm::erase(calls, rep);
  SmallVector<Value *, 8> primalArgs(rep->args());
  SmallVector<CallBase *, 1> lowered;
  if (Error E = optimizeSiteBody(S, rep, &B, primalArgs,
                                 site.target.value_or(0.0), lowered))
    return std::move(E);
  Function *target = lowered.back()->getCalledFunction();
  for (CallBase *CB : calls) {
    CB->setCalledFunction(target);
    if (target != &B)
      noteRewritten(S, *CB->getFunction());
  }
  return true;
}

// What the profiled run spent inside this site: over the slots the profile
// records, the execution count times the CSV price of the operation the slot
// names. A site the run never reached has no records and costs nothing; a
// profile recorded against another canonical form is an error at the site.
static Expected<double> profiledSiteCost(Session &S, Function &body) {
  SmallString<128> profilePath(S.Opts.Profiling.ProfileUse);
  llvm::sys::path::append(profilePath, bodyProfileFile(S.Sites, body));
  if (!llvm::sys::fs::exists(profilePath.str()))
    return 0.0;

  ProfileMap profileMap;
  SiteProfileHeader header;
  if (Error E =
          parseProfileFile(profilePath.str().str(), body, profileMap, header))
    return std::move(E);
  if (profileMap.empty())
    return 0.0;

  // The slots are numbered against the canonical form, so the prices have to be
  // read off it too. The clone is erased before the site's own clone is made
  // under the same name.
  Function *clone = canonicalize(body);
  setSlotMetadata(*clone);
  Expected<double> cost = [&]() -> Expected<double> {
    if (Error E = requireCanonicalHash(header, canonicalFormHash(*clone),
                                       profilePath, *clone))
      return std::move(E);
    if (Error E = requireCostModel(S.CostModel, *clone))
      return std::move(E);
    return profiledCost(S.CostModel, *clone, profileMap, slotOf);
  }();
  // An error names the clone, so the clone outlives it.
  if (!cost)
    return cost.takeError();
  clone->eraseFromParent();
  return cost;
}

// Profile use of the kernel sites: each kernel becomes a wrapper around its
// own body, and that body is solved and rewritten like a marked one.
static Expected<bool> optimizeKernels(Session &Sess,
                                      ArrayRef<const Site *> kernels,
                                      SmallVectorImpl<CallBase *> &calls) {
  const Options &O = Sess.Opts;
  if (kernels.empty())
    return false;

  // Without a profile nothing can be applied, so the kernels are left exactly
  // as written rather than outlined and never folded back.
  if (!O.Profiling.usesProfile()) {
    warnSitesWithoutProfile(Sess.Opts, Sess.Sites,
                            *kernels.front()->fn->getParent());
    return false;
  }

  // Only a target the annotation carried is the site's own; the others are
  // solved for -poseidon-tau.
  auto optimize = [&](CallInst *call, Function *body, const Site &site) {
    SmallVector<Value *, 8> primalArgs(call->args());
    noteSiteTarget(O, *site.fn, site.target);
    return optimizeSiteBody(Sess, call, body, primalArgs,
                            site.target.value_or(0.0), calls);
  };

  const bool filter =
      O.Profiling.MinCostShare > 0.0 &&
      any_of(kernels, [](const Site *site) { return site->byRegex; });
  bool changed = false;
  if (!filter) {
    for (const Site *site : kernels) {
      Function *body = nullptr;
      Expected<CallInst *> call =
          outlineKernelBody(Sess.Sites, *site->fn, body);
      if (!call)
        return call.takeError();
      markOutlineTransparent(body);
      if (Error E = optimize(*call, body, *site))
        return std::move(E);
      changed = true;
    }
    return changed;
  }

  // A site's share is only known once every regex-named site has been priced,
  // so with the filter on they are all outlined and priced before any of them
  // is optimized.
  struct Outlined {
    const Site *site;
    Function *body;
    CallInst *call;
    double cost;
  };
  SmallVector<Outlined, 4> outlined;
  double regexTotal = 0.0;
  for (const Site *site : kernels) {
    Function *body = nullptr;
    Expected<CallInst *> call = outlineKernelBody(Sess.Sites, *site->fn, body);
    if (!call)
      return call.takeError();
    markOutlineTransparent(body);
    double cost = 0.0;
    if (site->byRegex) {
      Expected<double> priced = profiledSiteCost(Sess, *body);
      if (!priced)
        return priced.takeError();
      cost = *priced;
    }
    regexTotal += cost;
    outlined.push_back({site, body, *call, cost});
  }

  for (Outlined &K : outlined) {
    double share = regexTotal > 0.0 ? K.cost / regexTotal : 1.0;
    if (K.site->byRegex && share < O.Profiling.MinCostShare) {
      const std::string below =
          formatv("{0}% of the profiled FP cost of the kernels "
                  "-poseidon-kernels named, below -poseidon-min-cost-share={1}",
                  format("%.4g", share * 100),
                  format("%g", (double)O.Profiling.MinCostShare))
              .str();
      if (O.Diagnostics.Print)
        llvm::errs() << "[poseidon] " << K.site->fn->getName() << " is "
                     << below << "; left unoptimized\n";
      explainLeftAlone(Sess, *K.body,
                       K.site->target.value_or((double)O.Solver.Tau),
                       K.site->target.has_value(), "below_cost_share",
                       "the site is " + below,
                       /*profile=*/"",
                       /*body=*/true);
      inlineKernelBody(K.call, K.body);
      continue;
    }
    if (Error E = optimize(K.call, K.body, *K.site))
      return std::move(E);
    changed = true;
  }
  return changed;
}

Expected<bool> optimizeSites(Session &S, ArrayRef<Site> sites,
                             SmallVectorImpl<CallBase *> &calls) {
  bool changed = false;
  SmallVector<const Site *, 4> kernels;
  for (const Site &site : sites)
    if (site.form == Site::Form::Kernel)
      kernels.push_back(&site);
  for (const Site &site : sites) {
    Expected<bool> optimized = false;
    switch (site.form) {
    case Site::Form::Marker:
      optimized = optimizeMarker(S, site.marker, calls);
      break;
    case Site::Form::HostBody:
      optimized = optimizeHostBody(S, site);
      break;
    case Site::Form::Kernel:
      // The kernels are solved together, where the first of them stands.
      if (&site == kernels.front())
        optimized = optimizeKernels(S, kernels, calls);
      break;
    }
    if (!optimized)
      return optimized.takeError();
    changed |= *optimized;
  }
  return changed;
}

Expected<bool> prepareModule(Session &S, Module &M) {
  if (!S.Opts.Profiling.generatesProfile())
    return false;
  injectHostProfilerInit(M, S.Opts.Diagnostics.Print);
  predeclareInactiveProfilerProbes(M);

  DenseMap<Function *, Function *> clones;
  unsigned nextSiteId = getNextSiteId(M);
  bool anySite = false, anyHostBody = false;
  for (const Site &site : collectSites(
           S.Opts, M,
           {Site::Form::Marker, Site::Form::Kernel, Site::Form::HostBody})) {
    switch (site.form) {
    case Site::Form::Marker:
      // A site whose body is still a derivative request is the late run's;
      // its marker stays where it is.
      if (deferSite(S.Opts, site.marker))
        continue;
      if (Error E = profileSite(S, site, clones, nextSiteId).takeError())
        return std::move(E);
      anySite = true;
      break;
    case Site::Form::HostBody:
      anyHostBody = true;
      [[fallthrough]];
    case Site::Form::Kernel: {
      Expected<bool> profiled = profileSite(S, site, clones, nextSiteId);
      if (!profiled)
        return profiled.takeError();
      anySite |= *profiled;
      break;
    }
    }
  }
  if (anyHostBody) {
    registerStackArrays(S, M);
    registerGlobals(M);
    // The profiling runtime records allocations only in a program that links
    // a host site; this definition is what it looks for.
    auto *present = new GlobalVariable(
        M, Type::getInt8Ty(M.getContext()), true, GlobalValue::WeakAnyLinkage,
        ConstantInt::get(Type::getInt8Ty(M.getContext()), 1),
        "__poseidon_host_sites_present");
    present->setVisibility(GlobalValue::HiddenVisibility);
  }
  setNextSiteId(M, nextSiteId);

  if (anySite) {
    // Launch geometry is recorded per profiled function; the probe is keyed by
    // the same poseidon.site.<name> global the value probes carry, so it has to
    // be placed after they exist.
    injectBlockDimProbes(M, S.Opts.Diagnostics.Print);
  }
  return anySite;
}

// Gives each host wrapper without a subprogram the one most of its inlined
// code is in, when no other function holds it, and moves the wrapper's own
// locations into that subprogram.
static void restoreHostWrapperDebugInfo(Module &M, bool profileGenerate) {
  for (Function &F : M) {
    if (!F.hasFnAttribute(kSiteWrapper))
      continue;
    F.removeFnAttr(kSiteWrapper);
    if (F.getSubprogram() || profileGenerate)
      continue;
    MapVector<DISubprogram *, unsigned> uses;
    for (Instruction &I : instructions(F))
      if (const DebugLoc &DL = I.getDebugLoc())
        ++uses[DL->getInlinedAtScope()->getSubprogram()];
    DISubprogram *SP = nullptr;
    for (auto &[sp, n] : uses)
      if (!SP || n > uses[SP])
        SP = sp;
    if (!SP ||
        llvm::any_of(M, [&](Function &G) { return G.getSubprogram() == SP; }))
      continue;
    F.setSubprogram(SP);
    for (Instruction &I : instructions(F))
      if (const DebugLoc &DL = I.getDebugLoc())
        if (!DL->getInlinedAt() && DL->getScope()->getSubprogram() != SP)
          I.setDebugLoc(
              DILocation::get(F.getContext(), DL.getLine(), DL.getCol(), SP));
  }
}

Error finalizeModule(Session &S, Module &M) {
  if (Error E = checkPinnedSites(S, M))
    return E;
  for (const Technique &T : techniques())
    if (T.Report)
      T.Report(S);
  // A site whose solve applied nothing calls the original body again, and an
  // applied site's wrapper call may have been inlined away, in both cases
  // leaving a canonicalized clone nothing calls.
  for (WeakVH &handle : S.SiteClones) {
    auto *clone = dyn_cast_or_null<Function>(handle);
    if (clone && clone->getParent() == &M && clone->use_empty())
      clone->eraseFromParent();
  }
  S.SiteClones.clear();
  S.SolvedSites.clear();
  restoreHostWrapperDebugInfo(M, S.Opts.Profiling.generatesProfile());
  // The site counter is a note the early run leaves for the late one; nothing
  // downstream reads it.
  if (auto *NMD = M.getNamedMetadata(kNextSiteIdMD))
    NMD->eraseFromParent();
  return Error::success();
}

} // namespace poseidon
