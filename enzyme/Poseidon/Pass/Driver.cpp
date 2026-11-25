//===- Driver.cpp - the site actions of the Poseidon passes ---------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Driver.h"

#include "IR/Utils.h"
#include "Model/CostModel.h"
#include "Pass/Session.h"
#include "Profile/Instrument.h"
#include "Profile/ProfileRead.h"
#include "Profile/SiteProfile.h"
#include "Sites/Annotations.h"
#include "Sites/Canonicalize.h"
#include "Sites/Markers.h"
#include "Sites/SiteIdentity.h"
#include "Solve/Explain.h"
#include "Solve/Optimize.h"
#include "Techniques/Herbie.h"
#include "Techniques/Matmul/HostDispatch.h"

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
static void requireProfileDirectory(CallBase *CI, StringRef dir,
                                    StringRef flag) {
  const bool isDir = sys::fs::is_directory(dir);
  std::error_code EC;
  if (isDir)
    for (sys::fs::directory_iterator it(dir, EC), end; !EC && it != end;
         it.increment(EC))
      if (sys::path::extension(it->path()) == kProfileFileSuffix)
        return;
  reportUserError(*CI,
                  Twine("the profile directory ") + dir + " (" + flag + ") " +
                      (isDir ? "holds no .fpprofile file" : "does not exist") +
                      "; no profiling run wrote into it. Run the "
                      "program built with -poseidon-profile-generate "
                      "first, or pass the directory it wrote");
}

namespace {
// Why a site has no records to solve from, read from every directory that
// holds a profile of it.
struct Unprofiled {
  bool neverExecuted;
  std::string path;
};
} // namespace

static std::optional<Unprofiled> unprofiledSite(Session &S, CallBase *CI,
                                                Function &clone) {
  std::optional<Unprofiled> absent, neverExecuted;
  auto read = [&](StringRef dir, StringRef flag) {
    const std::string path = siteProfilePath(S, clone, dir);
    if (!sys::fs::exists(path)) {
      requireProfileDirectory(CI, dir, flag);
      if (!absent)
        absent = Unprofiled{false, path};
      return;
    }
    ProfileMap records;
    FunctionProfileHeader header;
    parseProfileFile(path, clone, records, header);
    if (header.executed)
      return;
    requireCanonicalHash(header, canonicalFormHash(clone), path, clone);
    if (!neverExecuted)
      neverExecuted = Unprofiled{true, path};
  };
  read(S.Opts.Profiling.ProfileUse, "-poseidon-profile-use");
  for (const std::string &dir : S.Opts.Profiling.OutputProfiles)
    read(dir, "-poseidon-output-profiles");
  return absent ? absent : neverExecuted;
}

// `tau` > 0 is the site's accuracy target, its own when `siteTarget`.
static void reportUnprofiled(Session &S, CallBase *CI, const Function &body,
                             Function &clone, const Unprofiled &site,
                             double tau, bool siteTarget) {
  if (site.neverExecuted) {
    const char *cause =
        "the profiling run instrumented this site and never executed it";
    llvm::errs() << "[poseidon] " << body.getName() << ": " << cause << " ("
                 << site.path << "); left unchanged\n";
    explainUnprofiled(S, clone, tau, siteTarget, "never_executed", cause,
                      site.path);
    return;
  }
  const char *cause =
      "the profiled program did not contain this site (it was annotated "
      "after the profiling run, is in a translation unit the profiled "
      "program did not link, or was renamed)";
  // Strict mode leaves a site unchanged only for a cause that says the site
  // did not run.
  if (S.Opts.Model.StrictMode) {
    fail(CI, Twine("no profile of ") + body.getName() + " at " + site.path +
                 ": " + cause +
                 ", and -poseidon-strict-mode does not leave a site unchanged");
    return;
  }
  warn(CI, Twine("no profile at ") + site.path + ": " + cause +
               "; leaving the site unchanged");
  llvm::errs() << "[poseidon] " << body.getName()
               << ": the profiled program did not contain this site (no "
                  "profile at "
               << site.path << "); left unchanged\n";
  explainUnprofiled(S, clone, tau, siteTarget, "unprofiled", cause, site.path);
}

static void writeGemmDescriptorOf(Session &S, CallBase *CI, Function *body,
                                  ArrayRef<Value *> primalArgs) {
  if (!S.Opts.Matmul.OzakiHostDispatch)
    return;
  if (auto note = S.Matmul.GemmBodies.find(body);
      note != S.Matmul.GemmBodies.end() && note->second.valid)
    writeGemmDescriptor(S, *CI->getFunction(), primalArgs, note->second);
}

// Profile use, shared by both annotation forms: `CI` is the call the site's
// body is reached through (the marker call, or the wrapper kernel's call of its
// own outlined body), `F` the body and `primalArgs` the caller values its
// parameters take.
static void optimizeSiteBody(Session &S, CallBase *CI, Function *F,
                             SmallVectorImpl<Value *> &primalArgs,
                             double errTol,
                             SmallVectorImpl<CallBase *> &calls) {
  const Options &O = S.Opts;
  const bool profileUse = O.Profiling.usesProfile();
  if (!profileUse) {
    warn(CI, Twine(kNoProfileAction) + " Emitting the original computation.");
  } else {
    noteSiteForPins(S, *F);
    auto key = std::make_pair((const Function *)F, errTol);
    auto prior = S.SolvedSites.find(key);
    if (!O.Solver.JointDP && prior != S.SolvedSites.end())
      if (auto *done = dyn_cast_or_null<Function>(prior->second)) {
        if (done != F)
          noteRewritten(S, *CI->getFunction());
        writeGemmDescriptorOf(S, CI, done, primalArgs);
        llvm::errs() << "[poseidon] " << F->getName()
                     << ": same body and target as an earlier site; reusing "
                        "its solve ("
                     << done->getName() << ")\n";
        emitSiteCall(CI, done, primalArgs, calls);
        return;
      }

    // A site the solve applies nothing to calls the original body: the
    // canonical clone's lowering is not guaranteed bit-identical to it.
    Function *sourceBody = F;
    F = canonicalize(*F);
    S.SiteClones.push_back(F);
    setSlotMetadata(*F);
    noteSiteOrigin(S, F, sourceBody);
    noteSiteArgs(S, F, primalArgs);
    S.SiteCalls[F] = CI->getDebugLoc();

    // Profile generation recorded nothing for a site without an operation to
    // rewrite, so the absent profile says nothing about whether it ran.
    const double tau = errTol > 0.0 ? errTol : (double)O.Solver.Tau;
    if (none_of(instructions(*F),
                [](const Instruction &I) { return isOptimizable(I); })) {
      warn(CI, Twine(sourceBody->getName()) +
                   " holds no floating-point operation the pass rewrites; "
                   "leaving the site unchanged");
      explainLeftAlone(S, *F, tau, errTol > 0.0, "no_fp_operation",
                       "the site holds no floating-point operation the pass "
                       "rewrites");
      markOutlineTransparent(sourceBody);
      emitSiteCall(CI, sourceBody, primalArgs, calls);
      return;
    }

    // A site with nothing to solve from still has to build and run, so it keeps
    // the original body rather than failing the compile.
    if (std::optional<Unprofiled> site = unprofiledSite(S, CI, *F)) {
      reportUnprofiled(S, CI, *sourceBody, *F, *site, tau, errTol > 0.0);
      markOutlineTransparent(sourceBody);
      emitSiteCall(CI, sourceBody, primalArgs, calls);
      return;
    }

    if (O.Diagnostics.Print)
      llvm::errs() << "[poseidon] Optimizing " << F->getName()
                   << " with relative error tolerance: " << tau << '\n';

    if (O.Solver.JointDP) {
      F->addFnAttr(kJointErrTol, exactDouble(errTol));
      // The joint solve materializes F later; the wrapper's arguments are
      // recorded now, while primalArgs holds them.
      if (O.Matmul.OzakiHostDispatch)
        addPendingGemmDispatch(S, F, *CI->getFunction(), primalArgs);
    } else {
      bool optimized = fpOptimize(S, *F, errTol);
      if (optimized)
        noteRewritten(S, *CI->getFunction());

      if (!optimized)
        warn(CI,
             Twine("Poseidon returned false (no change) for ") + F->getName());

      writeGemmDescriptorOf(S, CI, F, primalArgs);

      // The canonical clone's lowering may differ from the original body's at
      // ulp level, so a site with no rewrite calls the original body.
      if (!optimized) {
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
}

static bool optimizeSite(Session &S, CallInst *CI,
                         SmallVectorImpl<CallBase *> &calls) {
  Function *F = markerTarget(CI);
  if (!F)
    return false;

  std::optional<MarkerArgs> marker = parseMarker(CI, F);
  if (!marker)
    return false;
  optimizeSiteBody(S, CI, F, marker->primalArgs, marker->errTol, calls);
  return true;
}

// Profile use of a host site: the body is solved once, and every call of it,
// the wrapper's and the ones the inliner placed in F's callers, is redirected
// to what the solve chose; a caller redirected to a rewrite is a function the
// rewrite reached.
static bool optimizeHostSite(Session &S, Function &B) {
  double errTol = takeHostBody(B);
  markOutlineTransparent(&B);
  SmallVector<CallBase *, 4> calls = callsOf(B);
  CallBase *rep = calls.empty() ? nullptr : calls.front();
  for (CallBase *CB : calls)
    if (isa<CallInst>(CB) && (!isa<CallInst>(rep) ||
                              CB->getFunction()->hasFnAttribute(kHostWrapper)))
      rep = CB;
  if (!rep)
    return false;
  llvm::erase(calls, rep);
  SmallVector<Value *, 8> primalArgs(rep->args());
  SmallVector<CallBase *, 1> lowered;
  optimizeSiteBody(S, rep, &B, primalArgs, errTol, lowered);
  Function *target = lowered.back()->getCalledFunction();
  for (CallBase *CB : calls) {
    CB->setCalledFunction(target);
    if (target != &B)
      noteRewritten(S, *CB->getFunction());
  }
  return true;
}

bool lowerMarkers(Session &S, ArrayRef<CallInst *> markers,
                  SmallVectorImpl<CallBase *> &calls) {
  bool changed = false;
  for (CallInst *CI : markers)
    changed |= optimizeSite(S, CI, calls);
  return changed;
}

// What the profiled run spent inside this site: over the slots the profile
// records, the execution count times the CSV price of the operation the slot
// names. A site the run never reached has no records and costs nothing; a
// profile recorded against another canonical form is an error at the site.
static double profiledSiteCost(Session &S, Function &body) {
  SmallString<128> profilePath(S.Opts.Profiling.ProfileUse);
  llvm::sys::path::append(profilePath, bodyProfileFile(body));
  if (!llvm::sys::fs::exists(profilePath.str()))
    return 0.0;

  ProfileMap profileMap;
  FunctionProfileHeader header;
  parseProfileFile(profilePath.str().str(), body, profileMap, header);
  if (profileMap.empty())
    return 0.0;

  // The slots are numbered against the canonical form, so the prices have to be
  // read off it too. The clone is erased before the site's own clone is made
  // under the same name.
  Function *clone = canonicalize(body);
  setSlotMetadata(*clone);
  requireCanonicalHash(header, canonicalFormHash(*clone), profilePath, *clone);
  requireCostModel(S.CostModel, *clone);
  double cost = profiledCost(S.CostModel, *clone, profileMap,
                             [](const Instruction &I, size_t &slot) {
                               return tryReadProfIdxMetadata(&I, slot);
                             });
  clone->eraseFromParent();
  return cost;
}

bool optimizeAnnotatedSites(Session &Sess, Module &M) {
  const Options &O = Sess.Opts;
  if (O.Profiling.ProfileGenerate)
    return false;
  bool changed = false;
  for (Function *B : hostSiteBodies(M))
    changed |= optimizeHostSite(Sess, *B);

  SmallVector<Function *, 4> sites;
  SmallPtrSet<Function *, 4> byRegex;
  DenseMap<Function *, double> siteTau;
  collectAnnotatedSites(O, M, sites, &byRegex, &siteTau);
  if (sites.empty())
    return changed;

  // Without a profile nothing can be applied, so the kernels are left exactly
  // as written rather than outlined and never folded back.
  if (!O.Profiling.usesProfile()) {
    warnSitesWithoutProfile(Sess, M);
    return changed;
  }

  SmallVector<CallBase *, 4> calls;
  // Only a target the annotation carried is the site's own; the others are
  // solved for -poseidon-tau.
  auto optimize = [&](CallInst *call, Function *body, Function *kernel) {
    SmallVector<Value *, 8> primalArgs(call->args());
    auto it = siteTau.find(kernel);
    double errTol = it != siteTau.end() ? it->second : 0.0;
    noteSiteTarget(O, *kernel,
                   it != siteTau.end() ? std::optional<double>(errTol)
                                       : std::nullopt);
    optimizeSiteBody(Sess, call, body, primalArgs, errTol, calls);
    changed = true;
  };

  if (byRegex.empty() || O.Profiling.MinCostShare <= 0.0) {
    for (Function *K : sites) {
      Function *body = nullptr;
      if (CallInst *call = outlineKernelBody(*K, body)) {
        markOutlineTransparent(body);
        optimize(call, body, K);
      }
    }
    return changed;
  }

  // A site's share is only known once every regex-named site has been priced,
  // so with the filter on they are all outlined and priced before any of them
  // is optimized.
  struct Site {
    Function *kernel;
    Function *body;
    CallInst *call;
    double cost;
  };
  SmallVector<Site, 4> outlined;
  double regexTotal = 0.0;
  for (Function *K : sites) {
    Function *body = nullptr;
    CallInst *call = outlineKernelBody(*K, body);
    if (!call)
      continue;
    markOutlineTransparent(body);
    double cost = byRegex.count(K) ? profiledSiteCost(Sess, *body) : 0.0;
    regexTotal += cost;
    outlined.push_back({K, body, call, cost});
  }

  for (Site &S : outlined) {
    double share = regexTotal > 0.0 ? S.cost / regexTotal : 1.0;
    if (byRegex.count(S.kernel) && share < O.Profiling.MinCostShare) {
      const std::string below =
          formatv("{0}% of the profiled FP cost of the kernels "
                  "-poseidon-kernels named, below -poseidon-min-cost-share={1}",
                  format("%.4g", share * 100),
                  format("%g", (double)O.Profiling.MinCostShare))
              .str();
      llvm::errs() << "[poseidon] " << S.kernel->getName() << " is " << below
                   << "; left unoptimized\n";
      auto target = siteTau.find(S.kernel);
      const bool siteTarget = target != siteTau.end();
      explainLeftAlone(Sess, *S.body,
                       siteTarget ? target->second : O.Solver.Tau, siteTarget,
                       "below_cost_share", "the site is " + below,
                       /*profile=*/"",
                       /*body=*/true);
      inlineKernelBody(S.call, S.body);
      continue;
    }
    optimize(S.call, S.body, S.kernel);
  }
  return changed;
}

bool prepareModule(Session &S, Module &M) {
  if (!S.Opts.Profiling.ProfileGenerate)
    return false;
  injectHostProfilerInit(M, S.Opts.Diagnostics.Print);
  predeclareInactiveProfilerProbes(M);

  SmallVector<CallInst *, 4> markers;
  collectMarkerCalls(M, markers);
  DenseMap<Function *, Function *> clones;
  unsigned nextSiteId = getNextSiteId(M);
  bool anySite = false;
  for (CallInst *CI : markers) {
    // A site whose body is still a derivative request is the late run's; its
    // marker stays where it is.
    if (deferSite(S.Opts, CI))
      continue;
    profileSite(S, CI, clones, nextSiteId);
    anySite = true;
  }

  SmallVector<Function *, 4> annotated;
  collectAnnotatedSites(S.Opts, M, annotated);
  for (Function *K : annotated)
    anySite |= profileAnnotatedKernel(S, *K, nextSiteId);
  SmallVector<Function *, 4> hostBodies = hostSiteBodies(M);
  for (Function *B : hostBodies)
    anySite |= profileHostSite(S, *B, nextSiteId);
  if (!hostBodies.empty()) {
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
    // the same poseidon_site_<name> global the value probes carry, so it has to
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
    if (!F.hasFnAttribute(kHostWrapper))
      continue;
    F.removeFnAttr(kHostWrapper);
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

void finalizeModule(Session &S, Module &M) {
  checkPinnedSites(S, M);
  reportHerbieAnswers(S);
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
  restoreHostWrapperDebugInfo(M, S.Opts.Profiling.ProfileGenerate);
  // The site counter is a note the early run leaves for the late one; nothing
  // downstream reads it.
  if (auto *NMD = M.getNamedMetadata(kNextSiteIdMD))
    NMD->eraseFromParent();
}

} // namespace poseidon
