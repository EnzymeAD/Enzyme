#include "Driver.h"

#include "CostModel.h"
#include "Optimize.h"
#include "Pass/Explain.h"
#include "Pass/Session.h"
#include "Profile/ProfileRead.h"
#include "Profile/SiteProfile.h"
#include "Sites/Annotations.h"
#include "Sites/Canonicalize.h"
#include "Sites/HostSites.h"
#include "Sites/KernelSites.h"
#include "Sites/Markers.h"
#include "Solvers.h"
#include "Techniques/Matmul/HostDispatch.h"
// The one profile-filename rule, shared verbatim with both FP profiler
// runtimes; the existence check below must look where they wrote.
#include "FPProfileName.h"

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
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>
#include <unordered_map>

using namespace llvm;

namespace poseidon {

namespace {

void noteRewritten(Session &S, Function &F) {
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
bool emitSiteCall(CallBase *CI, Function *F,
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

  return true;
}

// A profile-use compile pointed at a directory no profiling run wrote into
// would otherwise leave every site as written and succeed.
void requireProfileDirectory(StringRef dir, StringRef flag) {
  const bool isDir = sys::fs::is_directory(dir);
  std::error_code EC;
  if (isDir)
    for (sys::fs::directory_iterator it(dir, EC), end; !EC && it != end;
         it.increment(EC))
      if (sys::path::extension(it->path()) == ".fpprofile")
        return;
  report_fatal_error(
      Twine("Poseidon: the profile directory ") + dir + " (" + flag + ") " +
          (isDir ? "holds no .fpprofile file" : "does not exist") +
          "; no profiling run wrote into it",
      /*gen_crash_diag=*/false);
}

// Why a site has no records to solve from, read from every directory that
// holds a profile of it.
struct Unprofiled {
  bool neverExecuted;
  std::string path;
  // Absent from a directory whose profiles name no format: the run that wrote
  // it left no file for a site it never executed.
  bool mayNotHaveRun = false;
};

std::optional<Unprofiled> unprofiledSite(Session &S, Function &clone) {
  const std::string file = siteProfileStem(S, clone) + ".fpprofile";
  std::optional<Unprofiled> absent, neverExecuted;
  auto read = [&](StringRef dir, StringRef flag) {
    SmallString<128> path(dir);
    sys::path::append(path, file);
    if (!sys::fs::exists(path)) {
      requireProfileDirectory(dir, flag);
      if (!absent)
        absent = Unprofiled{false, path.str().str(),
                            !profileDirNamesFormat(S.ProfileDirFormats, dir)};
    } else if (isNeverExecutedProfile(path.str().str())) {
      std::unordered_map<size_t, ProfileInfo> records;
      FunctionProfileHeader header;
      parseProfileFile(path.str().str(), records, &header);
      requireCanonicalHash(header, canonicalFormHash(clone), path,
                           clone.getName());
      if (!neverExecuted)
        neverExecuted = Unprofiled{true, path.str().str()};
    }
  };
  read(S.Opts.Profiling.ProfileUse, "-poseidon-profile-use");
  for (const std::string &dir : S.Opts.Profiling.OutputProfiles)
    read(dir, "-poseidon-output-profiles");
  return absent ? absent : neverExecuted;
}

void reportUnprofiled(Session &S, CallBase *CI, const Function &body,
                      Function &clone, const Unprofiled &site) {
  if (site.neverExecuted) {
    const char *cause =
        "the profiling run instrumented this site and never executed it";
    llvm::errs() << "[poseidon] " << body.getName() << ": " << cause << " ("
                 << site.path << "); left unchanged\n";
    explainUnprofiled(S, clone, cause);
    return;
  }
  if (site.mayNotHaveRun) {
    const char *cause =
        "the profile directory was written before the profiling runtime "
        "recorded sites that never ran, so this site may simply not have run "
        "(or the profiled program did not contain it)";
    warn(CI, Twine("no profile at ") + site.path + ": " + cause +
                 "; leaving the site unchanged");
    llvm::errs() << "[poseidon] " << body.getName() << ": " << cause
                 << " (no profile at " << site.path << "); left unchanged\n";
    explainUnprofiled(S, clone, cause);
    return;
  }
  const char *cause =
      "the profiled program did not contain this site (it was annotated "
      "after the profiling run, is in a translation unit the profiled "
      "program did not link, or was renamed)";
  if (S.Opts.Model.StrictMode)
    report_fatal_error(Twine("Poseidon: no profile of ") + body.getName() +
                           " at " + site.path + ": " + cause +
                           ", and -poseidon-strict-mode does not leave a "
                           "site unchanged",
                       /*gen_crash_diag=*/false);
  warn(CI, Twine("no profile at ") + site.path + ": " + cause +
               "; leaving the site unchanged");
  llvm::errs() << "[poseidon] " << body.getName()
               << ": the profiled program did not contain this site (no "
                  "profile at "
               << site.path << "); left unchanged\n";
  explainUnprofiled(S, clone, cause);
}

// Profile use, shared by both annotation forms: `CI` is the call the site's
// body is reached through (the marker call, or the wrapper kernel's call of its
// own outlined body), `F` the body and `primalArgs` the caller values its
// parameters take.
bool optimizeSiteBody(Session &S, CallBase *CI, Function *F,
                      SmallVectorImpl<Value *> &primalArgs, double errTol,
                      SmallVectorImpl<CallBase *> &calls) {
  const Options &O = S.Opts;
  const bool profileUse = O.Profiling.usesProfile();
  if (!profileUse) {
    warn(CI, "a Poseidon site was compiled without -poseidon-profile-generate "
             "or -poseidon-profile-use=<dir>. Emitting the original "
             "computation.");
  } else {
    noteSiteForPins(S, *F);
    auto key = std::make_pair((const Function *)F, errTol);
    auto prior = S.SolvedSites.find(key);
    if (!O.Solver.JointDP && prior != S.SolvedSites.end())
      if (auto *done = dyn_cast_or_null<Function>(prior->second)) {
        if (done != F)
          noteRewritten(S, *CI->getFunction());
        GemmBodyNote ozNote;
        if (O.Matmul.OzakiHostDispatch && getGemmBody(S, done, ozNote))
          writeGemmDescriptor(S, *CI->getFunction(), primalArgs, ozNote);
        llvm::errs() << "[poseidon] " << F->getName()
                     << ": same body and target as an earlier site; reusing "
                        "its solve ("
                     << done->getName() << ")\n";
        return emitSiteCall(CI, done, primalArgs, calls);
      }

    // Kept for no-op transparency: if the solve ends up applying NOTHING to
    // this site, the wrapper must call the ORIGINAL body, not the
    // canonicalized clone (whose lowering is not guaranteed bit-identical to
    // the original).
    Function *sourceBody = F;
    F = canonicalize(*F);
    S.SiteClones.push_back(F);
    setSlotMetadata(*F);
    noteSiteOrigin(S, F, sourceBody);
    noteSiteArgs(S, F, primalArgs);
    noteSiteCall(S, F, CI->getDebugLoc());

    // A site with nothing to solve from still has to build and run, so it keeps
    // the ORIGINAL body rather than failing the compile.
    if (std::optional<Unprofiled> site = unprofiledSite(S, *F)) {
      reportUnprofiled(S, CI, *sourceBody, *F, *site);
      markOutlineTransparent(sourceBody);
      return emitSiteCall(CI, sourceBody, primalArgs, calls);
    }

    if (O.Diagnostics.Print) {
      llvm::errs() << "[poseidon] Optimizing " << F->getName()
                   << " with relative error tolerance: "
                   << (errTol > 0.0 ? errTol : (double)O.Solver.Tau) << "\n";
    }

    if (O.Solver.JointDP) {
      // Joint mode: defer; mark F with its tolerance so the joint solve takes
      // it with the module's other sites under one shared budget.
      F->addFnAttr("poseidon-joint-errtol", exactDouble(errTol));
      // Host-dispatch note is produced only by the later joint materialize;
      // capture the body-param -> wrapper-arg mapping now (primalArgs valid
      // here).
      if (O.Matmul.OzakiHostDispatch)
        addPendingGemmDispatch(S, F, *CI->getFunction(), primalArgs);
    } else {
      bool optimized = fpOptimize(S, *F, errTol);
      if (optimized)
        noteRewritten(S, *CI->getFunction());

      if (!optimized)
        warn(CI,
             Twine("Poseidon returned false (no change) for ") + F->getName());

      // Host-side GEMM dispatch: if fpOptimize recorded this body as a raised
      // GEMM to be dispatched host-side (-poseidon-ozaki-host-dispatch), record
      // the launch descriptor (wrapper kernel + C/A/B launch-arg indices +
      // geometry) for the host sub-compilation to consume.
      {
        GemmBodyNote ozNote;
        if (O.Matmul.OzakiHostDispatch && getGemmBody(S, F, ozNote))
          writeGemmDescriptor(S, *CI->getFunction(), primalArgs, ozNote);
      }

      // No rewrite was applied: call the ORIGINAL body so the wrapped site is
      // bit-transparent (the canonicalized clone's lowering may differ at ulp
      // level).
      if (!optimized) {
        llvm::errs() << "[poseidon] no rewrite applied for " << F->getName()
                     << "; calling original body " << sourceBody->getName()
                     << " (bit-transparent no-op)\n";
        F = sourceBody;
      }
      S.SolvedSites[key] = F;
    }
  }

  // Profile-generate leaves the marker call in place for the profiler, so this
  // only ever fires on the solve path.
  if (profileUse)
    markOutlineTransparent(F);
  return emitSiteCall(CI, F, primalArgs, calls);
}

bool optimizeSite(Session &S, CallInst *CI,
                  SmallVectorImpl<CallBase *> &calls) {
  Function *F = markerTarget(CI);
  if (!F)
    return false;

  MarkerArgs marker;
  if (!parseMarker(CI, F, marker))
    return false;
  return optimizeSiteBody(S, CI, F, marker.primalArgs, marker.errTol, calls);
}

// Profile use of a host site: the body is solved once, and every call of it,
// the wrapper's and the ones the inliner placed in F's callers, is redirected
// to what the solve chose.
bool optimizeHostSite(Session &S, Function &B) {
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
  bool changed = optimizeSiteBody(S, rep, &B, primalArgs, errTol, lowered);
  Function *target = lowered.back()->getCalledFunction();
  for (CallBase *CB : calls)
    CB->setCalledFunction(target);
  return changed;
}

} // namespace

bool lowerMarkers(Session &S, ArrayRef<CallInst *> markers,
                  SmallVectorImpl<CallBase *> &calls) {
  bool changed = false;
  for (CallInst *CI : markers)
    changed |= optimizeSite(S, CI, calls);
  return changed;
}

// What the profiled run spent inside this site: over the slots the profile
// records, the execution count times the CSV price of the operation the slot
// names. A site the run never reached has no records and costs nothing, which
// is what -poseidon-kernels=all over a whole library produces for most kernels.
double profiledSiteCost(Session &S, Function &body) {
  SmallString<128> profilePath(S.Opts.Profiling.ProfileUse);
  llvm::sys::path::append(
      profilePath,
      profileNameStem(("preprocess_" + body.getName()).str()) + ".fpprofile");
  if (!llvm::sys::fs::exists(profilePath.str()))
    return 0.0;

  std::unordered_map<size_t, ProfileInfo> profileMap;
  parseProfileFile(profilePath.str().str(), profileMap);
  if (profileMap.empty())
    return 0.0;

  // The slots are numbered against the canonical form, so the prices have to be
  // read off it too. Thrown away again: the site's real clone is made by
  // optimizeSiteBody, under the same name.
  Function *clone = canonicalize(body);
  requireCostModel(S.CostModel, *clone);
  double cost = 0.0;
  size_t slot = 0;
  for (Instruction &I : instructions(*clone)) {
    if (!isOptimizable(I))
      continue;
    auto it = profileMap.find(slot++);
    if (it != profileMap.end())
      cost += (double)it->second.exec * getInstructionCompCost(S.CostModel, &I);
  }
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
    for (Function *K : sites)
      warnWithoutProfile(*K);
    return changed;
  }

  SmallVector<CallBase *, 4> calls;
  // Only a target the annotation carried is the site's own; fpOptimize falls
  // back to -poseidon-tau for the others.
  auto optimize = [&](CallInst *call, Function *body, Function *kernel) {
    SmallVector<Value *, 8> primalArgs(call->args());
    auto it = siteTau.find(kernel);
    double errTol = it != siteTau.end() ? it->second : 0.0;
    noteSiteTarget(O, *kernel,
                   it != siteTau.end() ? std::optional<double>(errTol)
                                       : std::nullopt);
    changed |= optimizeSiteBody(Sess, call, body, primalArgs, errTol, calls);
  };

  if (byRegex.empty() || O.Profiling.MinCostShare <= 0.0 ||
      !O.Profiling.usesProfile()) {
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
      llvm::errs() << "[poseidon] " << S.kernel->getName() << " is "
                   << format("%.4g", share * 100)
                   << "% of the profiled FP cost of the kernels "
                      "-poseidon-kernels named, below -poseidon-min-cost-share="
                   << format("%g", (double)O.Profiling.MinCostShare)
                   << "; left unoptimized\n";
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
  // The FP profiler's CUDA registration has to happen from main, not from a
  // static constructor (see injectHostProfilerInit). No-op on the device module
  // and in any translation unit without main.
  injectHostProfilerInit(M);
  predeclareInactiveProfilerProbes(M);

  SmallVector<CallInst *, 4> markers;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    for (Instruction &I : instructions(F))
      if (auto *CI = dyn_cast<CallInst>(&I))
        if (Function *callee = calleeOf(CI))
          if (isMarkerCall(callee->getName()))
            markers.push_back(CI);
  }
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
    injectBlockDimProbes(M);
  }
  return anySite;
}

bool solveDeferredSites(Session &S, Module &M, FunctionAnalysisManager &FAM) {
  if (!S.Opts.Solver.JointDP)
    return false;
  return solveJointly(S, M, FAM);
}

// The body a site's wrapper ended up inlining carries the function's debug
// info: the subprogram most of its code is in goes back to the wrapper once
// nothing else holds it, and the return outlineHostFunction located in the
// source body follows it there.
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
  checkPinnedSites(S);
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
