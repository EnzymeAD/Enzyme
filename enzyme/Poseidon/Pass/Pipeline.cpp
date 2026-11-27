//===- Pipeline.cpp - the Poseidon passes and where they run --------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Core/Session.h"
#include "Core/Technique.h"
#include "Enzyme/PassUtils.h"
#include "PassInternal.h"
#include "Profile/Instrument.h"
#include "Sites/Annotations.h"
#include "Sites/CallSites.h"
#include "Sites/Canonicalize.h"
#include "Sites/LaunchManifest.h"
#include "Sites/Markers.h"
#include "Sites/Site.h"
#include "Sites/SiteIdentity.h"
#include "Solve/Solve.h"
#include "Support/Diagnostics.h"
#include "Support/Utils.h"
#include "Targets/MathLibrary.h"
#include "Targets/OffloadRuntime.h"
#include "Targets/Target.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Passes/PassBuilder.h"
#if LLVM_VERSION_MAJOR >= 22 && !defined(ENZYME_LLVM22_PRERELEASE)
#include "llvm/Plugins/PassPlugin.h"
#else
#include "llvm/Passes/PassPlugin.h"
#endif
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/Utils/Local.h"

#include <memory>
#include <string>

using namespace llvm;

namespace poseidon {

// Erases what a stopped compile leaves no pass to lower: the markers and the
// derivative requests Poseidon emits, whose metadata operands and parameters
// the verifier accepts only until they are lowered. A call's result reads as
// poison.
static void eraseUnloweredCalls(Module &M) {
  SmallVector<Function *, 4> callees;
  for (Function &F : M)
    if (F.isDeclaration() && (F.getName().starts_with(kAutodiffRequest) ||
                              isMarkerCall(F.getName())))
      callees.push_back(&F);
  for (Function *F : callees) {
    SmallVector<CallBase *, 8> calls;
    for (User *U : F->users())
      if (auto *CB = dyn_cast<CallBase>(U))
        calls.push_back(CB);
    for (CallBase *CB : calls) {
      if (auto *II = dyn_cast<InvokeInst>(CB))
        CB = changeToCall(II);
      CB->replaceAllUsesWith(PoisonValue::get(CB->getType()));
      CB->eraseFromParent();
    }
    if (F->use_empty())
      F->eraseFromParent();
  }
}

PreservedAnalyses stopOnError(Session &S, Module &M, Error E) {
  diagnoseUserErrors(M.getContext(), std::move(E));
  eraseUnloweredCalls(M);
  S.Stopped = true;
  return PreservedAnalyses::none();
}

AnalysisKey SessionAnalysis::Key;

SessionAnalysis::Result SessionAnalysis::run(Module &M,
                                             ModuleAnalysisManager &) {
  Result R;
  if (Given) {
    R.S = Given;
  } else {
    R.Owned = std::make_unique<Session>(*Opts, *PB);
    R.S = R.Owned.get();
    if (Error E = validateOptions(*Opts))
      stopOnError(*R.S, M, std::move(E));
  }
  return R;
}

Session &sessionOf(Module &M, ModuleAnalysisManager &MAM) {
  return *MAM.getResult<SessionAnalysis>(M).S;
}

// Whether Enzyme has already run on M: Enzyme removes its prev_fixup marks
// before its run ends, so enzyme_math functions without any prev_fixup, or a
// function Enzyme generated from one M defines, mean it has. Extension-point
// callbacks run in -fpass-plugin order, so a reversed plugin order shows here.
static bool enzymeAlreadyRan(Module &M) {
  bool anyMath = false, anyPending = false;
  for (const Function &F : M) {
    anyMath |= F.hasFnAttribute("enzyme_math");
    anyPending |= F.hasFnAttribute("prev_fixup");
  }
  if (anyMath && !anyPending)
    return true;
  for (const Function &G : M)
    if (std::optional<StringRef> from = enzymeGeneratedFrom(G.getName()))
      if (const Function *F = M.getFunction(*from); F && !F->empty())
        return true;
  return false;
}

// Whether M holds a site. The host half of a CUDA or HIP compilation holds no
// site of the annotation or the call form, whatever annotations and calls it
// carries: the annotation of a kernel is its device half's.
static bool hasSite(const Options &O, Module &M) {
  const bool hostHalf = isOffloadHostHalf(M);
  if (!hostHalf &&
      (!O.Profiling.Kernels.empty() || !annotatedFunctions(M).empty()))
    return true;
  for (Function &F : M) {
    if (F.hasFnAttribute(kSiteBody) || F.hasFnAttribute(kSiteCallee))
      return true;
    if (!F.use_empty() &&
        ((F.isDeclaration() && isMarkerCall(F.getName())) ||
         (!hostHalf && is_contained({kPlainIdentity, kPlainIdentityTau,
                                     kExplicitIdentity, kExplicitIdentityTau},
                                    F.getName()))))
      return true;
  }
  return false;
}

static Error requirePluginOrder(const Options &O, Module &M) {
  if (!hasSite(O, M) || !enzymeAlreadyRan(M))
    return Error::success();
  return userError("Enzyme's pass ran on this module before Poseidon's; give "
                   "the plugins in the order -fpass-plugin=<Poseidon>.so "
                   "-fpass-plugin=<ClangEnzyme>.so");
}

static void runAlwaysInliner(Module &M, ModuleAnalysisManager &MAM) {
  ModulePassManager Inline;
  Inline.addPass(AlwaysInlinerPass());
  Inline.run(M, MAM);
}

// Checks M's site annotations and lowers its per-call sites, then, for a
// profile action, records the markers' identities; true when M changed.
static Expected<bool> lowerAnnotatedSites(Session &S, Module &M) {
  Error errors = checkSiteAnnotations(M);
  Expected<bool> perCall = lowerPerCallSites(S.Opts, S.get<SiteRecords>(), M);
  if (!perCall || errors)
    return joinErrors(std::move(errors), perCall.takeError());
  if (S.Opts.Profiling.profileAction())
    recordMarkerIdentities(S.get<SiteRecords>(), M);
  return *perCall;
}

namespace {

// One module pass for both phases, at OptimizerEarly so that profile
// generation hands Enzyme the derivative requests it lowers later in the same
// extension point.
struct OptimizePass : PassParent<OptimizePass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
    Session &S = sessionOf(M, MAM);
    if (S.Stopped)
      return PreservedAnalyses::all();
    const Options &O = S.Opts;
    if (Error E = requirePluginOrder(O, M))
      return stopOnError(S, M, std::move(E));

    Expected<bool> perCall = lowerAnnotatedSites(S, M);
    if (!perCall)
      return stopOnError(S, M, perCall.takeError());
    bool changed = *perCall;
    Expected<bool> deviceCalls = lowerDeviceCallSites(O, M);
    if (!deviceCalls)
      return stopOnError(S, M, deviceCalls.takeError());
    changed |= *deviceCalls;
    Expected<bool> hostSites = outlineHostSites(O, S.get<SiteRecords>(), M);
    if (!hostSites)
      return stopOnError(S, M, hostSites.takeError());
    changed |= *hostSites;
    changed |= foldDeviceSiteCallees(O, M);
    Expected<bool> prepared = prepareModule(S, M);
    if (!prepared)
      return stopOnError(S, M, prepared.takeError());
    changed |= *prepared;
    SmallVector<CallBase *, 4> calls;
    if (!O.Profiling.generatesProfile()) {
      Expected<bool> optimized = optimizeSites(
          S, collectSites(O, M, {Site::Form::HostBody, Site::Form::Kernel}),
          calls);
      if (!optimized)
        return stopOnError(S, M, optimized.takeError());
      changed |= *optimized;
    }

    // The markers profile generation left, and every marker in profile use.
    SmallVector<Site, 4> markers = collectSites(O, M, {Site::Form::Marker});
    erase_if(markers, [&](Site &site) { return deferSite(O, site.marker); });
    Expected<bool> optimized = optimizeSites(S, markers, calls);
    if (!optimized)
      return stopOnError(S, M, optimized.takeError());
    changed |= *optimized;

    if (O.Solver.SolveJointly) {
      Expected<bool> joint = solveJointly(S, M);
      if (!joint)
        return stopOnError(S, M, joint.takeError());
      changed |= *joint;
    }
    return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
  static bool isRequired() { return true; }
};

} // namespace

// Whether Enzyme left the module's derivative requests to the LTO post-link
// run: in these phases it does unless -enzyme-lto-prelink is given, and an
// Enzyme without that option never does.
static bool enzymeDeferredToPostLink(ThinOrFullLTOPhase Phase,
                                     OptimizationLevel Level) {
  if (Phase != ThinOrFullLTOPhase::FullLTOPreLink &&
      (Phase != ThinOrFullLTOPhase::ThinLTOPreLink ||
       Level == OptimizationLevel::O0))
    return false;
  auto &Opts = cl::getRegisteredOptions();
  auto It = Opts.find("enzyme-lto-prelink");
  return It != Opts.end() &&
         !static_cast<cl::opt<bool> *>(It->second)->getValue();
}

// The end of the error about a derivative request still unlowered where the
// pass expects Enzyme to have run: why, when the phase says, and what to pass.
// A pass run by pipeline name is not told which phase it runs in.
static std::string
unloweredRequestReason(std::optional<ThinOrFullLTOPhase> Phase,
                       OptimizationLevel Level) {
  if (!Phase)
    return "; add -fpass-plugin=<ClangEnzyme>.so after the Poseidon plugin, "
           "and in an LTO pre-link compile -mllvm -enzyme-lto-prelink=1";
  if (enzymeDeferredToPostLink(*Phase, Level))
    return ", which Enzyme leaves to the post-link run of an LTO pre-link "
           "compile; pass -mllvm -enzyme-lto-prelink=1 (for HIP, -Xclang "
           "-mllvm -Xclang -enzyme-lto-prelink=1)";
  return "; add -fpass-plugin=<ClangEnzyme>.so after the Poseidon plugin";
}

namespace {

// The site handler for deferred sites, at OptimizerLast, where the derivative
// request a deferred body was made of has been lowered. A module without a
// deferred site is left untouched.
struct DeferredSitePass : PassParent<DeferredSitePass> {
  PassBuilder *PB;
  OptimizationLevel Level;
  // None when run by pipeline name, which does not say the phase.
  std::optional<ThinOrFullLTOPhase> Phase;

  DeferredSitePass(PassBuilder *PB, OptimizationLevel Level,
                   std::optional<ThinOrFullLTOPhase> Phase = std::nullopt)
      : PB(PB), Level(Level), Phase(Phase) {}

  // The canonical prefix, over the deferred bodies only: nothing follows this
  // pass, so a module-wide run would change the code of every other function.
  void canonicalizeBodies(const Options &O, Module &M,
                          ArrayRef<Function *> bodies,
                          LocalAnalysisManagers &AMs,
                          DenseMap<Function *, Function *> *asLeft) {
    for (Function *F : bodies) {
      if (asLeft)
        asWrittenCopy(*F, *asLeft);
      unsigned folded = foldEnzymeGenerated(*F);
      if (O.Diagnostics.Print && folded)
        llvm::errs() << "[poseidon] " << F->getName() << ": folded " << folded
                     << " Enzyme-generated call(s) into the deferred body\n";
    }
    FunctionPassManager Pre;
    addCanonicalPreInline(Pre, Level);
    for (Function *F : bodies)
      Pre.run(*F, AMs.FAM);

    runAlwaysInliner(M, AMs.MAM);

    FunctionPassManager Post;
    addCanonicalPostInline(Post);
    for (Function *F : bodies)
      Post.run(*F, AMs.FAM);
  }

  // Re-runs the function simplification pipeline over the kernels that carried
  // a marker: nothing later simplifies what this pass produced, and the marker
  // held the site's argument storage in memory for the whole pipeline.
  void resimplify(ArrayRef<Function *> kernels, LocalAnalysisManagers &AMs) {
    if (Level == OptimizationLevel::O0 || kernels.empty())
      return;
    FunctionPassManager FPM = PB->buildFunctionSimplificationPipeline(
        Level, ThinOrFullLTOPhase::None);
    for (Function *F : kernels)
      FPM.run(*F, AMs.FAM);
  }

  // Runs Enzyme once more, by its registered pipeline name, to lower the
  // reverse-mode request profile generation emits for a deferred clone; nothing
  // later in the pipeline would.
  Error runEnzyme(Module &M, LocalAnalysisManagers &AMs) {
    ModulePassManager MPM;
    if (Error E = PB->parsePassPipeline(MPM, "enzyme"))
      return userError(Twine("a deferred site needs a second Enzyme run, and "
                             "the Enzyme pass pipeline is not registered (") +
                       toString(std::move(E)) +
                       "); add -fpass-plugin=<ClangEnzyme>.so after the "
                       "Poseidon plugin");
    MPM.run(M, AMs.MAM);
    return Error::success();
  }

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
    SmallVector<Site, 4> markers = collectDeferredSites(M);
    if (markers.empty())
      return PreservedAnalyses::all();

    Session &S = sessionOf(M, MAM);
    if (S.Stopped)
      return PreservedAnalyses::all();
    const Options &O = S.Opts;
    LocalAnalysisManagers AMs(*PB, S);

    SmallVector<Function *, 4> bodies;
    SmallPtrSet<Function *, 4> bodySet;
    SmallVector<WeakVH, 4> kernels;
    SmallPtrSet<Function *, 4> kernelSet;
    for (const Site &site : markers) {
      Function *B = site.fn;
      if (!B->empty() && bodySet.insert(B).second)
        bodies.push_back(B);
      if (kernelSet.insert(site.marker->getFunction()).second)
        kernels.push_back(site.marker->getFunction());
    }
    // Unlowered here, the request would surface only at the link, as an
    // undefined __enzyme_* symbol.
    for (Function *B : bodies)
      if (callsEnzymeRequest(*B))
        return stopOnError(
            S, M,
            userError(*B, Twine("the marked body ") + B->getName() +
                              " still holds an __enzyme_* request that nothing "
                              "lowered" +
                              unloweredRequestReason(Phase, Level)));
    // The canonical form of a deferred site is taken here, in both phases, so
    // that the slots the profile was numbered against are the ones the solve
    // sees.
    DenseMap<Function *, Function *> asLeft;
    canonicalizeBodies(O, M, bodies, AMs,
                       O.Profiling.generatesProfile() ? nullptr : &asLeft);

    markers = collectDeferredSites(M);

    bool instrumented = false;
    if (O.Profiling.generatesProfile()) {
      Expected<bool> profiled = profileSites(S, M, markers);
      if (!profiled)
        return stopOnError(S, M, profiled.takeError());
      instrumented = *profiled;
      markers = collectDeferredSites(M);
    }

    SmallVector<CallBase *, 4> calls;
    if (Error E = optimizeSites(S, markers, calls).takeError())
      return stopOnError(S, M, std::move(E));
    SmallVector<WeakVH, 4> lowered(calls.begin(), calls.end());
    if (O.Solver.SolveJointly)
      if (Error E = solveJointly(S, M).takeError())
        return stopOnError(S, M, std::move(E));
    // A site that kept its body calls the body as the pipeline left it rather
    // than its canonical form.
    for (WeakVH &handle : lowered)
      if (auto *CB = dyn_cast_or_null<CallBase>(handle))
        if (Function *copy = asLeft.lookup(CB->getCalledFunction())) {
          markOutlineTransparent(copy);
          CB->setCalledFunction(copy);
        }
    for (bool erased = true; erased;) {
      erased = false;
      for (auto &[original, copy] : asLeft)
        if (copy && copy->use_empty()) {
          copy->eraseFromParent();
          copy = nullptr;
          erased = true;
        }
    }

    runAlwaysInliner(M, AMs.MAM);

    if (instrumented) {
      if (Error E = runEnzyme(M, AMs))
        return stopOnError(S, M, std::move(E));
      runAlwaysInliner(M, AMs.MAM);
    }

    SmallVector<Function *, 4> live;
    for (WeakVH &handle : kernels)
      if (auto *F = dyn_cast_or_null<Function>(handle))
        if (!F->isDeclaration() && F->getParent() == &M)
          live.push_back(F);
    resimplify(live, AMs);

    return PreservedAnalyses::none();
  }
  static bool isRequired() { return true; }
};

// POSEIDON_OPTIMIZE on a host function is lowered at PipelineStart, before the
// inliner can fold the function into its callers, and on a device function it
// is handed to the kernels that call it while those calls still exist.
struct LowerSitesPass : PassParent<LowerSitesPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
    Session &S = sessionOf(M, MAM);
    if (S.Stopped)
      return PreservedAnalyses::all();
    const Options &O = S.Opts;
    bool changed = false;
    if (Target::of(M).isDevice()) {
      const bool active = O.Profiling.profileAction() && hasSite(O, M);
      if (active)
        if (Error E = requireDeviceMathMacro(M))
          return stopOnError(S, M, std::move(E));
      changed = active ? keepDeviceMath(M) : releaseDeviceMath(M);
    }
    Expected<bool> perCall = lowerAnnotatedSites(S, M);
    if (!perCall)
      return stopOnError(S, M, perCall.takeError());
    changed |= *perCall;
    Expected<bool> deviceSites = propagateDeviceSites(O, M);
    if (!deviceSites)
      return stopOnError(S, M, deviceSites.takeError());
    changed |= *deviceSites;
    Expected<bool> hostSites = outlineHostSites(O, S.get<SiteRecords>(), M);
    if (!hostSites)
      return stopOnError(S, M, hostSites.takeError());
    changed |= *hostSites;
    Expected<bool> table = registerDeviceProfileTable(O, M);
    if (!table)
      return stopOnError(S, M, table.takeError());
    changed |= *table;
    changed |= defineProfileDirectory(O, M);
    if (!O.Profiling.profileAction())
      changed |= leaveSitesAsWritten(O, S.get<SiteRecords>(), M);
    return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
  static bool isRequired() { return true; }
};

// Clones a module with a site in a profile-use compile, at PipelineStart and
// before LowerSitesPass, so that every function no rewrite reaches is emitted
// from a compile as written: the canonical prefix, site bodies kept out of
// line and, on a device, libdevice math kept out of line change the code of
// every function of the module. The host half of a CUDA or HIP compilation
// holds no site, and its launch stubs are rewritten at PipelineStart, where no
// rewrite is recorded.
struct AsWrittenSnapshotPass : PassParent<AsWrittenSnapshotPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
    Session &S = sessionOf(M, MAM);
    const Options &O = S.Opts;
    if (S.Stopped || S.IsAsWritten || isOffloadHostHalf(M) || !hasSite(O, M))
      return PreservedAnalyses::all();
    if (O.Profiling.usesProfile() && !O.Profiling.generatesProfile() &&
        !O.Solver.SolveJointly)
      S.AsWritten = snapshotAsWritten(M);
    return PreservedAnalyses::all();
  }
  static bool isRequired() { return true; }
};

struct FinalizePass : PassParent<FinalizePass> {
  PassBuilder *PB = nullptr;
  OptimizationLevel Level = OptimizationLevel::O3;
  // None when run by pipeline name, which does not say the phase.
  std::optional<ThinOrFullLTOPhase> Phase;

  FinalizePass() = default;
  FinalizePass(PassBuilder *PB, OptimizationLevel Level,
               ThinOrFullLTOPhase Phase)
      : PB(PB), Level(Level), Phase(Phase) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
    Session &S = sessionOf(M, MAM);
    if (S.Stopped)
      return PreservedAnalyses::all();
    const Options &O = S.Opts;
    for (Function &F : M)
      if (F.isDeclaration() && !F.use_empty() &&
          F.getName().starts_with(kAutodiffRequest))
        return stopOnError(
            S, M,
            userError(Twine("profile generation emitted a derivative request "
                            "that nothing lowered") +
                      unloweredRequestReason(Phase, Level)));
    const Target &T = Target::of(M);
    if (Error E = finalizeModule(S, M))
      return stopOnError(S, M, std::move(E));
    // An annotation takes the address of the function it names, and a profiled
    // kernel may make an unknown call.
    if (O.Profiling.generatesProfile() && T.allocatesAddressTakenShared())
      dropSiteAnnotations(M);
    if (Error E =
            linkDeviceProfiler(M, O.Profiling.generatesProfile()).takeError())
      return stopOnError(S, M, std::move(E));
    if (Error E = requireDeviceCallees(M, O.Profiling.generatesProfile()))
      return stopOnError(S, M, std::move(E));
    // Written after every site is processed and every check that can stop the
    // compile has passed, so a host half never reads the manifest of a device
    // half that failed.
    if (O.Profiling.generatesProfile())
      recordProfileTableSize(S.get<LaunchManifest>(), M);
    if (T.isDevice() && !S.IsAsWritten && handsOverLaunches(O))
      if (Error E = writeLaunchManifest(S.Opts, S.get<LaunchManifest>(), M))
        return stopOnError(S, M, std::move(E));
    releaseDeviceMath(M);
    SmallSetVector<Function *, 4> rewritten;
    for (WeakVH &handle : S.RewrittenFunctions)
      if (auto *F = dyn_cast_or_null<Function>(handle);
          F && F->getParent() == &M && !F->isDeclaration())
        rewritten.insert(F);
    erase_if(S.RewrittenFunctions, [&](WeakVH &handle) {
      auto *F = dyn_cast_or_null<Function>(handle);
      return !F || F->getParent() == &M;
    });
    for (Function *F : rewritten)
      inlineDeviceMath(*F);
    if (std::unique_ptr<Module> Copy = std::move(S.AsWritten))
      if (PB)
        replaceWithAsWritten(M, std::move(Copy), S, *PB, Level, *Phase,
                             rewritten.getArrayRef());
    return PreservedAnalyses::none();
  }
  static bool isRequired() { return true; }
};

// A module without a site compiles as it does without the plugin, but that
// profile generation binds the profiler from main, wherever main is. The
// canonical prefix runs first in a compile that generates or uses a profile.
struct SiteGatePass : PassParent<SiteGatePass> {
  ModulePassManager Prefix;
  ModulePassManager Pipeline;

  SiteGatePass(ModulePassManager Prefix, ModulePassManager Pipeline)
      : Prefix(std::move(Prefix)), Pipeline(std::move(Pipeline)) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
    Session &S = sessionOf(M, MAM);
    if (S.Stopped)
      return PreservedAnalyses::all();
    const Options &O = S.Opts;
    if (!hasSite(O, M))
      return O.Profiling.generatesProfile() &&
                     injectHostProfilerInit(M, O.Diagnostics.Print)
                 ? PreservedAnalyses::none()
                 : PreservedAnalyses::all();
    PreservedAnalyses PA = PreservedAnalyses::all();
    if (O.Profiling.profileAction())
      PA.intersect(Prefix.run(M, MAM));
    PA.intersect(Pipeline.run(M, MAM));
    return PA;
  }
  static bool isRequired() { return true; }
};

// Rewrites, on the host module and before inlining, each __device_stub__
// launch stub that a descriptor in the module's manifest matches: for profile
// generation into a launch that supplies the profiling kernel's shadows, and
// for host library call. Nothing in a device module or a compile that hands no
// launch over.
struct HostStubPass : PassParent<HostStubPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
    Session &S = sessionOf(M, MAM);
    const Options &O = S.Opts;
    if (S.Stopped || Target::of(M).isDevice() || !handsOverLaunches(O))
      return PreservedAnalyses::all();
    // Library call materializes only in the optimize phase: profile-gen builds
    // instrument the original kernel and do not link the runtime.
    bool changed = false;
    if (O.Profiling.generatesProfile()) {
      Expected<bool> rewritten = rewriteProfileStubBodies(O, M);
      if (!rewritten)
        return stopOnError(S, M, rewritten.takeError());
      changed = *rewritten;
      Expected<bool> registered = registerProfiledKernels(O, M);
      if (!registered)
        return stopOnError(S, M, registered.takeError());
      changed |= *registered;
    } else {
      for (const Technique *T : techniques()) {
        Expected<bool> rewritten = T->rewriteHost(O, M);
        if (!rewritten)
          return stopOnError(S, M, rewritten.takeError());
        changed |= *rewritten;
      }
    }
    return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

} // namespace

// The canonical prefix, which brings a site body to the form the profile is
// numbered against. It matches the start of Enzyme's OptimizerEarly pipeline,
// which runs after this plugin's callback.
static void addCanonicalPrefix(ModulePassManager &MPM,
                               OptimizationLevel Level) {
  if (Level != OptimizationLevel::O0) {
    FunctionPassManager OptimizePM;
    addCanonicalPreInline(OptimizePM, Level);
    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(OptimizePM)));
  }
  MPM.addPass(AlwaysInlinerPass());
  FunctionPassManager FPM;
  addCanonicalPostInline(FPM);
  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
}

static void registerPasses(PassBuilder &PB) {
  // Every pass reads the options from the session of the module it runs on, so
  // the module compiled as written, whose session carries its own, runs the
  // same pipeline.
  PB.registerAnalysisRegistrationCallback(
      [O = readOptions(), &PB](ModuleAnalysisManager &MAM) {
        MAM.registerPass([&] { return SessionAnalysis(O, PB); });
      });

  auto loadHostStub = [](ModulePassManager &MPM, OptimizationLevel) {
    MPM.addPass(LowerSitesPass());
    // Host-side library call rewrites the launch stub before inlining.
    MPM.addPass(HostStubPass());
  };
  PB.registerPipelineStartEPCallback(
      [loadHostStub](ModulePassManager &MPM, OptimizationLevel Level) {
        MPM.addPass(AsWrittenSnapshotPass());
        loadHostStub(MPM, Level);
      });

  auto loadPoseidon = [](ModulePassManager &MPM, OptimizationLevel Level) {
    ModulePassManager Prefix;
    addCanonicalPrefix(Prefix, Level);
    ModulePassManager Sites;
    Sites.addPass(OptimizePass());
    // Nothing later in the pipeline folds the outlined site bodies and the
    // materialized clones back into their kernels.
    Sites.addPass(AlwaysInlinerPass());
    MPM.addPass(SiteGatePass(std::move(Prefix), std::move(Sites)));
  };
  PassBuilder *PBp = &PB;
  // DeferredSitePass precedes FinalizePass so that FinalizePass's check for
  // unlowered requests sees the second Enzyme run of a deferred site. The
  // module as written is compiled with the default pipeline of the phase the
  // module is in: a compile from source, also one that writes LTO bitcode. An
  // LTO link has no module as written.
  auto loadFinalize = [PBp](ModulePassManager &MPM, OptimizationLevel Level,
                            ThinOrFullLTOPhase Phase) {
    const bool fromSource = Phase == ThinOrFullLTOPhase::None ||
                            Phase == ThinOrFullLTOPhase::ThinLTOPreLink ||
                            Phase == ThinOrFullLTOPhase::FullLTOPreLink;
    MPM.addPass(DeferredSitePass(PBp, Level, Phase));
    MPM.addPass(FinalizePass(fromSource ? PBp : nullptr, Level, Phase));
  };

  PB.registerOptimizerEarlyEPCallback(
      [loadPoseidon](ModulePassManager &MPM, OptimizationLevel Level,
                     ThinOrFullLTOPhase) { loadPoseidon(MPM, Level); });
  PB.registerOptimizerLastEPCallback(loadFinalize);
  PB.registerFullLinkTimeOptimizationEarlyEPCallback(
      [loadHostStub, loadPoseidon](ModulePassManager &MPM,
                                   OptimizationLevel Level) {
        loadHostStub(MPM, Level);
        loadPoseidon(MPM, Level);
      });
  PB.registerFullLinkTimeOptimizationLastEPCallback(
      [loadFinalize](ModulePassManager &MPM, OptimizationLevel Level) {
        loadFinalize(MPM, Level, ThinOrFullLTOPhase::FullLTOPostLink);
      });

  PB.registerPipelineParsingCallback(
      [PBp](StringRef Name, ModulePassManager &MPM,
            ArrayRef<PassBuilder::PipelineElement>) {
        if (Name == "poseidon") {
          MPM.addPass(OptimizePass());
          return true;
        }
        if (Name == "poseidon-finalize") {
          MPM.addPass(FinalizePass());
          return true;
        }
        if (Name == "poseidon-deferred-sites") {
          MPM.addPass(DeferredSitePass(PBp, OptimizationLevel::O3));
          return true;
        }
        return false;
      });
}

} // namespace poseidon

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "Poseidon", "v0.1",
          [](llvm::PassBuilder &PB) { poseidon::registerPasses(PB); }};
}
