//===- Pipeline.cpp - the Poseidon passes and where they run --------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Driver.h"
#include "IR/Utils.h"
#include "Profile/Instrument.h"
#include "Profile/SiteProfile.h"
#include "Session.h"
#include "Sites/Annotations.h"
#include "Sites/CallSites.h"
#include "Sites/Canonicalize.h"
#include "Sites/LaunchDescriptors.h"
#include "Sites/LaunchStubs.h"
#include "Sites/Markers.h"
#include "Sites/Site.h"
#include "Sites/SiteIdentity.h"
#include "Solve/Optimize.h"
#include "Targets/MathLibrary.h"
#include "Targets/OffloadRuntime.h"
#include "Targets/Target.h"
#include "Twin.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
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
#include "llvm/Transforms/Utils/Cloning.h"

#include <memory>
#include <string>

using namespace llvm;

namespace poseidon {

AnalysisKey SessionAnalysis::Key;

SessionAnalysis::Result SessionAnalysis::run(Module &M,
                                             ModuleAnalysisManager &) {
  Result R;
  if (Given) {
    R.S = Given;
  } else {
    validateOptions(*Opts, M.getContext());
    R.Owned = std::make_unique<Session>(*Opts, *PB);
    R.S = R.Owned.get();
  }
  return R;
}

Session &sessionOf(Module &M, ModuleAnalysisManager &MAM) {
  return *MAM.getResult<SessionAnalysis>(M).S;
}

// Whether Enzyme has already run on M: Enzyme removes its prev_fixup marks
// before its run ends, so enzyme_math functions without any prev_fixup, or an
// augmented_/diffe clone, mean it has. Extension-point callbacks run in
// -fpass-plugin order, so a reversed plugin order shows here.
static bool enzymeAlreadyRan(Module &M) {
  bool anyMath = false, anyPending = false;
  for (const Function &F : M) {
    anyMath |= F.hasFnAttribute("enzyme_math");
    anyPending |= F.hasFnAttribute("prev_fixup");
  }
  if (anyMath && !anyPending)
    return true;
  for (const Function &F : M) {
    if (F.empty())
      continue;
    StringRef name = F.getName();
    if (M.getFunction(("augmented_" + name).str()) ||
        M.getFunction(("diffe" + name).str()))
      return true;
  }
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
    if (F.hasFnAttribute(kHostBody) || F.hasFnAttribute(kSiteCallee))
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

static void requirePluginOrder(const Options &O, Module &M) {
  if (!hasSite(O, M) || !enzymeAlreadyRan(M))
    return;
  reportUserError(M.getContext(),
                  "Enzyme's pass has already run on this module. The plugins "
                  "must be given in the order -fpass-plugin=<Poseidon>.so "
                  "-fpass-plugin=<ClangEnzyme>.so.");
}

static void runAlwaysInliner(Module &M, ModuleAnalysisManager &MAM) {
  ModulePassManager Inline;
  Inline.addPass(AlwaysInlinerPass());
  Inline.run(M, MAM);
}

namespace {

// One module pass for both phases, at OptimizerEarly so that profile
// generation hands Enzyme the derivative requests it lowers later in the same
// extension point.
struct OptimizePass : PassParent<OptimizePass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
    Session &S = sessionOf(M, MAM);
    const Options &O = S.Opts;
    requirePluginOrder(O, M);

    bool changed = lowerPerCallSites(O, S.Sites, M);
    if (O.Profiling.profileAction())
      recordMarkerIdentities(S.Sites, M);
    changed |= lowerDeviceCallSites(O, M);
    changed |= outlineHostSites(O, S.Sites, M);
    changed |= foldDeviceSiteCallees(O, M);
    changed |= prepareModule(S, M);
    SmallVector<CallBase *, 4> calls;
    if (!O.Profiling.generatesProfile())
      changed |= optimizeSites(
          S, collectSites(O, M, {Site::Form::HostBody, Site::Form::Kernel}),
          calls);

    // The markers profile generation left, and every marker in profile use.
    SmallVector<Site, 4> markers = collectSites(O, M, {Site::Form::Marker});
    erase_if(markers, [&](Site &site) { return deferSite(O, site.marker); });
    changed |= optimizeSites(S, markers, calls);

    if (O.Solver.JointDP)
      changed |= solveJointly(S, M);
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

// Why a derivative request is still unlowered where the pass expects Enzyme to
// have run. A pass run by pipeline name is not told which phase it runs in.
static std::string unloweredRequestReason(bool phaseKnown,
                                          ThinOrFullLTOPhase Phase,
                                          OptimizationLevel Level) {
  if (phaseKnown && enzymeDeferredToPostLink(Phase, Level))
    return "This is an LTO pre-link compile, where Enzyme leaves derivative "
           "requests to the post-link run unless -enzyme-lto-prelink is "
           "given; pass -mllvm -enzyme-lto-prelink=1 (for HIP as -Xclang "
           "-mllvm -Xclang -enzyme-lto-prelink=1).";
  std::string reason =
      "Add -fpass-plugin=<ClangEnzyme>.so after the Poseidon plugin.";
  if (!phaseKnown)
    reason += " In an LTO pre-link pipeline Enzyme also leaves derivative "
              "requests to the post-link run unless -enzyme-lto-prelink is "
              "given.";
  return reason;
}

namespace {

// The site handler for deferred sites, at OptimizerLast, where the derivative
// request a deferred body was made of has been lowered. A module without a
// deferred site is left untouched.
struct LatePass : PassParent<LatePass> {
  PassBuilder *PB;
  OptimizationLevel Level;
  ThinOrFullLTOPhase Phase = ThinOrFullLTOPhase::None;
  bool PhaseKnown = false;

  LatePass(PassBuilder *PB, OptimizationLevel Level) : PB(PB), Level(Level) {}
  LatePass(PassBuilder *PB, OptimizationLevel Level, ThinOrFullLTOPhase Phase)
      : PB(PB), Level(Level), Phase(Phase), PhaseKnown(true) {}

  // The canonical prefix, over the deferred bodies only: nothing follows this
  // pass, so a module-wide run would change the code of every other function.
  void canonicalizeBodies(const Options &O, Module &M,
                          ArrayRef<Function *> bodies,
                          LocalAnalysisManagers &AMs,
                          DenseMap<Function *, Function *> *asLeft) {
    for (Function *F : bodies) {
      if (asLeft)
        asLeftByPipeline(*F, *asLeft);
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
  void runEnzyme(Module &M, LocalAnalysisManagers &AMs) {
    ModulePassManager MPM;
    if (Error E = PB->parsePassPipeline(MPM, "enzyme")) {
      std::string msg = toString(std::move(E));
      reportUserError(M.getContext(),
                      Twine("a deferred site needs a second Enzyme run and the "
                            "Enzyme pass pipeline is not registered (") +
                          msg +
                          "). Add -fpass-plugin=<ClangEnzyme>.so after the "
                          "Poseidon plugin.");
    }
    MPM.run(M, AMs.MAM);
  }

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
    SmallVector<Site, 4> markers = collectDeferredSites(M);
    if (markers.empty())
      return PreservedAnalyses::all();

    Session &S = sessionOf(M, MAM);
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
        reportUserError(*B, B->getSubprogram(),
                        Twine("the marked body ") + B->getName() +
                            " still holds an __enzyme_* request that nothing "
                            "lowered. " +
                            unloweredRequestReason(PhaseKnown, Phase, Level));
    // The canonical form of a deferred site is taken here, in both phases, so
    // that the slots the profile was numbered against are the ones the solve
    // sees.
    DenseMap<Function *, Function *> asLeft;
    canonicalizeBodies(O, M, bodies, AMs,
                       O.Profiling.generatesProfile() ? nullptr : &asLeft);

    markers = collectDeferredSites(M);

    bool instrumented = false;
    if (O.Profiling.generatesProfile()) {
      DenseMap<Function *, Function *> clones;
      unsigned nextSiteId = getNextSiteId(M);
      for (const Site &site : markers)
        instrumented |= profileSite(S, site, clones, nextSiteId);
      setNextSiteId(M, nextSiteId);
      if (instrumented)
        injectBlockDimProbes(M, O.Diagnostics.Print);
      markers = collectDeferredSites(M);
    }

    SmallVector<CallBase *, 4> calls;
    optimizeSites(S, markers, calls);
    SmallVector<WeakVH, 4> lowered(calls.begin(), calls.end());
    if (O.Solver.JointDP)
      solveJointly(S, M);
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
      runEnzyme(M, AMs);
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
struct HostSitePass : PassParent<HostSitePass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
    Session &S = sessionOf(M, MAM);
    const Options &O = S.Opts;
    bool changed = false;
    if (Target::of(M).isDevice()) {
      const bool active = O.Profiling.profileAction() && hasSite(O, M);
      if (active)
        requireDeviceMathMacro(M);
      changed = active ? keepDeviceMath(M) : releaseDeviceMath(M);
    }
    changed |= lowerPerCallSites(O, S.Sites, M);
    if (O.Profiling.profileAction())
      recordMarkerIdentities(S.Sites, M);
    changed |= propagateDeviceSites(O, M);
    changed |= outlineHostSites(O, S.Sites, M);
    changed |= registerDeviceProfileTable(O, M);
    changed |= defineProfileDirectory(O, M);
    if (!O.Profiling.profileAction())
      changed |= leaveSitesAsWritten(O, S.Sites, M);
    return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
  static bool isRequired() { return true; }
};

// Clones a module with a site in a profile-use compile, at PipelineStart and
// before HostSitePass, so that every function no rewrite reaches is emitted
// from a compile as written: the canonical prefix, site bodies kept out of
// line and, on a device, libdevice math kept out of line change the code of
// every function of the module. The host half of a CUDA or HIP compilation
// holds no site, and its launch stubs are rewritten at PipelineStart, where no
// rewrite is recorded.
struct TwinSnapshotPass : PassParent<TwinSnapshotPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
    Session &S = sessionOf(M, MAM);
    const Options &O = S.Opts;
    if (S.IsTwin || isOffloadHostHalf(M) || !hasSite(O, M))
      return PreservedAnalyses::all();
    if (O.Profiling.usesProfile() && !O.Profiling.generatesProfile() &&
        !O.Solver.JointDP)
      S.AsWrittenTwin = CloneModule(M);
    return PreservedAnalyses::all();
  }
  static bool isRequired() { return true; }
};

struct FinalizePass : PassParent<FinalizePass> {
  PassBuilder *PB = nullptr;
  OptimizationLevel Level = OptimizationLevel::O3;
  ThinOrFullLTOPhase Phase = ThinOrFullLTOPhase::None;
  // Run by pipeline name, the pass is not told which phase it runs in.
  bool PhaseKnown = false;

  FinalizePass() = default;
  FinalizePass(PassBuilder *PB, OptimizationLevel Level,
               ThinOrFullLTOPhase Phase)
      : PB(PB), Level(Level), Phase(Phase), PhaseKnown(true) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
    Session &S = sessionOf(M, MAM);
    const Options &O = S.Opts;
    for (Function &F : M)
      if (F.isDeclaration() && !F.use_empty() &&
          F.getName().starts_with(kAutodiffRequest))
        reportUserError(M.getContext(),
                        Twine("profile generation emitted a derivative "
                              "request that nothing lowered. ") +
                            unloweredRequestReason(PhaseKnown, Phase, Level));
    const Target &T = Target::of(M);
    finalizeModule(S, M);
    // An annotation takes the address of the function it names, and a profiled
    // kernel may make an unknown call.
    if (O.Profiling.generatesProfile() && T.allocatesAddressTakenShared())
      dropSiteAnnotations(M);
    linkDeviceProfiler(M, O.Profiling.generatesProfile());
    if (O.Profiling.generatesProfile())
      requireSharedMemoryLimit(M);
    requireDeviceCallees(M, O.Profiling.generatesProfile());
    // Written after every site is processed and every check that can stop the
    // compile has passed, so a host half never reads the manifest of a device
    // half that failed.
    if (O.Profiling.generatesProfile())
      recordProfileTableSize(S.Launches, M);
    if (T.isDevice() && !S.IsTwin && handsOverLaunches(O))
      writeLaunchManifest(S.Opts, S.Launches, M);
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
    if (std::unique_ptr<Module> Twin = std::move(S.AsWrittenTwin))
      if (PB)
        replaceWithAsWritten(M, std::move(Twin), S, *PB, Level, Phase,
                             rewritten.getArrayRef());
    return PreservedAnalyses::none();
  }
  static bool isRequired() { return true; }
};

// A module without a site compiles as it does without the plugin, but that
// profile generation binds the profiler from main, wherever main is. The
// canonical prefix runs first in a compile that generates or uses a profile.
struct IfSitePass : PassParent<IfSitePass> {
  ModulePassManager Prefix;
  ModulePassManager Pipeline;

  IfSitePass(ModulePassManager Prefix, ModulePassManager Pipeline)
      : Prefix(std::move(Prefix)), Pipeline(std::move(Pipeline)) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
    const Options &O = sessionOf(M, MAM).Opts;
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
    MPM.addPass(HostSitePass());
    // Host-side GEMM dispatch rewrites the launch stub before inlining.
    MPM.addPass(HostStubPass());
  };
  PB.registerPipelineStartEPCallback(
      [loadHostStub](ModulePassManager &MPM, OptimizationLevel Level) {
        MPM.addPass(TwinSnapshotPass());
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
    MPM.addPass(IfSitePass(std::move(Prefix), std::move(Sites)));
  };
  PassBuilder *PBp = &PB;
  // LatePass precedes FinalizePass so that FinalizePass's check for unlowered
  // requests sees the second Enzyme run of a deferred site. The module as
  // written is compiled with the default pipeline of the phase the module is
  // in: a compile from source, also one that writes LTO bitcode. An LTO link
  // has no module as written.
  auto loadFinalize = [PBp](ModulePassManager &MPM, OptimizationLevel Level,
                            ThinOrFullLTOPhase Phase) {
    const bool fromSource = Phase == ThinOrFullLTOPhase::None ||
                            Phase == ThinOrFullLTOPhase::ThinLTOPreLink ||
                            Phase == ThinOrFullLTOPhase::FullLTOPreLink;
    MPM.addPass(LatePass(PBp, Level, Phase));
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
        if (Name == "poseidon-late") {
          MPM.addPass(LatePass(PBp, OptimizationLevel::O3));
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
