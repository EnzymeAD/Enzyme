#include "Pipeline.h"

#include "Driver.h"
#include "Flags.h"
#include "Profile/DeviceProfile.h"
#include "Profile/ProfileRead.h"
#include "Profile/SiteProfile.h"
#include "Session.h"
#include "Sites/Annotations.h"
#include "Sites/Canonicalize.h"
#include "Sites/HostSites.h"
#include "Sites/KernelSites.h"
#include "Sites/LaunchDescriptors.h"
#include "Sites/Markers.h"
#include "Targets/DeviceMath.h"
#include "Targets/OffloadRuntime.h"
#include "Targets/Target.h"
#include "Techniques/Matmul/HostDispatch.h"
#include "Twin.h"
#include "Utils.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <memory>
#include <string>

using namespace llvm;

namespace poseidon {

namespace {

// Enzyme's PreserveNVVM marks every function it must keep with prev_fixup at
// the start of its OptimizerEarly pipeline and removes the mark at the end, so
// a module that carries the enzyme_math side of that pass without the mark has
// already been through Enzyme. LLVM runs extension-point callbacks in
// -fpass-plugin order, which makes that the wrong order.
bool enzymeAlreadyRan(Module &M) {
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

// The host half of a CUDA or HIP compilation holds no site of the annotation
// or the call form, whatever annotations and calls it carries
// (lowerPerCallSites); the annotation of a kernel is its device half's.
bool hasSite(const Options &O, Module &M) {
  const bool hostHalf = isOffloadHostHalf(M);
  if (!hostHalf &&
      (!O.Profiling.Kernels.empty() || !annotatedFunctions(M).empty()))
    return true;
  for (Function &F : M) {
    if (F.hasFnAttribute(kHostBody) || F.hasFnAttribute(kSiteCallee))
      return true;
    if (!F.use_empty() &&
        ((F.isDeclaration() && isMarkerCall(F.getName())) ||
         (!hostHalf && (F.getName() == "__poseidon_site" ||
                        F.getName() == "__poseidon_site_tau"))))
      return true;
  }
  return false;
}

void requirePluginOrder(const Options &O, Module &M) {
  if (!hasSite(O, M) || !enzymeAlreadyRan(M))
    return;
  reportUserError(M.getContext(),
                  "Enzyme's pass has already run on this module. The plugins "
                  "must be given in the order -fpass-plugin=<Poseidon>.so "
                  "-fpass-plugin=<ClangEnzyme>.so.");
}

// One module pass for both phases, at OptimizerEarly so that profile
// generation hands Enzyme the derivative requests it lowers later in the same
// extension point.
struct OptimizePass : PassParent<OptimizePass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
    Session &S = sessionOf(M, MAM);
    const Options &O = S.Opts;
    requirePluginOrder(O, M);

    bool changed = lowerPerCallSites(O, M);
    changed |= lowerDeviceCallSites(M);
    changed |= outlineHostSites(O, M);
    changed |= foldDeviceSiteCallees(M);
    changed |= prepareModule(S, M);
    changed |= optimizeAnnotatedSites(S, M);

    SmallVector<CallInst *, 4> markers;
    SmallVector<CallBase *, 4> calls;
    for (Function &F : M) {
      if (F.empty())
        continue;
      markers.clear();
      for (BasicBlock &BB : F)
        for (Instruction &I : BB)
          if (auto *CI = dyn_cast<CallInst>(&I))
            if (Function *callee = calleeOf(CI))
              if (isMarkerCall(callee->getName()) && !deferSite(O, CI))
                markers.push_back(CI);
      if (!markers.empty())
        changed |= lowerMarkers(S, markers, calls);
    }

    auto &FAM =
        MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
    changed |= solveDeferredSites(S, M, FAM);
    return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
  static bool isRequired() { return true; }
};

// Whether Enzyme left the module's derivative requests to the LTO post-link
// run: in these phases it does unless -enzyme-lto-prelink is given
// (augmentPassBuilder in Enzyme.cpp), and an Enzyme without that option never
// does.
bool enzymeDeferredToPostLink(ThinOrFullLTOPhase Phase,
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
std::string unloweredRequestReason(bool phaseKnown, ThinOrFullLTOPhase Phase,
                                   OptimizationLevel Level) {
  if (phaseKnown && enzymeDeferredToPostLink(Phase, Level))
    return "This is an LTO pre-link compile, where Enzyme leaves derivative "
           "requests to the post-link run unless -enzyme-lto-prelink is "
           "given; pass -mllvm -enzyme-lto-prelink=1 (for HIP through "
           "-Xclang, as poseidon-clang does).";
  std::string reason =
      "Add -fpass-plugin=<ClangEnzyme>.so after the Poseidon plugin.";
  if (!phaseKnown)
    reason += " In an LTO pre-link pipeline Enzyme also leaves derivative "
              "requests to the post-link run unless -enzyme-lto-prelink is "
              "given.";
  return reason;
}

// The late run of the site handler, at OptimizerLast: by now Enzyme has
// lowered the derivative request the deferred body was made of, so the body
// carries the derivative arithmetic and is profiled and solved like any other
// site. A module with no deferred site leaves here without touching anything,
// so its output is byte for byte what it is without this pass.
struct LatePass : PassParent<LatePass> {
  PassBuilder *PB;
  OptimizationLevel Level;
  ThinOrFullLTOPhase Phase = ThinOrFullLTOPhase::None;
  bool PhaseKnown = false;

  LatePass(PassBuilder *PB, OptimizationLevel Level) : PB(PB), Level(Level) {}
  LatePass(PassBuilder *PB, OptimizationLevel Level, ThinOrFullLTOPhase Phase)
      : PB(PB), Level(Level), Phase(Phase), PhaseKnown(true) {}

  // The canonical prefix, applied to the deferred bodies only. At
  // OptimizerEarly the same passes run over the whole module because the rest
  // of the pipeline follows them; here nothing follows, so a module-wide run
  // would be the only thing standing between a deferred-site build and the
  // codegen it would otherwise have had.
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

    ModulePassManager Inline;
    Inline.addPass(AlwaysInlinerPass());
    Inline.run(M, AMs.MAM);

    FunctionPassManager Post;
    addCanonicalPostInline(Post);
    for (Function *F : bodies)
      Post.run(*F, AMs.FAM);
  }

  // Nothing after this pass simplifies what it produced, and the marker held
  // the site's argument storage in memory for the whole pipeline, so the
  // kernels that carried one are put back through the function simplification
  // pipeline. Only those kernels: the rest of the module keeps the code it
  // already had.
  void resimplify(ArrayRef<Function *> kernels, LocalAnalysisManagers &AMs) {
    if (Level == OptimizationLevel::O0 || kernels.empty())
      return;
    FunctionPassManager FPM = PB->buildFunctionSimplificationPipeline(
        Level, ThinOrFullLTOPhase::None);
    for (Function *F : kernels)
      FPM.run(*F, AMs.FAM);
  }

  // Profile generation instruments the deferred clone and emits the
  // reverse-mode request the probes record from; nothing later in the pipeline
  // would lower it, so Enzyme is run once more, from its own registered
  // pipeline name.
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
    SmallVector<CallInst *, 4> markers;
    collectDeferredMarkers(M, markers);
    if (markers.empty())
      return PreservedAnalyses::all();

    Session &S = sessionOf(M, MAM);
    const Options &O = S.Opts;
    LocalAnalysisManagers AMs(*PB, S);

    SmallVector<Function *, 4> bodies;
    SmallPtrSet<Function *, 4> bodySet;
    SmallVector<WeakVH, 4> kernels;
    SmallPtrSet<Function *, 4> kernelSet;
    for (CallInst *CI : markers) {
      Function *B = markerTargetOrNull(CI);
      if (B && !B->empty() && bodySet.insert(B).second)
        bodies.push_back(B);
      if (kernelSet.insert(CI->getFunction()).second)
        kernels.push_back(CI->getFunction());
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
                       O.Profiling.ProfileGenerate ? nullptr : &asLeft);

    markers.clear();
    collectDeferredMarkers(M, markers);

    bool instrumented = false;
    if (O.Profiling.ProfileGenerate) {
      DenseMap<Function *, Function *> clones;
      unsigned nextSiteId = getNextSiteId(M);
      for (CallInst *CI : markers)
        instrumented |= profileSite(S, CI, clones, nextSiteId);
      setNextSiteId(M, nextSiteId);
      if (instrumented)
        injectBlockDimProbes(M);
      markers.clear();
      collectDeferredMarkers(M, markers);
    }

    // Whatever is left is lowered exactly as the early run lowers a site: the
    // solve in profile-use mode, a call of the original body otherwise.
    SmallVector<CallBase *, 4> calls;
    if (!markers.empty())
      lowerMarkers(S, markers, calls);
    SmallVector<WeakVH, 4> lowered(calls.begin(), calls.end());
    solveDeferredSites(S, M, AMs.FAM);
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
    // The body a site ends up calling is made always-inline by
    // optimizeSiteBody, which both runs share, so the AlwaysInlinerPass below
    // folds a refused site back exactly as it folds a rewritten one.

    ModulePassManager Inline;
    Inline.addPass(AlwaysInlinerPass());
    Inline.run(M, AMs.MAM);

    if (instrumented) {
      runEnzyme(M, AMs);
      ModulePassManager Inline2;
      Inline2.addPass(AlwaysInlinerPass());
      Inline2.run(M, AMs.MAM);
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
    const Options &O = sessionOf(M, MAM).Opts;
    bool changed = false;
    if (Target::of(M).isDevice())
      changed = O.Profiling.profileAction() && hasSite(O, M)
                    ? keepDeviceMath(M)
                    : releaseDeviceMath(M);
    changed |= lowerPerCallSites(O, M);
    changed |= propagateDeviceSites(M);
    changed |= outlineHostSites(O, M);
    changed |= registerDeviceProfileTable(M, O.Profiling.ProfileGenerate);
    if (!O.Profiling.profileAction())
      changed |= leaveSitesAsWritten(O, M);
    return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
  static bool isRequired() { return true; }
};

// A module with a site is compiled for the solve otherwise than as written in
// every function, not only the rewritten ones: the canonical prefix, a site's
// body or per-call copy kept out of line until it is lowered (150 registers
// against 134 for one finite-element kernel; a host loop over the call no
// longer vectorized), and on a device libdevice math kept out of line through
// the inliner. The module is compiled once more, from its state before any of
// that and as a build without a profile compiles it, and every function no
// rewrite reached is emitted from that compile. Not the host half of a CUDA or
// HIP compilation, which holds no site of the call or the annotation form and
// whose launch stubs are rewritten for its device half's sites at
// PipelineStart, where no rewrite is recorded.
// At PipelineStart, before HostSitePass.
struct TwinSnapshotPass : PassParent<TwinSnapshotPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
    Session &S = sessionOf(M, MAM);
    const Options &O = S.Opts;
    if (S.IsTwin || isOffloadHostHalf(M) || !hasSite(O, M))
      return PreservedAnalyses::all();
    if (O.Profiling.usesProfile() && !O.Profiling.ProfileGenerate &&
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
          F.getName().starts_with("__enzyme_autodiff_poseidon"))
        reportUserError(M.getContext(),
                        Twine("profile generation emitted a derivative "
                              "request that nothing lowered. ") +
                            unloweredRequestReason(PhaseKnown, Phase, Level));
    // Every site of the module has been processed by now, in the early run or
    // the late one. A module without a descriptor removes the unit's manifest,
    // so that what an earlier compile of the unit recorded is not applied.
    const Target &T = Target::of(M);
    if (T.isDevice() && !S.IsTwin && handsOverLaunches(O))
      writeLaunchManifest(S, M);
    finalizeModule(S, M);
    // An annotation takes the address of the function it names, and a profiled
    // kernel may make an unknown call.
    if (O.Profiling.ProfileGenerate && T.allocatesAddressTakenShared())
      dropSiteAnnotations(M);
    linkDeviceProfiler(M, O.Profiling.ProfileGenerate);
    requireDeviceCallees(M, O.Profiling.ProfileGenerate);
    releaseDeviceMath(M);
    SmallVector<Function *, 4> rewritten;
    for (WeakVH &handle : S.RewrittenFunctions)
      if (auto *F = dyn_cast_or_null<Function>(handle);
          F && F->getParent() == &M && !F->isDeclaration() &&
          !is_contained(rewritten, F))
        rewritten.push_back(F);
    erase_if(S.RewrittenFunctions, [&](WeakVH &handle) {
      auto *F = dyn_cast_or_null<Function>(handle);
      return !F || F->getParent() == &M;
    });
    for (Function *F : rewritten)
      inlineDeviceMath(*F);
    if (std::unique_ptr<Module> Twin = std::move(S.AsWrittenTwin)) {
      if (PB && compileAsWritten(*Twin, S, *PB, Level, Phase)) {
        if (takeRewritten(*Twin, rewritten)) {
          adoptModule(M, *Twin);
          if (O.Diagnostics.Print && rewritten.empty())
            llvm::errs() << "[poseidon] every site of " << M.getName()
                         << " was left alone; emitting the module as written\n";
          else if (O.Diagnostics.Print)
            llvm::errs() << "[poseidon] " << M.getName() << ": "
                         << rewritten.size()
                         << " function(s) a rewrite reached; every other "
                            "function as written\n";
        } else {
          llvm::errs() << "[poseidon] " << M.getName()
                       << ": a rewritten function refers to what the module "
                          "as written does not hold alike; emitting the "
                          "rewritten module\n";
        }
      }
    }
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
      return O.Profiling.ProfileGenerate && injectHostProfilerInit(M)
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

// The prefix of Enzyme's OptimizerEarly pipeline that a site body passes
// through before Poseidon sees it. Poseidon owns a copy because its own
// callback runs first, and the canonical form the profile is numbered against
// is the form after these passes.
void addCanonicalPrefix(ModulePassManager &MPM, OptimizationLevel Level) {
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

} // namespace

void registerPasses(PassBuilder &PB) {
  // Every pass reads the options from the session of the module it runs on, so
  // the module compiled as written, whose session carries its own, runs the
  // same pipeline.
  PB.registerAnalysisRegistrationCallback(
      [O = readOptions()](ModuleAnalysisManager &MAM) {
        MAM.registerPass([&] { return SessionAnalysis(O); });
      });

  auto loadHostStub = [](ModulePassManager &MPM, OptimizationLevel) {
    MPM.addPass(HostSitePass());
    // Host-side GEMM dispatch must rewrite the launch stub BEFORE inlining, so
    // it runs at PipelineStart. No-op unless -poseidon-ozaki-host-dispatch is
    // set and the module is a host module.
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
    // Fold the outlined site bodies and the materialized clones back into
    // their kernels here; nothing later in the pipeline inlines, and the
    // AlwaysInlinerPass the Enzyme plugin adds is another plugin's business.
    Sites.addPass(AlwaysInlinerPass());
    MPM.addPass(IfSitePass(std::move(Prefix), std::move(Sites)));
  };
  // The late run owns the sites the early run deferred: the canonical prefix,
  // the site handler, the always-inliner and, in profile-generate mode, a
  // second Enzyme run, all inside LatePass because they must not happen at all
  // in a module that deferred nothing. FinalizePass's "nothing lowered this
  // request" check therefore runs after that second Enzyme run.
  PassBuilder *PBp = &PB;
  // The module as written is compiled with the default pipeline of the phase
  // the module is in: a compile from source, also one that writes LTO bitcode
  // (every HIP device compile does). An LTO link has no module as written.
  auto loadFinalize = [PBp](ModulePassManager &MPM, OptimizationLevel Level,
                            ThinOrFullLTOPhase Phase) {
    const bool fromSource = Phase == ThinOrFullLTOPhase::None ||
                            Phase == ThinOrFullLTOPhase::ThinLTOPreLink ||
                            Phase == ThinOrFullLTOPhase::FullLTOPreLink;
    MPM.addPass(LatePass(PBp, Level, Phase));
    MPM.addPass(FinalizePass(fromSource ? PBp : nullptr, Level, Phase));
  };

#if LLVM_VERSION_MAJOR >= 20
  PB.registerOptimizerEarlyEPCallback(
      [loadPoseidon](ModulePassManager &MPM, OptimizationLevel Level,
                     ThinOrFullLTOPhase) { loadPoseidon(MPM, Level); });
  PB.registerOptimizerLastEPCallback(loadFinalize);
#else
  PB.registerOptimizerEarlyEPCallback(loadPoseidon);
  PB.registerOptimizerLastEPCallback(
      [loadFinalize](ModulePassManager &MPM, OptimizationLevel Level) {
        loadFinalize(MPM, Level, ThinOrFullLTOPhase::None);
      });
#endif
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
      [](StringRef Name, ModulePassManager &MPM,
         ArrayRef<PassBuilder::PipelineElement>) {
        if (Name == "poseidon") {
          MPM.addPass(OptimizePass());
          return true;
        }
        if (Name == "poseidon-finalize") {
          MPM.addPass(FinalizePass());
          return true;
        }
        return false;
      });

  PB.registerPipelineParsingCallback(
      [PBp](StringRef Name, ModulePassManager &MPM,
            ArrayRef<PassBuilder::PipelineElement>) {
        if (Name == "poseidon-late") {
          MPM.addPass(LatePass(PBp, OptimizationLevel::O3));
          return true;
        }
        return false;
      });
}

} // namespace poseidon
