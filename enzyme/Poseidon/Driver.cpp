#include "Driver.h"

#include "Canonicalize.h"
#include "CostModel.h"
#include "Flags.h"
#include "HostDispatch.h"
#include "Instrument.h"
#include "LaunchDescriptors.h"
#include "Matmul.h"
#include "Optimize.h"
#include "ProfileRead.h"
#include "RaiseWMMA.h"
// The one profile-filename rule, shared verbatim with both FP profiler
// runtimes; the existence check below must look where they wrote.
#include "FPProfileName.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/Scalar/Float2Int.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/LoopDeletion.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include "llvm/Transforms/Scalar/LoopRotation.h"
#include "llvm/Transforms/Scalar/LoopUnrollPass.h"
#include "llvm/Transforms/Scalar/LowerConstantIntrinsics.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <optional>
#include <string>
#include <unordered_map>

using namespace llvm;

namespace poseidon {

bool isMarkerCall(StringRef calleeName) {
  return calleeName.contains("__poseidon_fp_optimize");
}

namespace {

void warn(CallInst *CI, const Twine &message) {
  CI->getContext().diagnose(DiagnosticInfoUnsupported(
      *CI->getFunction(), "Poseidon: " + message.str(), CI->getDebugLoc(),
      DS_Warning));
}

void fail(CallInst *CI, const Twine &message) {
  CI->getContext().diagnose(DiagnosticInfoUnsupported(
      *CI->getFunction(), "Poseidon: " + message.str(), CI->getDebugLoc(),
      DS_Error));
}

// The call's function type may disagree with the declaration's when the
// annotation is declared fully variadic, and getCalledFunction() gives up in
// that case.
Function *calleeOf(CallInst *CI) {
  Value *callee = CI->getCalledOperand();
  if (auto *ce = dyn_cast<ConstantExpr>(callee); ce && ce->isCast())
    callee = ce->getOperand(0);
  return dyn_cast<Function>(callee);
}

// The same walk without the diagnostics, for the scans that only need to know
// which body a marker names (a malformed marker is reported once, by the
// handler that tries to lower it). An empty body is returned as found; only
// markerTarget refuses it.
Function *markerTargetOrNull(CallInst *CI) {
  Value *fn = CI->getArgOperand(CI->hasStructRetAttr() ? 1 : 0);
  while (!isa<Function>(fn)) {
    if (auto *ci = dyn_cast<CastInst>(fn)) {
      fn = ci->getOperand(0);
    } else if (auto *ce = dyn_cast<ConstantExpr>(fn); ce && ce->isCast()) {
      fn = ce->getOperand(0);
    } else if (auto *ba = dyn_cast<BlockAddress>(fn)) {
      fn = ba->getFunction();
    } else if (auto *ga = dyn_cast<GlobalAlias>(fn)) {
      fn = ga->getAliasee();
    } else {
      return nullptr;
    }
  }
  return cast<Function>(fn);
}

Function *markerTarget(CallInst *CI) {
  Function *F = markerTargetOrNull(CI);
  if (!F) {
    std::string buf;
    llvm::raw_string_ostream os(buf);
    os << *CI->getArgOperand(CI->hasStructRetAttr() ? 1 : 0);
    fail(CI, "failed to find the annotated function in " + buf);
    return nullptr;
  }
  if (F->empty()) {
    fail(CI, "the annotated function " + F->getName() + " has no body");
    return nullptr;
  }
  return F;
}

// --------------------------------------------------------------------------
// Deferred sites: a marked body that is still an __enzyme_* request
// --------------------------------------------------------------------------

// A marked body whose content is a derivative request holds no arithmetic
// until Enzyme has lowered it, so at OptimizerEarly there is nothing to
// profile and nothing to rewrite. Such a site is DEFERRED: its marker call is
// left in place and the body is marked, and the late run
// (registerOptimizerLastEPCallback, after Enzyme) handles it.
constexpr StringLiteral kSiteDeferred = "poseidon-deferred";

// The Enzyme requests whose result IS the body's content: until one of them is
// lowered the body holds no arithmetic. Spelled as substrings and matched with
// contains() because a C++ frontend mangles the request's template
// instantiation, which is how Enzyme itself recognizes them
// (Enzyme/Enzyme.cpp:2419-2468).
bool isEnzymeRequestName(StringRef name) {
  static constexpr StringLiteral kRequests[] = {
      "__enzyme_autodiff",   "__enzyme_fwddiff",  "__enzyme_fwdsplit",
      "__enzyme_augmentfwd", "__enzyme_reverse",  "__enzyme_virtualreverse",
      "__enzyme_batch",      "__enzyme_truncate", "__enzyme_error_estimate",
      "__enzyme_trace"};
  for (StringRef req : kRequests)
    if (name.contains(req))
      return true;
  return false;
}

// Enzyme names what it generates after the function it differentiated, which
// is the rule enzymeAlreadyRan below already relies on.
bool isEnzymeGeneratedName(StringRef name) {
  return name.starts_with("fwddiffe") || name.starts_with("diffe") ||
         name.starts_with("augmented_") || name.starts_with("fwdsplit") ||
         name.starts_with("batch_");
}

// Fold the code Enzyme generated for the request into the deferred body. What
// Enzyme leaves behind is a body holding one call to the generated function,
// and nothing in the -O2/-O3 pipeline inlines after OptimizerEarly, so without
// this the site is still a call and still has no arithmetic. The frozen
// integration had it for free: it ran inside lowerEnzymeCalls, which folded the
// generated function into the marked body before the site handler saw it.
// Only the generated functions are folded; a libdevice call the body makes
// stays a call, so the solve prices it exactly as it prices one in a primal
// site.
unsigned foldEnzymeGenerated(Function &body) {
  unsigned folded = 0;
  for (unsigned round = 0; round < 8; ++round) {
    SmallVector<CallInst *, 4> todo;
    for (Instruction &I : instructions(body))
      if (auto *CI = dyn_cast<CallInst>(&I))
        if (Function *callee = calleeOf(CI))
          if (!callee->isDeclaration() && callee != &body &&
              isEnzymeGeneratedName(callee->getName()))
            todo.push_back(CI);
    if (todo.empty())
      break;
    for (CallInst *CI : todo) {
      InlineFunctionInfo IFI;
      if (InlineFunction(*CI, IFI).isSuccess())
        ++folded;
    }
  }
  return folded;
}

// Whether `F`, or anything it reaches through a direct call, still asks for a
// derivative. The walk goes through the always-inline wrappers a frontend puts
// between the marked body and the request.
bool callsEnzymeRequest(Function &F) {
  SmallPtrSet<Function *, 8> seen;
  SmallVector<Function *, 8> work;
  seen.insert(&F);
  work.push_back(&F);
  while (!work.empty()) {
    Function *cur = work.pop_back_val();
    for (Instruction &I : instructions(*cur)) {
      auto *CI = dyn_cast<CallInst>(&I);
      if (!CI)
        continue;
      Function *callee = calleeOf(CI);
      if (!callee)
        continue;
      // Poseidon's own request is emitted over an instrumented clone, never
      // inside a body it is about to profile.
      if (isEnzymeRequestName(callee->getName()))
        return true;
      if (!callee->isDeclaration() && seen.insert(callee).second)
        work.push_back(callee);
    }
  }
  return false;
}

// True when this site belongs to the late run. Marks the body the first time,
// so that both phases and both passes agree without re-deciding.
bool deferSite(CallInst *CI) {
  Function *F = markerTargetOrNull(CI);
  if (!F || F->empty())
    return false;
  if (F->hasFnAttribute(kSiteDeferred))
    return true;
  if (!callsEnzymeRequest(*F))
    return false;
  F->addFnAttr(kSiteDeferred);
  if (flags::Print)
    llvm::errs() << "[poseidon] " << F->getName()
                 << ": marked body still holds an __enzyme_* request; "
                    "deferred to the late run\n";
  return true;
}

// The marker calls this module still owes the late run.
void collectDeferredMarkers(Module &M, SmallVectorImpl<CallInst *> &out) {
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    for (Instruction &I : instructions(F)) {
      auto *CI = dyn_cast<CallInst>(&I);
      if (!CI)
        continue;
      Function *callee = calleeOf(CI);
      if (!callee || !isMarkerCall(callee->getName()))
        continue;
      Function *target = markerTargetOrNull(CI);
      if (target && target->hasFnAttribute(kSiteDeferred))
        out.push_back(CI);
    }
  }
}

// Site ids number the profiled clones of one module and arm the
// condition-number perturbation one site at a time, so the late run has to
// continue the early run's numbering rather than restart it. Carried on the
// module because the two runs are different pass instances.
constexpr StringLiteral kNextSiteIdMD = "poseidon.next.site.id";

unsigned getNextSiteId(Module &M) {
  auto *NMD = M.getNamedMetadata(kNextSiteIdMD);
  if (!NMD || NMD->getNumOperands() == 0)
    return 0;
  auto *C = mdconst::extract<ConstantInt>(NMD->getOperand(0)->getOperand(0));
  return (unsigned)C->getZExtValue();
}

void setNextSiteId(Module &M, unsigned next) {
  auto *NMD = M.getOrInsertNamedMetadata(kNextSiteIdMD);
  NMD->clearOperands();
  NMD->addOperand(MDNode::get(M.getContext(),
                              ConstantAsMetadata::get(ConstantInt::get(
                                  Type::getInt32Ty(M.getContext()), next))));
}

// The name an Enzyme activity marker carries, however the frontend spelled it:
// a metadata string in hand-written IR, a load of the __device__ int global in
// CUDA sources.
std::optional<StringRef> activityMarkerName(Value *res) {
  if (auto *av = dyn_cast<MetadataAsValue>(res))
    if (auto *ms = dyn_cast<MDString>(av->getMetadata()))
      return ms->getString();
  if (auto *I = dyn_cast<Instruction>(res))
    if (isa<LoadInst>(I) || isa<CastInst>(I))
      res = I->getOperand(0);
  if (auto *ce = dyn_cast<ConstantExpr>(res); ce && ce->isCast())
    res = ce->getOperand(0);
  if (auto *gv = dyn_cast<GlobalVariable>(res))
    return gv->getName();
  return std::nullopt;
}

// What the marker call says about one site: the caller value behind each of the
// site's parameters, the accuracy target and the confidence level it is read
// at, and which marker arguments are Poseidon's own and must not be forwarded
// to the derivative request.
struct MarkerArgs {
  SmallVector<Value *, 8> primalArgs;
  SmallVector<bool, 16> forward;
  double errTol = 0.0;
  // 0 = the site wrote none, so -poseidon-confidence decides.
  double confidence = 0.0;
};

bool parseMarker(CallInst *CI, Function *F, MarkerArgs &out) {
  bool sret =
      CI->hasStructRetAttr() || F->hasParamAttribute(0, Attribute::StructRet);
  out.forward.assign(CI->arg_size(), true);

  for (size_t i = 1 + sret; i < CI->arg_size(); ++i) {
    Value *res = CI->getArgOperand(i);
    std::optional<bool> dup;
    bool skipArg = false;

    while (auto name = activityMarkerName(res)) {
      // Everything forwarded to the derivative request is spelled enzyme_*;
      // poseidon_tau and poseidon_confidence are Poseidon's own and are
      // consumed here.
      if (!name->starts_with("enzyme_") && *name != "poseidon_tau" &&
          *name != "poseidon_confidence")
        break;
      if (*name == "enzyme_dup" || *name == "enzyme_dupnoneed") {
        dup = true;
      } else if (*name == "enzyme_out" || *name == "enzyme_const") {
        dup = false;
      } else if (*name == "poseidon_tau") {
        out.forward[i] = false;
        ++i;
        if (i == CI->arg_size()) {
          fail(CI, "poseidon_tau without a value");
          return false;
        }
        auto *CFP = dyn_cast<ConstantFP>(CI->getArgOperand(i));
        if (!CFP) {
          fail(CI, "the relative error tolerance must be a floating-point "
                   "constant");
          return false;
        }
        out.errTol = CFP->getValueAPF().convertToDouble();
        out.forward[i] = false;
        skipArg = true;
        break;
      } else if (*name == "poseidon_confidence") {
        out.forward[i] = false;
        ++i;
        if (i == CI->arg_size()) {
          fail(CI, "poseidon_confidence without a value");
          return false;
        }
        auto *CFP = dyn_cast<ConstantFP>(CI->getArgOperand(i));
        if (!CFP) {
          fail(CI, "the accuracy-target confidence level must be a "
                   "floating-point constant");
          return false;
        }
        out.confidence = CFP->getValueAPF().convertToDouble();
        if (!(out.confidence > 0.0) || out.confidence > 1.0) {
          fail(CI, "poseidon_confidence " + std::to_string(out.confidence) +
                       " is outside (0, 1]");
          return false;
        }
        out.forward[i] = false;
        skipArg = true;
        break;
      } else if (*name == "enzyme_not_overwritten") {
        // A modifier of the marker that follows, not a marker of its own; the
        // decoding has to match Enzyme's handleArguments or the activity of the
        // argument it prefixes is lost.
      } else if (*name == "enzyme_noret" || *name == "enzyme_primal_return" ||
                 *name == "enzyme_const_return" ||
                 *name == "enzyme_active_return" ||
                 *name == "enzyme_dup_return" ||
                 *name == "enzyme_runtime_activity" ||
                 *name == "enzyme_strong_zero" || *name == "enzyme_nofree") {
        skipArg = true;
        break;
      } else {
        fail(CI, "unsupported Poseidon annotation argument " + *name);
        return false;
      }
      ++i;
      if (i == CI->arg_size()) {
        fail(CI, "activity marker without an argument");
        return false;
      }
      res = CI->getArgOperand(i);
    }
    if (skipArg)
      continue;

    size_t param = out.primalArgs.size() + sret;
    if (param >= F->getFunctionType()->getNumParams()) {
      fail(CI, "more arguments than " + F->getName() + " has parameters");
      return false;
    }
    if (!dup) {
      Type *PTy = F->getFunctionType()->getParamType(param);
      if (PTy->isPointerTy())
        dup = true;
      else if (PTy->isFloatingPointTy() || PTy->isIntegerTy())
        dup = false;
      else {
        fail(CI, "parameter " + Twine(param) + " of " + F->getName() +
                     " needs an explicit activity marker");
        return false;
      }
    }
    out.primalArgs.push_back(res);
    if (*dup)
      ++i;
  }

  // A confidence level is the level the site's own accuracy target is read at,
  // so it says nothing on its own: without a target the site is solved against
  // the compute budget, which has no percentile in it.
  if (out.confidence > 0.0 && !(out.errTol > 0.0)) {
    fail(CI, "poseidon_confidence needs a poseidon_tau at the same call; a "
             "confidence level is the level this site's own accuracy target "
             "is read at");
    return false;
  }

  size_t expected = F->getFunctionType()->getNumParams() - sret;
  if (out.primalArgs.size() != expected) {
    fail(CI, "decoded " + Twine(out.primalArgs.size()) + " argument(s) for " +
                 F->getName() + ", which takes " + Twine(expected));
    return false;
  }
  return true;
}

SmallVector<WeakVH, 4> &siteClones() {
  static SmallVector<WeakVH, 4> clones;
  return clones;
}

// The body a site's call targets is a device of this pass and must not reach
// codegen as a call: under -fcuda-rdc a call that is not folded back is a real
// ABI call, and the kernel entry then pays its register and frame budget
// (measured: 148 registers and one resident block per SM against 70 and three
// for the same drift kernel without the annotation, a 2x wall-clock gap on
// GH200; and 91 registers against 62 for the elasticity reduce, which at 648
// threads per block exceeds the 64K per-block register file and makes the
// launch fail outright). Marked on the solve path for both annotation forms, so
// that a site the solve leaves unchanged is inlined exactly like one it
// rewrote; the AlwaysInlinerPass registerPasses schedules right after this pass
// does the folding.
void markOutlineTransparent(Function *body) {
  body->removeFnAttr(Attribute::NoInline);
  body->removeFnAttr(Attribute::OptimizeNone);
  body->addFnAttr(Attribute::AlwaysInline);
}

// Redirect the site's call to `F`: the rewritten clone, or the original body
// when nothing was applied and the site has to stay bit-transparent.
bool emitSiteCall(CallInst *CI, Function *F,
                  SmallVectorImpl<Value *> &primalArgs,
                  SmallVectorImpl<CallInst *> &calls) {
  IRBuilder<> Builder(CI);
  CallInst *optCall = Builder.CreateCall(F->getFunctionType(), F, primalArgs);
  optCall->setCallingConv(CI->getCallingConv());
  optCall->setDebugLoc(CI->getDebugLoc());

  CI->replaceAllUsesWith(optCall);
  CI->eraseFromParent();

  calls.push_back(optCall);

  return true;
}

// Profile use, shared by both annotation forms: `CI` is the call the site's
// body is reached through (the marker call, or the wrapper kernel's call of its
// own outlined body), `F` the body and `primalArgs` the caller values its
// parameters take.
bool optimizeSiteBody(CallInst *CI, Function *F,
                      SmallVectorImpl<Value *> &primalArgs, double errTol,
                      double siteConfidence,
                      SmallVectorImpl<CallInst *> &calls) {
  const bool profileUse =
      flags::ProfileUse.getNumOccurrences() && !flags::ProfileUse.empty();
  if (!profileUse) {
    warn(CI, "a Poseidon site was compiled without -poseidon-profile-generate "
             "or -poseidon-profile-use=<dir>. Emitting the original "
             "computation.");
  } else {
    // Kept for no-op transparency: if the solve ends up applying NOTHING to
    // this site, the wrapper must call the ORIGINAL body, not the
    // canonicalized clone (whose lowering is not guaranteed bit-identical to
    // the original).
    Function *sourceBody = F;
    F = canonicalize(*F);
    siteClones().push_back(F);
    setSlotMetadata(*F);
    noteSiteOrigin(F, sourceBody);

    SmallString<128> profilePath(flags::ProfileUse);
    llvm::sys::path::append(profilePath, siteProfileStem(*F) + ".fpprofile");
    // A site the profiling run never reached still has to build and run, so it
    // keeps the ORIGINAL body rather than failing the compile.
    if (!llvm::sys::fs::exists(profilePath.str())) {
      warn(CI, Twine("no profile found at ") + profilePath.str() +
                   " (-poseidon-profile-use=" + StringRef(flags::ProfileUse) +
                   "); leaving the site unchanged");
      llvm::errs() << "[poseidon] " << sourceBody->getName()
                   << ": no profile at " << profilePath.str()
                   << "; left unchanged\n";
      markOutlineTransparent(sourceBody);
      return emitSiteCall(CI, sourceBody, primalArgs, calls);
    }

    if (flags::Print) {
      llvm::errs() << "[poseidon] Optimizing " << F->getName()
                   << " with relative error tolerance: " << errTol << "\n";
    }

    if (flags::JointDP) {
      // Joint mode: defer; mark F with its tolerance so the joint solve takes
      // it with the module's other sites under one shared budget.
      F->addFnAttr("poseidon-joint-errtol", std::to_string(errTol));
      if (siteConfidence > 0.0)
        F->addFnAttr("poseidon-joint-confidence",
                     std::to_string(siteConfidence));
      // Host-dispatch note is produced only by the later joint materialize;
      // capture the body-param -> wrapper-arg mapping now (primalArgs valid
      // here).
      if (flags::OzakiHostDispatch)
        addPendingGemmDispatch(F, *CI->getFunction(), primalArgs, flags::Cache);
    } else {
      bool optimized = fpOptimize(*F, errTol, siteConfidence);

      if (!optimized)
        warn(CI,
             Twine("Poseidon returned false (no change) for ") + F->getName());

      // Host-side GEMM dispatch: if fpOptimize recorded this body as a raised
      // GEMM to be dispatched host-side (-poseidon-ozaki-host-dispatch), write
      // the launch descriptor (wrapper kernel + C/A/B launch-arg indices +
      // geometry) for the host sub-compilation to consume.
      {
        GemmBodyNote ozNote;
        if (flags::OzakiHostDispatch && getGemmBody(F, ozNote))
          writeGemmDescriptor(*CI->getFunction(), primalArgs, ozNote,
                              flags::Cache);
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
    }
  }

  // Profile-generate leaves the marker call in place for the profiler, so this
  // only ever fires on the solve path.
  if (profileUse)
    markOutlineTransparent(F);
  return emitSiteCall(CI, F, primalArgs, calls);
}

bool optimizeSite(CallInst *CI, SmallVectorImpl<CallInst *> &calls) {
  Function *F = markerTarget(CI);
  if (!F)
    return false;

  MarkerArgs marker;
  if (!parseMarker(CI, F, marker))
    return false;
  return optimizeSiteBody(CI, F, marker.primalArgs, marker.errTol,
                          marker.confidence, calls);
}

// --------------------------------------------------------------------------
// Attribute sites: a kernel carrying POSEIDON_OPTIMIZE
// (__attribute__((annotate("poseidon")))) is a site whose annotated
// computation is its whole body.
// --------------------------------------------------------------------------

constexpr StringLiteral kAnnotation = "poseidon";
constexpr StringLiteral kTauModifier = "tau=";

// A site annotation is "poseidon", or "poseidon;tau=<value>" when the source
// wrote POSEIDON_OPTIMIZE_TAU. False means the string belongs to some other
// tool; an unreadable Poseidon modifier is an error rather than a site without
// the target its source asked for.
bool parseSiteAnnotation(StringRef text, StringRef fnName,
                         std::optional<double> &tau) {
  if (!text.consume_front(kAnnotation))
    return false;
  if (text.empty())
    return true;
  if (!text.consume_front(";"))
    return false;
  if (!text.consume_front(kTauModifier))
    report_fatal_error(Twine("Poseidon: ") + fnName +
                       " carries an unknown Poseidon annotation modifier '" +
                       text + "'");
  double value = 0.0;
  if (text.getAsDouble(value) || !(value > 0.0) || !std::isfinite(value))
    report_fatal_error(Twine("Poseidon: ") + fnName +
                       " carries the accuracy target '" + text +
                       "', which is not a positive finite number");
  tau = value;
  return true;
}

// The attribute stays on the kernel after its site has been lowered, and the
// pass reaches a module more than once, so a lowered kernel has to say so or
// its wrapper body is outlined again as a second site.
constexpr StringLiteral kSiteLowered = "poseidon-site-lowered";

bool isLaunchStub(const Function &F) {
  return F.getName().contains("__device_stub__");
}

// A site's profiled clone takes a shadow per pointer argument, which only a
// kernel launch can supply; any other annotated function stays as written.
constexpr StringLiteral kNotASite = "poseidon-not-a-site";

void rejectAnnotatedFunction(Function &F) {
  if (F.hasFnAttribute(kNotASite))
    return;
  F.addFnAttr(kNotASite);
  F.getContext().diagnose(DiagnosticInfoUnsupported(
      F,
      "Poseidon: POSEIDON_OPTIMIZE marks GPU kernels; on the host, wrap the "
      "call in __poseidon_fp_optimize. Emitting the original computation.",
      F.getSubprogram(), DS_Warning));
}

// The kernels of this module that are sites, in module order so that the site
// numbering does not depend on the order a container happened to hash them in.
// `byRegex`, when given, receives the sites -poseidon-kernels named and the
// attribute did not: those are the ones the cost-share filter applies to.
// `tauOut`, when given, receives the accuracy target of each site whose
// annotation carried one.
void collectAnnotatedSites(Module &M, SmallVectorImpl<Function *> &out,
                           SmallPtrSetImpl<Function *> *byRegex = nullptr,
                           DenseMap<Function *, double> *tauOut = nullptr) {
  SmallPtrSet<Function *, 8> found;

  if (auto *GA = M.getGlobalVariable("llvm.global.annotations")) {
    if (auto *arr = dyn_cast_or_null<ConstantArray>(GA->getInitializer())) {
      for (Value *op : arr->operands()) {
        auto *entry = dyn_cast<ConstantStruct>(op);
        if (!entry || entry->getNumOperands() < 2)
          continue;
        auto *fn =
            dyn_cast<Function>(entry->getOperand(0)->stripPointerCasts());
        auto *ann =
            dyn_cast<GlobalVariable>(entry->getOperand(1)->stripPointerCasts());
        if (!fn || fn->isDeclaration() || !ann || !ann->hasInitializer())
          continue;
        auto *text = dyn_cast<ConstantDataArray>(ann->getInitializer());
        if (!text || !text->isCString())
          continue;
        std::optional<double> tau;
        if (!parseSiteAnnotation(text->getAsCString(), fn->getName(), tau))
          continue;
        // In a CUDA host compilation the same annotation lands on the launch
        // stub; the site itself lives in the device module.
        if (isLaunchStub(*fn) || fn->hasFnAttribute(kSiteLowered))
          continue;
        if (fn->getCallingConv() != CallingConv::PTX_Kernel) {
          rejectAnnotatedFunction(*fn);
          continue;
        }
        found.insert(fn);
        if (tau && tauOut) {
          auto [it, inserted] = tauOut->try_emplace(fn, *tau);
          if (!inserted && it->second != *tau)
            report_fatal_error(Twine("Poseidon: ") + fn->getName() +
                               " carries two different accuracy targets");
        }
      }
    }
  }

  if (!flags::Kernels.empty()) {
    bool all = flags::Kernels == "all";
    Regex re(flags::Kernels);
    std::string err;
    if (!all && !re.isValid(err))
      report_fatal_error(Twine("-poseidon-kernels=") + flags::Kernels +
                         " is not a valid regular expression: " + err);
    for (Function &F : M)
      if (!F.isDeclaration() && !F.hasFnAttribute(kSiteLowered) &&
          F.getCallingConv() == CallingConv::PTX_Kernel &&
          (all || re.match(F.getName())))
        if (found.insert(&F).second && byRegex)
          byRegex->insert(&F);
  }

  for (Function &F : M)
    if (found.count(&F))
      out.push_back(&F);
}

// Make the kernel a wrapper around its own body so that both phases see the
// structure they already handle: a function holding the annotated computation,
// reached through one call whose arguments are the site's inputs. Returns that
// call, or null if the kernel cannot be a site.
CallInst *outlineKernelBody(Function &K, Function *&bodyOut) {
  if (K.isVarArg() || !K.getReturnType()->isVoidTy()) {
    K.getContext().diagnose(DiagnosticInfoUnsupported(
        K,
        "Poseidon: an annotated kernel must be a void, non-variadic "
        "function",
        K.getSubprogram(), DS_Warning));
    return nullptr;
  }
  Module &M = *K.getParent();
  Function *B =
      Function::Create(K.getFunctionType(), GlobalValue::InternalLinkage,
                       K.getName() + "_poseidon_body", &M);
  B->setAttributes(K.getAttributes());
  // The body is an ordinary device function; only the kernel stays an entry.
  B->setCallingConv(CallingConv::C);
  B->setSubprogram(K.getSubprogram());
  K.setSubprogram(nullptr);
  B->splice(B->begin(), &K);
  for (auto i = K.arg_begin(), j = B->arg_begin(); i != K.arg_end(); ++i, ++j) {
    j->setName(i->getName());
    i->replaceAllUsesWith(&*j);
  }

  IRBuilder<> Builder(BasicBlock::Create(M.getContext(), "entry", &K));
  SmallVector<Value *, 8> args;
  for (Argument &A : K.args())
    args.push_back(&A);
  CallInst *call = Builder.CreateCall(B, args);
  Builder.CreateRetVoid();
  K.addFnAttr(kSiteLowered);
  bodyOut = B;
  return call;
}

// Canonicalize, slot-number and instrument one site body exactly as profile
// use canonicalizes and slot-numbers it, so the profile is recorded against
// the form the solve will later see. Null if the body holds nothing profilable.
Function *instrumentedClone(Function &F, unsigned siteId) {
  Function *clone = canonicalize(F);
  siteClones().push_back(clone);
  setSlotMetadata(*clone);
  // Before the probes: they break the reduction shape the trip count is read
  // from.
  SmallVector<std::pair<size_t, unsigned>, 4> trips;
  collectScalarLoopReductionTrips(*clone, trips);
  if (!instrumentForProfiling(*clone, siteId)) {
    clone->eraseFromParent();
    return nullptr;
  }
  std::string staticData;
  {
    raw_string_ostream os(staticData);
    os << "SiteId = " << siteId << "\n";
    for (const auto &kv : trips)
      os << "RedTrip = " << kv.first << " " << kv.second << "\n";
    os << "CanonicalHash = " << canonicalFormHash(*clone) << "\n";
  }
  emitProfileStaticData(*clone, staticData);
  return clone;
}

// Profile generation. The marker becomes an ordinary reverse-mode derivative
// request over the instrumented clone: the custom derivative on each probe is
// what puts the gradient record inside the reverse pass, so the host's AD needs
// no profiling mode of its own.
bool profileSite(CallInst *CI, DenseMap<Function *, Function *> &clones,
                 unsigned &nextSiteId) {
  Function *F = markerTarget(CI);
  if (!F)
    return false;

  MarkerArgs marker;
  if (!parseMarker(CI, F, marker))
    return false;

  // Two annotations on the same body share one instrumented clone, so their
  // executions accumulate into one profile record, as they did when the clone
  // came from the host's preprocessing cache.
  Function *&clone = clones[F];
  if (!clone) {
    // Two markers on the same body share one profile record and therefore one
    // site id; the id counts clones in marker order.
    clone = instrumentedClone(*F, nextSiteId++);
    if (!clone) {
      warn(CI, Twine("no profilable operation in ") + F->getName());
      return false;
    }
  }

  Module &M = *CI->getModule();
  FunctionType *FT = CI->getFunctionType();
  Function *autodiff = Function::Create(FT, GlobalValue::ExternalLinkage,
                                        "__enzyme_autodiff_poseidon", &M);
  autodiff->setCallingConv(CI->getCallingConv());

  SmallVector<Value *, 16> args;
  SmallVector<size_t, 16> from;
  bool sret =
      CI->hasStructRetAttr() || F->hasParamAttribute(0, Attribute::StructRet);
  for (size_t i = 0; i < CI->arg_size(); ++i) {
    if (!marker.forward[i])
      continue;
    args.push_back(i == (size_t)sret ? clone : CI->getArgOperand(i));
    from.push_back(i);
  }

  IRBuilder<> Builder(CI);
  CallInst *adCall = Builder.CreateCall(FT, autodiff, args);
  adCall->setCallingConv(CI->getCallingConv());
  adCall->setDebugLoc(CI->getDebugLoc());
  AttributeList old = CI->getAttributes();
  adCall->setAttributes(AttributeList::get(M.getContext(), old.getFnAttrs(),
                                           old.getRetAttrs(), {}));
  for (size_t j = 0; j < from.size(); ++j)
    adCall->setAttributes(adCall->getAttributes().addParamAttributes(
        M.getContext(), j,
        AttrBuilder(M.getContext(), old.getParamAttrs(from[j]))));

  CI->replaceAllUsesWith(adCall);
  CI->eraseFromParent();
  return true;
}

// The width of the floating-point scalar a store writes, 0 if it writes none.
unsigned fpStoreWidth(Type *T) {
  if (auto *VT = dyn_cast<VectorType>(T))
    T = VT->getElementType();
  if (T->isFloatingPointTy())
    return T->getScalarSizeInBits() / 8;
  if (auto *ST = dyn_cast<StructType>(T))
    for (Type *E : ST->elements())
      if (unsigned w = fpStoreWidth(E))
        return w;
  if (auto *AT = dyn_cast<ArrayType>(T))
    return fpStoreWidth(AT->getElementType());
  return 0;
}

// Which parameters the site writes through, and in what element width. An
// application does not say which of its buffers are outputs, and reverse mode
// needs to know: the output shadows are the ones the seed goes into.
void collectOutputParams(Function &clone,
                         SmallVectorImpl<unsigned> &seedWidth) {
  seedWidth.assign(clone.arg_size(), 0);
  auto mark = [&](Value *ptr, unsigned w) {
    int idx = traceToArgIndex(ptr);
    if (w && idx >= 0 && (size_t)idx < seedWidth.size())
      seedWidth[idx] = std::max(seedWidth[idx], w);
  };
  for (Instruction &I : instructions(clone)) {
    if (auto *SI = dyn_cast<StoreInst>(&I))
      mark(SI->getPointerOperand(),
           fpStoreWidth(SI->getValueOperand()->getType()));
    else if (auto *RMW = dyn_cast<AtomicRMWInst>(&I))
      mark(RMW->getPointerOperand(),
           fpStoreWidth(RMW->getValOperand()->getType()));
  }
}

// Profile generation for an attribute site. The kernel keeps its name, its
// launch handle and its original parameters and gains one shadow parameter per
// pointer parameter; its body becomes the derivative request over the
// instrumented clone. The host half of the same compilation reads the
// descriptor written here and fills those shadows in at every launch.
bool profileAnnotatedKernel(Function &K, unsigned &nextSiteId) {
  Module &M = *K.getParent();
  LLVMContext &C = M.getContext();

  Function *body = nullptr;
  CallInst *call = outlineKernelBody(K, body);
  if (!call)
    return false;
  const unsigned nOrigArgs = K.arg_size();

  Function *clone = instrumentedClone(*body, nextSiteId);
  if (!clone) {
    InlineFunctionInfo IFI;
    (void)InlineFunction(*call, IFI);
    body->eraseFromParent();
    return false;
  }
  unsigned siteId = nextSiteId++;

  SmallVector<unsigned, 8> seedWidth;
  collectOutputParams(*clone, seedWidth);

  SmallVector<unsigned, 8> ptrParams;
  SmallVector<Type *, 16> params(K.getFunctionType()->params());
  Type *PtrTy = PointerType::getUnqual(C);
  for (unsigned i = 0; i < K.arg_size(); ++i)
    if (params[i]->isPointerTy()) {
      ptrParams.push_back(i);
      params.push_back(PtrTy);
    }

  Function *P =
      Function::Create(FunctionType::get(Type::getVoidTy(C), params, false),
                       K.getLinkage(), K.getName() + ".poseidon.profile", &M);
  P->setCallingConv(K.getCallingConv());
  // Function attributes only: the parameter attributes describe the primal
  // kernel's accesses, which say nothing about the shadows or about what the
  // reverse pass touches.
  P->setAttributes(AttributeList::get(C, K.getAttributes().getFnAttrs(),
                                      AttributeSet(), {}));
  P->setMemoryEffects(MemoryEffects::unknown());

  auto activity = [&](StringRef name) {
    return MetadataAsValue::get(C, MDString::get(C, name));
  };
  SmallVector<Value *, 32> args{clone};
  unsigned shadow = K.arg_size();
  for (unsigned i = 0; i < K.arg_size(); ++i) {
    Value *a = P->getArg(i);
    if (!a->getType()->isPointerTy()) {
      args.push_back(activity("enzyme_const"));
      args.push_back(a);
      continue;
    }
    // Every pointer is duplicated, inputs included: with a constant input the
    // computation reading it has no active argument and Enzyme drops the
    // reverse pass the profile is recorded from.
    args.push_back(activity("enzyme_dup"));
    args.push_back(a);
    args.push_back(P->getArg(shadow++));
  }

  SmallVector<Type *, 32> argTys;
  for (Value *a : args)
    argTys.push_back(a->getType());
  Function *autodiff = Function::Create(
      FunctionType::get(Type::getVoidTy(C), argTys, false),
      GlobalValue::ExternalLinkage, "__enzyme_autodiff_poseidon", &M);

  IRBuilder<> B(BasicBlock::Create(C, "entry", P));
  B.CreateCall(autodiff, args);
  B.CreateRetVoid();

  std::string kernelName = K.getName().str();
  P->addFnAttr(kSiteLowered);
  K.replaceAllUsesWith(P);
  K.eraseFromParent();
  if (body->use_empty())
    body->eraseFromParent();
  P->setName(kernelName);

  writeDescriptor(flags::Cache, kernelName, kProfGenScheme,
                  "[poseidon-profgen]", [&](raw_ostream &os) {
                    os << ' ' << siteId << ' ' << nOrigArgs << ' '
                       << ptrParams.size();
                    for (unsigned p : ptrParams)
                      os << ' ' << p;
                    for (unsigned p : ptrParams)
                      os << ' ' << seedWidth[p];
                  });
  errs() << "[poseidon-profgen] " << kernelName << ": site " << siteId << ", "
         << ptrParams.size() << " shadow buffer(s), output(s)";
  for (unsigned p : ptrParams)
    if (seedWidth[p])
      errs() << ' ' << p;
  errs() << "\n";
  return true;
}

} // namespace

bool lowerMarkers(ArrayRef<CallInst *> markers,
                  SmallVectorImpl<CallInst *> &calls) {
  bool changed = false;
  for (CallInst *CI : markers)
    changed |= optimizeSite(CI, calls);
  return changed;
}

// What the profiled run spent inside this site: over the slots the profile
// records, the execution count times the CSV price of the operation the slot
// names. A site the run never reached has no profile and costs nothing, which
// is what -poseidon-kernels=all over a whole library produces for most kernels.
double profiledSiteCost(Function &body) {
  SmallString<128> profilePath(flags::ProfileUse);
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
  requireCostModel(*clone);
  double cost = 0.0;
  size_t slot = 0;
  for (Instruction &I : instructions(*clone)) {
    if (!isOptimizable(I))
      continue;
    auto it = profileMap.find(slot++);
    if (it != profileMap.end())
      cost += (double)it->second.exec * getInstructionCompCost(&I);
  }
  clone->eraseFromParent();
  return cost;
}

// Undo outlineKernelBody: a filtered site has to leave the kernel exactly as it
// was found.
void inlineKernelBody(CallInst *call, Function *body) {
  InlineFunctionInfo IFI;
  (void)InlineFunction(*call, IFI);
  if (body->use_empty())
    body->eraseFromParent();
}

bool optimizeAnnotatedSites(Module &M) {
  if (flags::ProfileGenerate)
    return false;
  SmallVector<Function *, 4> sites;
  SmallPtrSet<Function *, 4> byRegex;
  DenseMap<Function *, double> siteTau;
  collectAnnotatedSites(M, sites, &byRegex, &siteTau);
  if (sites.empty())
    return false;

  // Without a profile nothing can be applied, so the kernels are left exactly
  // as written rather than outlined and never folded back.
  if (!flags::ProfileUse.getNumOccurrences() || flags::ProfileUse.empty()) {
    for (Function *K : sites)
      K->getContext().diagnose(DiagnosticInfoUnsupported(
          *K,
          "Poseidon: a Poseidon site was compiled without "
          "-poseidon-profile-generate or -poseidon-profile-use=<dir>. "
          "Emitting the original computation.",
          K->getSubprogram(), DS_Warning));
    return false;
  }

  bool changed = false;
  SmallVector<CallInst *, 4> calls;
  // Call-site target first, then the global flag; a site with neither is solved
  // against the cost budget instead of a tolerance.
  auto optimize = [&](CallInst *call, Function *body, Function *kernel) {
    SmallVector<Value *, 8> primalArgs(call->args());
    auto it = siteTau.find(kernel);
    double errTol = it != siteTau.end() ? it->second : (double)flags::Tau;
    // An attribute site writes no confidence level of its own yet, so
    // -poseidon-confidence decides for it.
    changed |= optimizeSiteBody(call, body, primalArgs, errTol,
                                /*siteConfidence=*/0.0, calls);
  };

  if (byRegex.empty() || flags::MinCostShare <= 0.0 ||
      flags::ProfileUse.empty()) {
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
    double cost = byRegex.count(K) ? profiledSiteCost(*body) : 0.0;
    regexTotal += cost;
    outlined.push_back({K, body, call, cost});
  }

  for (Site &S : outlined) {
    double share = regexTotal > 0.0 ? S.cost / regexTotal : 1.0;
    if (byRegex.count(S.kernel) && share < flags::MinCostShare) {
      llvm::errs() << "[poseidon] " << S.kernel->getName() << " is "
                   << format("%.4g", share * 100)
                   << "% of the profiled FP cost of the kernels "
                      "-poseidon-kernels named, below -poseidon-min-cost-share="
                   << format("%g", (double)flags::MinCostShare)
                   << "; left unoptimized\n";
      inlineKernelBody(S.call, S.body);
      continue;
    }
    optimize(S.call, S.body, S.kernel);
  }
  return changed;
}

void prepareModule(Module &M) {
  applyFlagDefaults();
  if (!flags::ProfileGenerate)
    return;
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
    if (deferSite(CI))
      continue;
    profileSite(CI, clones, nextSiteId);
    anySite = true;
  }

  SmallVector<Function *, 4> annotated;
  collectAnnotatedSites(M, annotated);
  for (Function *K : annotated)
    anySite |= profileAnnotatedKernel(*K, nextSiteId);
  setNextSiteId(M, nextSiteId);

  if (anySite) {
    // Launch geometry is recorded per profiled function; the probe is keyed by
    // the same poseidon_site_<name> global the value probes carry, so it has to
    // be placed after they exist.
    injectBlockDimProbes(M);
  }
}

bool solveDeferredSites(Module &M, FunctionAnalysisManager &FAM) {
  if (!flags::JointDP)
    return false;
  return solveJointly(M, FAM);
}

void finalizeModule(Module &M) {
  // A site whose solve applied nothing calls the original body again, and an
  // applied site's wrapper call may have been inlined away, in both cases
  // leaving a canonicalized clone nothing calls.
  for (WeakVH &handle : siteClones()) {
    auto *clone = dyn_cast_or_null<Function>(handle);
    if (clone && clone->getParent() == &M && clone->use_empty())
      clone->eraseFromParent();
  }
  siteClones().clear();
  // The site counter is a note the early run leaves for the late one; nothing
  // downstream reads it.
  if (auto *NMD = M.getNamedMetadata(kNextSiteIdMD))
    NMD->eraseFromParent();
}

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

bool hasSite(Module &M) {
  if (!flags::Kernels.empty() || M.getNamedGlobal("llvm.global.annotations"))
    return true;
  for (Function &F : M)
    if (F.isDeclaration() && isMarkerCall(F.getName()) && !F.use_empty())
      return true;
  return false;
}

void requirePluginOrder(Module &M) {
  if (!hasSite(M) || !enzymeAlreadyRan(M))
    return;
  report_fatal_error(
      "Poseidon: Enzyme's pass has already run on this module. The plugins "
      "must be given in the order -fpass-plugin=<Poseidon>.so "
      "-fpass-plugin=<ClangEnzyme>.so.");
}

// One module pass for both phases, at OptimizerEarly so that profile
// generation hands Enzyme the derivative requests it lowers later in the same
// extension point.
struct OptimizePass : PassParent<OptimizePass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
    requirePluginOrder(M);

    prepareModule(M);
    bool changed = optimizeAnnotatedSites(M);

    SmallVector<CallInst *, 4> markers, calls;
    for (Function &F : M) {
      if (F.empty())
        continue;
      markers.clear();
      for (BasicBlock &BB : F)
        for (Instruction &I : BB)
          if (auto *CI = dyn_cast<CallInst>(&I))
            if (Function *callee = calleeOf(CI))
              if (isMarkerCall(callee->getName()) && !deferSite(CI))
                markers.push_back(CI);
      if (!markers.empty())
        changed |= lowerMarkers(markers, calls);
    }

    auto &FAM =
        MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
    changed |= solveDeferredSites(M, FAM);
    return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
  static bool isRequired() { return true; }
};

// The analysis managers the plugin's own nested pipelines run against. They are
// registered through the PassBuilder the surrounding pipeline was built with,
// which is what makes another plugin's pipeline name (Enzyme's "enzyme")
// resolvable from here; keeping them local means nothing the outer pipeline
// cached is handed to a pass that runs out of order.
struct LocalAnalysisManagers {
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  explicit LocalAnalysisManagers(PassBuilder &PB) {
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
  }
};

// The late run of the site handler, at OptimizerLast: by now Enzyme has
// lowered the derivative request the deferred body was made of, so the body
// carries the derivative arithmetic and is profiled and solved like any other
// site. A module with no deferred site leaves here without touching anything,
// which is what keeps every other build byte for byte what it was.
struct LatePass : PassParent<LatePass> {
  PassBuilder *PB;
  OptimizationLevel Level;

  LatePass(PassBuilder *PB, OptimizationLevel Level) : PB(PB), Level(Level) {}

  // The canonical prefix, applied to the deferred bodies only. At
  // OptimizerEarly the same passes run over the whole module because the rest
  // of the pipeline follows them; here nothing follows, so a module-wide run
  // would be the only thing standing between a deferred-site build and the
  // codegen it would otherwise have had.
  void canonicalizeBodies(Module &M, ArrayRef<Function *> bodies,
                          LocalAnalysisManagers &AMs) {
    for (Function *F : bodies) {
      unsigned folded = foldEnzymeGenerated(*F);
      if (flags::Print && folded)
        llvm::errs() << "[poseidon] " << F->getName() << ": folded " << folded
                     << " Enzyme-generated call(s) into the deferred body\n";
    }
    FunctionPassManager Pre;
    if (Level != OptimizationLevel::O0) {
      Pre.addPass(Float2IntPass());
      Pre.addPass(LowerConstantIntrinsicsPass());
      LoopPassManager LPM;
      LPM.addPass(LoopRotatePass(/*EnableHeaderDuplication=*/true, false));
      LPM.addPass(LoopDeletionPass());
      LPM.addPass(LoopFullUnrollPass());
      Pre.addPass(createFunctionToLoopPassAdaptor(std::move(LPM)));
    }
    for (Function *F : bodies)
      Pre.run(*F, AMs.FAM);

    ModulePassManager Inline;
    Inline.addPass(AlwaysInlinerPass());
    Inline.run(M, AMs.MAM);

    FunctionPassManager Post;
    Post.addPass(GVNPass());
    Post.addPass(SROAPass(SROAOptions::PreserveCFG));
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
      report_fatal_error(Twine("Poseidon: a deferred site needs a second "
                               "Enzyme run and the Enzyme pass pipeline is "
                               "not registered (") +
                         msg +
                         "). Add -fpass-plugin=<ClangEnzyme>.so after the "
                         "Poseidon plugin.");
    }
    MPM.run(M, AMs.MAM);
  }

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    SmallVector<CallInst *, 4> markers;
    collectDeferredMarkers(M, markers);
    if (markers.empty())
      return PreservedAnalyses::all();

    LocalAnalysisManagers AMs(*PB);

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
    // The canonical form of a deferred site is taken here, in both phases, so
    // that the slots the profile was numbered against are the ones the solve
    // sees.
    canonicalizeBodies(M, bodies, AMs);

    markers.clear();
    collectDeferredMarkers(M, markers);

    bool instrumented = false;
    if (flags::ProfileGenerate) {
      DenseMap<Function *, Function *> clones;
      unsigned nextSiteId = getNextSiteId(M);
      for (CallInst *CI : markers)
        instrumented |= profileSite(CI, clones, nextSiteId);
      setNextSiteId(M, nextSiteId);
      if (instrumented)
        injectBlockDimProbes(M);
      markers.clear();
      collectDeferredMarkers(M, markers);
    }

    // Whatever is left is lowered exactly as the early run lowers a site: the
    // solve in profile-use mode, a call of the original body otherwise.
    SmallVector<CallInst *, 4> calls;
    if (!markers.empty())
      lowerMarkers(markers, calls);
    solveDeferredSites(M, AMs.FAM);
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

struct FinalizePass : PassParent<FinalizePass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    for (Function &F : M)
      if (F.isDeclaration() && !F.use_empty() &&
          F.getName().starts_with("__enzyme_autodiff_poseidon"))
        report_fatal_error(
            "Poseidon: profile generation emitted a derivative request that "
            "nothing lowered. Add -fpass-plugin=<ClangEnzyme>.so after the "
            "Poseidon plugin.");
    finalizeModule(M);
    return PreservedAnalyses::none();
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
    OptimizePM.addPass(Float2IntPass());
    OptimizePM.addPass(LowerConstantIntrinsicsPass());
    LoopPassManager LPM;
    LPM.addPass(LoopRotatePass(/*EnableHeaderDuplication=*/true, false));
    LPM.addPass(LoopDeletionPass());
    LPM.addPass(LoopFullUnrollPass());
    OptimizePM.addPass(createFunctionToLoopPassAdaptor(std::move(LPM)));
    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(OptimizePM)));
  }
  MPM.addPass(AlwaysInlinerPass());
  FunctionPassManager FPM;
  FPM.addPass(GVNPass());
  FPM.addPass(SROAPass(SROAOptions::PreserveCFG));
  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
}

} // namespace

void registerPasses(PassBuilder &PB) {
  auto loadHostStub = [](ModulePassManager &MPM, OptimizationLevel) {
    // Host-side GEMM dispatch must rewrite the launch stub BEFORE inlining, so
    // it runs at PipelineStart. No-op unless -poseidon-ozaki-host-dispatch is
    // set and the module is the host (non-NVPTX) module.
    MPM.addPass(HostStubPass());
  };
  PB.registerPipelineStartEPCallback(loadHostStub);

  auto loadPoseidon = [](ModulePassManager &MPM, OptimizationLevel Level) {
    addCanonicalPrefix(MPM, Level);
    MPM.addPass(OptimizePass());
    // Fold the outlined site bodies and the materialized clones back into
    // their kernels here; nothing later in the pipeline inlines, and the
    // AlwaysInlinerPass the Enzyme plugin adds is another plugin's business.
    MPM.addPass(AlwaysInlinerPass());
  };
  // The late run owns the sites the early run deferred: the canonical prefix,
  // the site handler, the always-inliner and, in profile-generate mode, a
  // second Enzyme run, all inside LatePass because they must not happen at all
  // in a module that deferred nothing. FinalizePass's "nothing lowered this
  // request" check therefore runs after that second Enzyme run.
  PassBuilder *PBp = &PB;
  auto loadFinalize = [PBp](ModulePassManager &MPM, OptimizationLevel Level) {
    MPM.addPass(LatePass(PBp, Level));
    MPM.addPass(FinalizePass());
  };

#if LLVM_VERSION_MAJOR >= 20
  PB.registerOptimizerEarlyEPCallback(
      [loadPoseidon](ModulePassManager &MPM, OptimizationLevel Level,
                     ThinOrFullLTOPhase) { loadPoseidon(MPM, Level); });
  PB.registerOptimizerLastEPCallback(
      [loadFinalize](ModulePassManager &MPM, OptimizationLevel Level,
                     ThinOrFullLTOPhase) { loadFinalize(MPM, Level); });
#else
  PB.registerOptimizerEarlyEPCallback(loadPoseidon);
  PB.registerOptimizerLastEPCallback(loadFinalize);
#endif
  PB.registerFullLinkTimeOptimizationEarlyEPCallback(
      [loadHostStub, loadPoseidon](ModulePassManager &MPM,
                                   OptimizationLevel Level) {
        loadHostStub(MPM, Level);
        loadPoseidon(MPM, Level);
      });
  PB.registerFullLinkTimeOptimizationLastEPCallback(loadFinalize);

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
