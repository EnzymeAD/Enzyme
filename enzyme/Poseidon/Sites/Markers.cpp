#include "Markers.h"

#include "Core/Options.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <optional>
#include <string>

using namespace llvm;

namespace poseidon {

bool isMarkerCall(StringRef calleeName) {
  return calleeName.contains("__poseidon_fp_optimize");
}

void warn(CallBase *CI, const Twine &message) {
  CI->getContext().diagnose(DiagnosticInfoUnsupported(
      *CI->getFunction(), "Poseidon: " + message.str(), CI->getDebugLoc(),
      DS_Warning));
}

void fail(CallBase *CI, const Twine &message) {
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

namespace {

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

} // namespace

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

// A copy of a deferred body as the pipeline left it, with the functions Enzyme
// generated for it, for a site the solve leaves alone. Nothing inlines after
// Enzyme in a plugin-off build, so the always-inline functions Enzyme's code
// calls (libdevice's __nv_*) stay calls there; in the copy those calls are
// noinline, so that the late run's always-inliners leave them too and a
// refused site is not optimized further than the code as written (the dFEM
// JVP's primal and tangent sqrt would merge into one).
Function *asLeftByPipeline(Function &body,
                           DenseMap<Function *, Function *> &copies) {
  if (Function *copy = copies.lookup(&body))
    return copy;
  ValueToValueMapTy VMap;
  Function *copy = CloneFunction(&body, VMap);
  copy->setName(body.getName() + ".left");
  // NVPTX gives only a local symbol a name PTX can spell.
  copy->setLinkage(GlobalValue::InternalLinkage);
  copy->removeFnAttr(kSiteDeferred);
  copies[&body] = copy;
  for (Instruction &I : instructions(*copy))
    if (auto *CB = dyn_cast<CallBase>(&I))
      if (Function *callee = CB->getCalledFunction()) {
        if (!callee->isDeclaration() &&
            isEnzymeGeneratedName(callee->getName()))
          CB->setCalledFunction(asLeftByPipeline(*callee, copies));
        if (callee->hasFnAttribute(Attribute::AlwaysInline))
          CB->addFnAttr(Attribute::NoInline);
      }
  return copy;
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
bool deferSite(const Options &O, CallInst *CI) {
  Function *F = markerTargetOrNull(CI);
  if (!F || F->empty())
    return false;
  if (F->hasFnAttribute(kSiteDeferred))
    return true;
  if (!callsEnzymeRequest(*F))
    return false;
  F->addFnAttr(kSiteDeferred);
  if (O.Diagnostics.Print)
    llvm::errs() << "[poseidon] " << F->getName()
                 << ": marked body still holds an __enzyme_* request; "
                    "deferred to the late run\n";
  return true;
}

void collectMarkerCalls(Module &M, SmallVectorImpl<CallInst *> &out) {
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    for (Instruction &I : instructions(F))
      if (auto *CI = dyn_cast<CallInst>(&I))
        if (Function *callee = calleeOf(CI))
          if (isMarkerCall(callee->getName()))
            out.push_back(CI);
  }
}

// The marker calls this module still owes the late run.
void collectDeferredMarkers(Module &M, SmallVectorImpl<CallInst *> &out) {
  SmallVector<CallInst *, 4> markers;
  collectMarkerCalls(M, markers);
  for (CallInst *CI : markers) {
    Function *target = markerTargetOrNull(CI);
    if (target && target->hasFnAttribute(kSiteDeferred))
      out.push_back(CI);
  }
}

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

namespace {

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

} // namespace

bool parseMarker(CallInst *CI, Function *F, MarkerArgs &out) {
  bool sret =
      CI->hasStructRetAttr() || F->hasParamAttribute(0, Attribute::StructRet);
  out.forward.assign(CI->arg_size(), true);

  for (size_t i = 1 + sret; i < CI->arg_size(); ++i) {
    Value *res = CI->getArgOperand(i);
    std::optional<bool> dup;
    std::optional<unsigned> madeShadow;
    bool skipArg = false;

    while (auto name = activityMarkerName(res)) {
      // Everything forwarded to the derivative request is spelled enzyme_*;
      // poseidon_* is Poseidon's own and is consumed here.
      if (!name->starts_with("enzyme_") && !name->starts_with("poseidon_"))
        break;
      if (*name == "enzyme_dup" || *name == "enzyme_dupnoneed") {
        dup = true;
      } else if (*name == "enzyme_out" || *name == "enzyme_const") {
        dup = false;
      } else if (*name == "poseidon_shadow") {
        madeShadow = i;
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
    if (madeShadow) {
      if (!F->getFunctionType()->getParamType(param)->isPointerTy()) {
        fail(CI, "poseidon_shadow on parameter " + Twine(param) + " of " +
                     F->getName() + ", which is not a pointer");
        return false;
      }
      out.madeShadows.push_back({*madeShadow, (unsigned)i, (unsigned)param});
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

  size_t expected = F->getFunctionType()->getNumParams() - sret;
  if (out.primalArgs.size() != expected) {
    fail(CI, "decoded " + Twine(out.primalArgs.size()) + " argument(s) for " +
                 F->getName() + ", which takes " + Twine(expected));
    return false;
  }
  return true;
}

} // namespace poseidon
