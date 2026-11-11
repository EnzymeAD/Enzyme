#include "HostSites.h"

#include "Annotations.h"
#include "Core/Options.h"
#include "Markers.h"
#include "Targets/OffloadRuntime.h"
#include "Targets/Target.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/bit.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include <cmath>
#include <cstdio>
#include <optional>

using namespace llvm;

namespace poseidon {

// A double as a function attribute value that reads back to the same bits.
std::string exactDouble(double v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.17g", v);
  return buf;
}

namespace {

// The call operator of a lambda: what an annotated mfem::forall body is on the
// host. A closure type demangles as 'lambda'(...), or as $_<n> where clang
// needs no mangling number for it (a lambda in a function of a host-only
// compile).
bool isLambdaCallOperator(const Function &F) {
  // The demangler's nodes point into the name it was given.
  std::string mangled = F.getName().str();
  ItaniumPartialDemangler D;
  if (D.partialDemangle(mangled.c_str()) || !D.isFunction())
    return false;
  char *base = D.getFunctionBaseName(nullptr, nullptr);
  char *context = D.getFunctionDeclContextName(nullptr, nullptr);
  bool lambda = base && context && StringRef(base) == "operator()" &&
                (StringRef(context).contains("'lambda") ||
                 StringRef(context).rsplit("::").second.starts_with("$_"));
  std::free(base);
  std::free(context);
  return lambda;
}

// --------------------------------------------------------------------------
// Host sites: a host function carrying POSEIDON_OPTIMIZE
// --------------------------------------------------------------------------

bool isHostScalar(Type *T) {
  return T->isIntegerTy() || T->isHalfTy() || T->isBFloatTy() ||
         T->isFloatTy() || T->isDoubleTy();
}

// Why `F` cannot be a host site, empty if it can. The profiled clone is
// differentiated with one activity per parameter, which a pointer, a
// floating-point scalar and an integer have, and its value comes back only as
// one of the latter two.
std::string hostSiteRefusal(Function &F) {
  if (F.isVarArg())
    return "is variadic";
  Type *R = F.getReturnType();
  if (!R->isVoidTy() && !isHostScalar(R))
    return "returns a value that is not a floating-point or integer scalar";
  for (Argument &A : F.args()) {
    if (A.hasStructRetAttr())
      return "returns its value through memory (sret)";
    bool ok = A.getType()->isPointerTy()
                  ? !A.hasPassPointeeByValueCopyAttr() && !A.hasByRefAttr()
                  : isHostScalar(A.getType());
    if (!ok)
      return ("takes parameter " + Twine(A.getArgNo() + 1) +
              " as an aggregate or vector value")
          .str();
  }
  for (Instruction &I : instructions(F))
    if (auto *CB = dyn_cast<CallBase>(&I))
      if (CB->getCalledOperand()->stripPointerCasts() == &F)
        return "calls itself";
  return "";
}

// The name of a caller with the accuracy targets of its per-call sites left
// out. A target the caller takes as a template argument (dFEM's Numerics tag)
// gives it one instantiation, and one name, per target, while the profile of a
// call does not depend on its target. In an Itanium name a target is a
// floating-point literal, L, the type, the hex of its bits, E; it becomes the
// external name `tau`.
std::string targetFreeName(StringRef name, ArrayRef<double> targets) {
  std::string out = name.str();
  for (double t : targets) {
    SmallVector<std::string, 2> spellings{
        "Ld" + utohexstr(bit_cast<uint64_t>(t), /*LowerCase=*/true, 16) + "E"};
    if (float f = (float)t; (double)f == t)
      spellings.push_back(
          "Lf" + utohexstr(bit_cast<uint32_t>(f), /*LowerCase=*/true, 8) + "E");
    for (const std::string &s : spellings)
      for (size_t at; (at = out.find(s)) != std::string::npos;)
        out.replace(at, s.size(), "L_Z3tauE");
  }
  return out;
}

// Make `F` a wrapper around its own body, as outlineKernelBody does for a
// kernel, but before the inliner runs: every call of F, direct, indirect or
// from another translation unit, reaches the body through F, and a caller the
// inliner puts F into calls the body itself. Until the site is lowered the
// body is kept from being inlined and from interprocedural rewriting, so that
// it keeps the signature and the code F was written with.
void outlineHostFunction(Function &F, std::optional<double> tau) {
  Module &M = *F.getParent();
  Function *B =
      Function::Create(F.getFunctionType(), GlobalValue::InternalLinkage,
                       F.getName() + "_poseidon_body", &M);
  B->setAttributes(F.getAttributes());
  B->setCallingConv(F.getCallingConv());
  if (F.hasPersonalityFn())
    B->setPersonalityFn(F.getPersonalityFn());
  B->removeFnAttr(Attribute::AlwaysInline);
  B->removeFnAttr(Attribute::InlineHint);
  B->addFnAttr(Attribute::NoInline);
  B->addFnAttr(kHostBody, tau ? exactDouble(*tau) : "");
  B->setSubprogram(F.getSubprogram());
  F.setSubprogram(nullptr);
  B->splice(B->begin(), &F);
  for (auto i = F.arg_begin(), j = B->arg_begin(); i != F.arg_end(); ++i, ++j) {
    j->setName(i->getName());
    i->replaceAllUsesWith(&*j);
  }

  IRBuilder<> Builder(BasicBlock::Create(M.getContext(), "entry", &F));
  SmallVector<Value *, 8> args;
  for (Argument &A : F.args())
    args.push_back(&A);
  CallInst *call = Builder.CreateCall(B, args);
  call->setCallingConv(B->getCallingConv());
  Instruction *ret = F.getReturnType()->isVoidTy() ? Builder.CreateRetVoid()
                                                   : Builder.CreateRet(call);
  for (BasicBlock &BB : *B)
    if (auto *R = dyn_cast<ReturnInst>(BB.getTerminator())) {
      ret->setDebugLoc(R->getDebugLoc());
      break;
    }
  F.addFnAttr(kSiteLowered);
  F.addFnAttr(kHostWrapper);
  appendToCompilerUsed(M, {B});
}

} // namespace

// __poseidon_optimize: the call through the identity helper becomes a call of
// a copy of the function made for that call alone, which is then a host site
// like an outlined body. The copy is named after the caller, without the
// targets (targetFreeName), and the call's position among the caller's per-call
// sites, taken here, before inlining, so that the generate and use compiles
// name it alike at any -O level and for any target: its profile, DP table and
// rewritten copy are that call's own. In device code the copy becomes a marker
// site instead (lowerDeviceCallSites).
bool lowerPerCallSites(const Options &O, Module &M) {
  Function *site = M.getFunction("__poseidon_site");
  Function *siteTau = M.getFunction("__poseidon_site_tau");
  if (!site && !siteTau)
    return false;
  SmallVector<CallInst *, 4> helpers;
  for (Function &F : M)
    for (Instruction &I : instructions(F))
      if (auto *CI = dyn_cast<CallInst>(&I))
        if (Function *callee = CI->getCalledFunction();
            callee && (callee == site || callee == siteTau))
          helpers.push_back(CI);
  if (helpers.empty())
    return false;
  const bool active = O.Profiling.profileAction();
  auto annotated = annotatedFunctions(M);
  DenseMap<Function *, SmallVector<double, 2>> targets;
  for (CallInst *H : helpers)
    if (H->arg_size() > 1)
      if (auto *C = dyn_cast<ConstantFP>(H->getArgOperand(1)))
        targets[H->getFunction()].push_back(C->getValueAPF().convertToDouble());
  // Two callers whose names differ in targets alone number their calls on
  // from each other, so that they stay two sites.
  StringMap<unsigned> ordinal;

  for (CallInst *H : helpers) {
    Function *caller = H->getFunction();
    const std::string callerKey =
        targetFreeName(caller->getName(), targets.lookup(caller));
    unsigned k = ordinal[callerKey]++;
    auto *fn = dyn_cast<Function>(H->getArgOperand(0)->stripPointerCasts());
    CallBase *call = nullptr;
    for (User *U : H->users())
      if (auto *CB = dyn_cast<CallBase>(U); CB && CB->getCalledOperand() == H)
        call = CB;
    // Whatever the outcome, the call no longer goes through the helper, so a
    // later run of the pass does not see it again.
    auto asWritten = [&](CallBase *site) {
      if (site && fn && site->getFunctionType() == fn->getFunctionType())
        site->setCalledOperand(fn);
      if (H->use_empty())
        H->eraseFromParent();
    };
    if (!fn || !call || !H->hasOneUse()) {
      warn(H, "__poseidon_optimize needs a function name and a call of it; "
              "the call runs as written");
      asWritten(call);
      continue;
    }
    const std::string name = displayName(*fn);
    const bool hasTau = H->arg_size() > 1;
    double tau = 0.0;
    if (hasTau) {
      auto *C = dyn_cast<ConstantFP>(H->getArgOperand(1));
      if (!C) {
        fail(call, "the accuracy target of __poseidon_optimize_tau for " +
                       name +
                       " must be a compile-time constant (a literal, a macro "
                       "or a constant expression)");
        asWritten(call);
        continue;
      }
      tau = C->getValueAPF().convertToDouble();
      if (!(tau > 0.0) || !std::isfinite(tau)) {
        fail(call, "the accuracy target " + shortDouble(tau) + " of " + name +
                       " is not a positive finite number");
        asWritten(call);
        continue;
      }
    }
    if (annotated.count(fn)) {
      fail(call, name +
                     " carries POSEIDON_OPTIMIZE, so every call of it is "
                     "already a site; remove this __poseidon_optimize or the "
                     "annotation");
      asWritten(call);
      continue;
    }
    std::string why = fn->isDeclaration()
                          ? "has no body in this translation unit"
                          : hostSiteRefusal(*fn);
    if (call->getFunctionType() != fn->getFunctionType())
      why = "is called with another signature";
    if (!why.empty() || !active) {
      warn(call, !why.empty()
                     ? name + " " + why +
                           ", so this call is not a Poseidon site. The call "
                           "runs as written."
                     : std::string("a Poseidon site was compiled without "
                                   "-poseidon-profile-generate or "
                                   "-poseidon-profile-use=<dir>. The call "
                                   "runs as written."));
      asWritten(call);
      continue;
    }

    ValueToValueMapTy VMap;
    Function *B = CloneFunction(fn, VMap);
    B->setName(fn->getName() + ".poseidon." + callerKey + "." + Twine(k));
    B->setLinkage(GlobalValue::InternalLinkage);
    B->setVisibility(GlobalValue::DefaultVisibility);
    B->setComdat(nullptr);
    B->removeFnAttr(Attribute::AlwaysInline);
    B->removeFnAttr(Attribute::InlineHint);
    B->addFnAttr(Attribute::NoInline);
    B->addFnAttr(kHostBody, hasTau ? exactDouble(tau) : "");
    appendToCompilerUsed(M, {B});
    call->setCalledOperand(B);
    H->eraseFromParent();
    if (O.Diagnostics.Print && hasTau)
      llvm::errs() << "[poseidon] site " << name << " at its call in "
                   << displayName(*caller) << ": accuracy target "
                   << shortDouble(tau) << "\n";
  }
  return true;
}

bool outlineHostSites(const Options &O, Module &M) {
  if (Target::of(M).isDevice())
    return false;
  auto annotated = annotatedFunctions(M);
  if (annotated.empty())
    return false;
  const bool active = O.Profiling.profileAction();

  SmallVector<CallInst *, 4> markers;
  for (Function &G : M)
    for (Instruction &I : instructions(G))
      if (auto *CI = dyn_cast<CallInst>(&I))
        if (Function *callee = calleeOf(CI))
          if (isMarkerCall(callee->getName()))
            markers.push_back(CI);

  bool changed = false;
  for (auto &[F, tau] : annotated) {
    if (isLaunchStub(*F) || F->hasFnAttribute(kSiteLowered))
      continue;
    // An annotated mfem::forall body: its kernels are the sites, which the
    // device half of a CUDA compilation owns. A lambda's closure is no
    // pointer the profiler can shadow, so a host-only build leaves it alone.
    if (isLambdaCallOperator(*F)) {
      if (launchesKernels(M))
        F->addFnAttr(kNotASite);
      else
        diagnoseSite(*F,
                     displayName(*F) +
                         " is a lambda; POSEIDON_OPTIMIZE on a lambda marks "
                         "the GPU kernels that call it, and this compilation "
                         "has none. Emitting the original computation.",
                     DS_Warning);
      continue;
    }
    if (!active) {
      diagnoseSite(*F,
                   "a Poseidon site was compiled without "
                   "-poseidon-profile-generate or -poseidon-profile-use=<dir>. "
                   "Emitting the original computation.",
                   DS_Warning);
      continue;
    }
    auto both = llvm::find_if(
        markers, [F = F](CallInst *CI) { return markerTargetOrNull(CI) == F; });
    if (both != markers.end()) {
      fail(*both, displayName(*F) +
                      " carries POSEIDON_OPTIMIZE, so every call of it is "
                      "already a site; remove this __poseidon_fp_optimize "
                      "or the annotation");
      F->addFnAttr(kNotASite);
      continue;
    }
    std::string why = hostSiteRefusal(*F);
    if (!why.empty()) {
      diagnoseSite(*F,
                   displayName(*F) + " " + why +
                       ", so it is not a Poseidon site. Emitting the original "
                       "computation.",
                   DS_Warning);
      continue;
    }
    noteSiteTarget(O, *F, tau);
    outlineHostFunction(*F, tau);
    changed = true;
  }
  return changed;
}

SmallVector<Function *, 4> hostSiteBodies(Module &M) {
  SmallVector<Function *, 4> out;
  for (Function &F : M)
    if (!F.empty() && F.hasFnAttribute(kHostBody))
      out.push_back(&F);
  return out;
}

SmallVector<CallBase *, 4> callsOf(Function &F) {
  SmallVector<CallBase *, 4> out;
  for (User *U : F.users())
    if (auto *CB = dyn_cast<CallBase>(U); CB && CB->getCalledOperand() == &F)
      out.push_back(CB);
  return out;
}

// The site's accuracy target, and the body handed back to the optimizer: out
// of llvm.compiler.used and without the mark its clones would inherit.
double takeHostBody(Function &B) {
  double tau = 0.0;
  StringRef value = B.getFnAttribute(kHostBody).getValueAsString();
  if (!value.empty())
    value.getAsDouble(tau);
  B.removeFnAttr(kHostBody);
  removeFromUsedLists(*B.getParent(), [&](Constant *C) {
    return C->stripPointerCasts() == &B;
  });
  return tau;
}

} // namespace poseidon
