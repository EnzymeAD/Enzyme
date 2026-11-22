//===- HostSites.cpp - host functions and calls that are sites ------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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
#include <optional>

using namespace llvm;

namespace poseidon {

// Whether `F` is the call operator of a lambda. A closure type demangles as
// 'lambda'(...), or as $_<n> where clang needs no mangling number for it (a
// lambda in a function of a host-only compile).
static bool isLambdaCallOperator(const Function &F) {
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

// Host sites: a host function carrying POSEIDON_OPTIMIZE.

static bool isHostScalar(Type *T) {
  return T->isIntegerTy() || T->isHalfTy() || T->isBFloatTy() ||
         T->isFloatTy() || T->isDoubleTy();
}

// Why `F` cannot be a host site, empty if it can. The profiled clone is
// differentiated with one activity per parameter, which a pointer, a
// floating-point scalar and an integer have, and its value comes back only as
// one of the latter two.
static std::string hostSiteRefusal(Function &F) {
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
// out. A target the caller takes as a template argument gives it one
// instantiation, and one name, per target, while the profile of a call does
// not depend on its target. In an Itanium name a target is a literal, L, the
// type, the value, E: a floating-point literal spells the hex of its bits, and
// a 64-bit unsigned integer that carries the bits of the double, the form a
// C++17 program uses, spells them in decimal. A spelling runs from its L to its
// E, so it matches whole literals only; it becomes the external name `tau`.
static std::string targetFreeName(StringRef name, ArrayRef<double> targets) {
  std::string out = name.str();
  for (double t : targets) {
    const uint64_t bits = bit_cast<uint64_t>(t);
    SmallVector<std::string, 4> spellings{
        "Ld" + utohexstr(bits, /*LowerCase=*/true, 16) + "E",
        "Lm" + utostr(bits) + "E", "Ly" + utostr(bits) + "E"};
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
static void outlineHostFunction(Function &F, std::optional<double> tau) {
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

// The call that calls through `H`, a call of an identity, or null.
static CallBase *callThrough(CallInst *H) {
  CallBase *call = nullptr;
  for (User *U : H->users())
    if (auto *CB = dyn_cast<CallBase>(U); CB && CB->getCalledOperand() == H)
      call = CB;
  return call;
}

// The accuracy target the identity call `H` carries for `call`, 0 when it
// carries none; none, with an error at `call`, when it is not a positive
// finite constant.
static std::optional<double> identityTarget(CallInst *H, CallBase *call,
                                            StringRef name) {
  if (H->arg_size() < 2)
    return 0.0;
  auto *C = dyn_cast<ConstantFP>(H->getArgOperand(1));
  if (!C) {
    fail(call, "the accuracy target of __poseidon_optimize_tau for " + name +
                   " must be a compile-time constant (a literal, a macro or "
                   "a constant expression)");
    return std::nullopt;
  }
  double tau = C->getValueAPF().convertToDouble();
  if (!(tau > 0.0) || !std::isfinite(tau)) {
    fail(call, "the accuracy target " + shortDouble(tau) + " of " + name +
                   " is not a positive finite number");
    return std::nullopt;
  }
  return tau;
}

static void reportAnnotated(CallBase *call, StringRef name) {
  fail(call, name + " carries POSEIDON_OPTIMIZE, so every call of it is "
                    "already a site; remove this __poseidon_optimize or the "
                    "annotation");
}

// `call`, a call of `fn` with an explicit list, made the marker call that
// means the same, with `tau`, unless 0, as its poseidon_tau. The call's value
// is fn's, in the profiling run as well.
static CallInst *asMarker(CallBase *call, Function *fn, double tau) {
  LLVMContext &C = call->getContext();
  SmallVector<Value *, 16> args{fn};
  if (tau > 0.0) {
    args.push_back(activityArg(C, "poseidon_tau"));
    args.push_back(ConstantFP::get(Type::getDoubleTy(C), tau));
  }
  if (!call->getType()->isVoidTy())
    args.push_back(activityArg(C, "enzyme_primal_return"));
  const unsigned before = args.size();
  args.append(call->arg_begin(), call->arg_end());
  auto *FT = FunctionType::get(call->getType(), {PointerType::getUnqual(C)},
                               /*isVarArg=*/true);
  FunctionCallee marker =
      call->getModule()->getOrInsertFunction(kMarkerName, FT);
  IRBuilder<> Builder(call);
  CallInst *CI = Builder.CreateCall(FT, marker.getCallee(), args);
  CI->setDebugLoc(call->getDebugLoc());
  AttributeList old = call->getAttributes();
  SmallVector<AttributeSet, 16> params(before);
  for (unsigned i = 0; i < call->arg_size(); ++i)
    params.push_back(old.getParamAttrs(i));
  CI->setAttributes(
      AttributeList::get(C, old.getFnAttrs(), old.getRetAttrs(), params));
  call->replaceAllUsesWith(CI);
  call->eraseFromParent();
  return CI;
}

// `marker`, a marker call of `fn`, made the call of fn with the primal values
// of its arguments, or erased when its list cannot be read.
static void callAsWritten(CallInst *marker, Function *fn) {
  if (std::optional<MarkerArgs> parsed = parseMarker(marker, fn)) {
    IRBuilder<> Builder(marker);
    CallInst *direct =
        Builder.CreateCall(fn->getFunctionType(), fn, parsed->primalArgs);
    direct->setCallingConv(fn->getCallingConv());
    direct->setDebugLoc(marker->getDebugLoc());
    marker->replaceAllUsesWith(direct);
  } else if (!marker->use_empty()) {
    marker->replaceAllUsesWith(PoisonValue::get(marker->getType()));
  }
  marker->eraseFromParent();
}

// The calls of the module's functions named `names`, in module order.
static SmallVector<CallInst *, 4> callsOfAny(Module &M,
                                             ArrayRef<StringRef> names) {
  SmallVector<Function *, 2> callees;
  for (StringRef name : names)
    if (Function *F = M.getFunction(name))
      callees.push_back(F);
  SmallVector<CallInst *, 4> out;
  if (callees.empty())
    return out;
  for (Function &F : M)
    for (Instruction &I : instructions(F))
      if (auto *CI = dyn_cast<CallInst>(&I))
        if (is_contained(callees, CI->getCalledFunction()))
          out.push_back(CI);
  return out;
}

// __poseidon_optimize with an explicit list: the call through the explicit
// identity is the marker call over fn, and the marker's lowering makes it the
// site. The host half of a CUDA or HIP compilation holds no site, so there the
// call runs fn with the primal values of its arguments.
static bool lowerExplicitCalls(Module &M) {
  SmallVector<CallInst *, 4> identities =
      callsOfAny(M, {kExplicitIdentity, kExplicitIdentityTau});
  if (identities.empty())
    return false;
  const bool hostHalf = isOffloadHostHalf(M);
  auto annotated = annotatedFunctions(M);
  for (CallInst *H : identities) {
    auto *fn = dyn_cast<Function>(H->getArgOperand(0)->stripPointerCasts());
    CallBase *call = callThrough(H);
    if (!fn || !call || !H->hasOneUse()) {
      fail(H, "__poseidon_optimize needs a function name and a call of it");
      H->replaceAllUsesWith(PoisonValue::get(H->getType()));
      H->eraseFromParent();
      continue;
    }
    const std::string name = displayName(*fn);
    std::optional<double> tau = identityTarget(H, call, name);
    // Whatever the outcome, the call stops going through the identity, which
    // nothing defines.
    call->setCalledOperand(fn);
    H->eraseFromParent();
    if (!tau || !readArgumentList(call, *fn, Identity::Explicit))
      continue;
    if (*tau > 0.0 && llvm::any_of(call->args(), [](Value *A) {
          return isActivityMarker(A, "poseidon_tau");
        })) {
      fail(call, "this call of " + name +
                     " gives its accuracy target twice, as the first "
                     "argument of __poseidon_optimize_tau and after "
                     "poseidon_tau; give it once");
      continue;
    }
    if (annotated.count(fn)) {
      reportAnnotated(call, name);
      continue;
    }
    CallInst *marker = asMarker(call, fn, *tau);
    if (hostHalf)
      callAsWritten(marker, fn);
  }
  return true;
}

// A marker call whose list is automatic and passes a pointer is the call form
// of __poseidon_optimize, through the plain identity, so that the pass makes
// its shadows as it makes that form's. One whose list passes no pointer needs
// no shadow and stays the marker call it is, as does one with an explicit
// list. In the host half of a CUDA or HIP compilation, which holds no site,
// every marker call runs fn with the primal values of its arguments.
static bool routeAutomaticMarkers(Module &M) {
  SmallVector<CallInst *, 4> markers;
  collectMarkerCalls(M, markers);
  const bool hostHalf = isOffloadHostHalf(M);
  bool changed = false;
  for (CallInst *CI : markers) {
    Function *fn = markerTargetOrNull(CI);
    if (!fn)
      continue;
    std::optional<ArgumentList> list =
        readArgumentList(CI, *fn, Identity::None);
    if (list == ArgumentList::Explicit) {
      if (hostHalf) {
        callAsWritten(CI, fn);
        changed = true;
      }
      continue;
    }
    SmallVector<Value *, 8> args(drop_begin(CI->args()));
    if (list && !hostHalf && llvm::none_of(args, [](Value *A) {
          return A->getType()->isPointerTy();
        }))
      continue;
    changed = true;
    auto types = map_range(args, [](Value *A) { return A->getType(); });
    if (list && (CI->hasStructRetAttr() ||
                 !llvm::equal(fn->getFunctionType()->params(), types))) {
      fail(CI, "this call passes " + displayName(*fn) +
                   " arguments of other types than it takes, and none of "
                   "them is an activity marker; pass its own arguments");
      list = std::nullopt;
    }
    if (!list) {
      if (!CI->use_empty())
        CI->replaceAllUsesWith(PoisonValue::get(CI->getType()));
      CI->eraseFromParent();
      continue;
    }
    IRBuilder<> Builder(CI);
    Value *callee = fn;
    if (!hostHalf) {
      LLVMContext &C = M.getContext();
      Type *Ptr = PointerType::getUnqual(C);
      FunctionCallee identity = M.getOrInsertFunction(
          kPlainIdentity, FunctionType::get(Ptr, {Ptr}, false));
      callee = Builder.CreateCall(identity, {fn});
      cast<CallInst>(callee)->setDebugLoc(CI->getDebugLoc());
    }
    CallInst *call = Builder.CreateCall(fn->getFunctionType(), callee, args);
    call->setCallingConv(fn->getCallingConv());
    call->setDebugLoc(CI->getDebugLoc());
    CI->replaceAllUsesWith(call);
    CI->eraseFromParent();
  }
  return changed;
}

// __poseidon_optimize: the call through the plain identity becomes a call of
// a copy of the function made for that call alone, which is then a host site
// like an outlined body. The copy is named after the caller, without the
// targets (targetFreeName), and the call's position among the caller's per-call
// sites, taken before inlining, so that the generate and use compiles name it
// alike at any -O level and for any target. The host half of a CUDA or HIP
// compilation holds no site: the compilation has one device and one cost
// model, and the call of a host-device function is the device half's site.
// There the call through the identity is left for the inliner to fold. The
// calls through the explicit identity and the marker calls are read first
// (lowerExplicitCalls, routeAutomaticMarkers).
bool lowerPerCallSites(const Options &O, Module &M) {
  // The targets of the explicit calls are left out of their callers' names
  // too, so they are read before those calls become marker calls.
  DenseMap<Function *, SmallVector<double, 2>> targets;
  for (CallInst *H : callsOfAny(M, {kPlainIdentityTau, kExplicitIdentityTau}))
    if (H->arg_size() > 1)
      if (auto *C = dyn_cast<ConstantFP>(H->getArgOperand(1)))
        targets[H->getFunction()].push_back(C->getValueAPF().convertToDouble());
  bool changed = lowerExplicitCalls(M);
  changed |= routeAutomaticMarkers(M);
  if (isOffloadHostHalf(M))
    return changed;
  SmallVector<CallInst *, 4> helpers =
      callsOfAny(M, {kPlainIdentity, kPlainIdentityTau});
  if (helpers.empty())
    return changed;
  const bool active = O.Profiling.profileAction();
  auto annotated = annotatedFunctions(M);
  // Two callers whose names differ in targets alone number their calls on
  // from each other, so that they stay two sites.
  StringMap<unsigned> ordinal;

  for (CallInst *H : helpers) {
    Function *caller = H->getFunction();
    const std::string callerKey =
        targetFreeName(caller->getName(), targets.lookup(caller));
    unsigned k = ordinal[callerKey]++;
    auto *fn = dyn_cast<Function>(H->getArgOperand(0)->stripPointerCasts());
    CallBase *call = callThrough(H);
    // Whatever the outcome, the call stops going through the helper, so a
    // later run of the pass does not see it again.
    auto asWritten = [&](CallBase *site) {
      if (site && fn && site->getFunctionType() == fn->getFunctionType())
        site->setCalledOperand(fn);
      if (H->use_empty())
        H->eraseFromParent();
    };
    if (!fn || !call || !H->hasOneUse()) {
      if (active)
        fail(H, "__poseidon_optimize needs a function name and a call of it");
      else
        warn(H, "__poseidon_optimize needs a function name and a call of it; "
                "the call runs as written");
      asWritten(call);
      continue;
    }
    const std::string name = displayName(*fn);
    std::optional<double> tau = identityTarget(H, call, name);
    if (!tau || !readArgumentList(call, *fn, Identity::Plain)) {
      asWritten(call);
      continue;
    }
    if (annotated.count(fn)) {
      reportAnnotated(call, name);
      asWritten(call);
      continue;
    }
    std::string why = fn->isDeclaration()
                          ? "has no body in this translation unit"
                          : hostSiteRefusal(*fn);
    if (call->getFunctionType() != fn->getFunctionType())
      why = "is called with another signature";
    if (!why.empty() && active)
      fail(call, name + " " + why + ", so this call cannot be a Poseidon site");
    else if (!why.empty())
      warn(call, name + " " + why +
                     ", so this call is not a Poseidon site. The call runs as "
                     "written.");
    else if (!active)
      warn(call, "a Poseidon site was compiled without "
                 "-poseidon-profile-generate or -poseidon-profile-use=<dir>. "
                 "The call runs as written.");
    if (!why.empty() || !active) {
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
    B->addFnAttr(kHostBody, *tau > 0.0 ? exactDouble(*tau) : "");
    appendToCompilerUsed(M, {B});
    call->setCalledOperand(B);
    H->eraseFromParent();
    if (O.Diagnostics.Print && *tau > 0.0)
      llvm::errs() << "[poseidon] site " << name << " at its call in "
                   << displayName(*caller) << ": accuracy target "
                   << shortDouble(*tau) << '\n';
  }
  return true;
}

// As lowerPerCallSites, the host half of a CUDA or HIP compilation holds no
// site: a function carrying POSEIDON_OPTIMIZE there compiles as written.
bool outlineHostSites(const Options &O, Module &M) {
  if (Target::of(M).isDevice() || isOffloadHostHalf(M))
    return false;
  auto annotated = annotatedFunctions(M);
  if (annotated.empty())
    return false;
  const bool active = O.Profiling.profileAction();

  SmallVector<CallInst *, 4> markers;
  collectMarkerCalls(M, markers);

  bool changed = false;
  for (auto &[F, tau] : annotated) {
    if (isLaunchStub(*F) || F->hasFnAttribute(kSiteLowered))
      continue;
    // An annotated lambda marks the kernels that call it, which only a CUDA
    // or HIP compilation has. Its closure is no pointer the profiler can
    // shadow, so a host-only build leaves it alone.
    if (isLambdaCallOperator(*F)) {
      diagnoseSite(*F,
                   displayName(*F) +
                       " is a lambda; POSEIDON_OPTIMIZE on a lambda marks the "
                       "GPU kernels that call it, and this compilation has "
                       "none. Emitting the original computation.",
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
      reportAnnotated(*both, displayName(*F));
      F->addFnAttr(kNotASite);
      continue;
    }
    std::string why = hostSiteRefusal(*F);
    if (!why.empty()) {
      diagnoseSite(
          *F, displayName(*F) + " " + why + ", so it cannot be a Poseidon site",
          DS_Error);
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
  if (!value.empty() && value.getAsDouble(tau))
    report_fatal_error("Poseidon: the body " + B.getName() +
                       " carries the accuracy target '" + value +
                       "', which is not a number");
  B.removeFnAttr(kHostBody);
  removeFromUsedLists(*B.getParent(), [&](Constant *C) {
    return C->stripPointerCasts() == &B;
  });
  return tau;
}

} // namespace poseidon
