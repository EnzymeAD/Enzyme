//===- Site.cpp - the sites of a module -----------------------------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Site.h"
#include "Annotations.h"
#include "Enzyme/Utils.h"
#include "LaunchManifest.h"
#include "Markers.h"
#include "SiteIdentity.h"
#include "Support/Diagnostics.h"
#include "Support/Options.h"
#include "Support/Utils.h"
#include "Targets/OffloadRuntime.h"
#include "Targets/Target.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Regex.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include <cassert>

using namespace llvm;

namespace poseidon {

bool Site::deferred() const {
  return form == Form::Marker && fn && isDeferredBody(*fn);
}

static void collectMarkers(Module &M, SmallVectorImpl<Site> &out) {
  SmallVector<CallInst *, 4> calls;
  collectMarkerCalls(M, calls);
  for (CallInst *CI : calls)
    out.push_back({Site::Form::Marker, markerTargetOrNull(CI), CI});
}

/// The annotated kernels, then the kernels only -poseidon-kernels names, in
/// module order.
static void collectKernels(const Options &O, Module &M,
                           SmallVectorImpl<Site> &out) {
  const Target &T = Target::of(M);
  DenseMap<Function *, Site> found;
  for (auto &[fn, tau] : annotatedFunctions(M)) {
    if (fn->hasFnAttribute(kSiteLowered))
      continue;
    // A host function is a site of its own (outlineHostSites); a device
    // function makes the kernels that call it sites (propagateDeviceSites),
    // and what is left here no kernel of this translation unit calls.
    if (!T.isKernel(*fn)) {
      if (T.isDevice())
        warnSite(*fn, displayName(*fn) +
                          " is a device function no kernel of this "
                          "translation unit calls, so it is left as written; "
                          "on the device, put POSEIDON_OPTIMIZE on a kernel "
                          "or on a function a kernel calls");
      continue;
    }
    found.insert({fn, {Site::Form::Kernel, fn, nullptr, tau}});
  }

  if (!O.Profiling.Kernels.empty()) {
    // validateOptions refuses a pattern that is not a regular expression.
    bool all = O.Profiling.Kernels == "all";
    Regex re(O.Profiling.Kernels);
    for (Function &F : M)
      if (!F.isDeclaration() && !F.hasFnAttribute(kSiteLowered) &&
          T.isKernel(F) && (all || re.match(F.getName())))
        found.insert({&F,
                      {Site::Form::Kernel, &F, nullptr, std::nullopt,
                       /*byRegex=*/true}});
  }

  for (Function &F : M) {
    auto it = found.find(&F);
    if (it != found.end())
      out.push_back(it->second);
  }
}

static void collectHostBodies(Module &M, SmallVectorImpl<Site> &out) {
  for (Function &F : M) {
    if (F.empty() || !F.hasFnAttribute(kSiteBody))
      continue;
    std::optional<double> tau;
    StringRef value = F.getFnAttribute(kSiteBody).getValueAsString();
    if (!value.empty()) {
      double parsed = 0.0;
      [[maybe_unused]] const bool failed = value.getAsDouble(parsed);
      assert(!failed && "the pass writes a body's accuracy target as a number");
      tau = parsed;
    }
    out.push_back({Site::Form::HostBody, &F, nullptr, tau});
  }
}

SmallVector<Site, 4> collectDeferredSites(Module &M) {
  SmallVector<Site, 4> sites;
  collectMarkers(M, sites);
  erase_if(sites, [](const Site &site) { return !site.deferred(); });
  return sites;
}

SmallVector<Site, 4> collectSites(const Options &O, Module &M,
                                  ArrayRef<Site::Form> forms) {
  SmallVector<Site, 4> sites;
  for (Site::Form form : forms)
    switch (form) {
    case Site::Form::Marker:
      collectMarkers(M, sites);
      break;
    case Site::Form::Kernel:
      collectKernels(O, M, sites);
      break;
    case Site::Form::HostBody:
      collectHostBodies(M, sites);
      break;
    }
  return sites;
}

// Outlined sites: a site's computation moved into a body of its own that the
// function or kernel carrying the annotation calls.

static bool isHostScalar(Type *T) {
  return T->isIntegerTy() || T->isHalfTy() || T->isBFloatTy() ||
         T->isFloatTy() || T->isDoubleTy();
}

/// Why `F` cannot be a host site, empty if it can. The profiled clone is
/// differentiated with one activity per parameter, which a pointer, a
/// floating-point scalar and an integer have, and its value comes back only as
/// one of the latter two.
std::optional<std::string> hostSiteRefusal(Function &F) {
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
      if (getFunctionFromCall(CB) == &F)
        return "calls itself";
  return std::nullopt;
}

/// The body a site's call targets must not reach codegen as a call: under
/// -fcuda-rdc a call that is not folded back is a real ABI call, and the kernel
/// entry then pays its register and frame budget, which lowers occupancy or
/// makes a large block's launch fail. Marked on the solve path for both
/// annotation forms, so that a site the solve leaves unchanged is inlined
/// exactly like one it rewrote.
void markOutlineTransparent(Function *body) {
  body->removeFnAttr(Attribute::NoInline);
  body->removeFnAttr(Attribute::OptimizeNone);
  body->addFnAttr(Attribute::AlwaysInline);
}

/// Whether every use of `P`, a pointer to an aggregate, is one reroot can
/// rebuild over a pointer in another address space.
static bool reroots(const Value *P) {
  for (const Use &U : P->uses()) {
    const User *I = U.getUser();
    if (auto *GEP = dyn_cast<GetElementPtrInst>(I)) {
      if (U.getOperandNo() != GEP->getPointerOperandIndex() || !reroots(GEP))
        return false;
    } else if (isa<MemTransferInst>(I)) {
      if (U.getOperandNo() != 1)
        return false;
    } else if (!isa<LoadInst>(I) && !isa<AddrSpaceCastInst>(I)) {
      return false;
    }
  }
  return true;
}

/// The uses of `from` rebuilt over `to`, which points to the same aggregate
/// through the generic address space.
static void reroot(Value *from, Value *to) {
  for (Use &U : make_early_inc_range(from->uses())) {
    auto *I = cast<Instruction>(U.getUser());
    IRBuilder<> B(I);
    if (auto *GEP = dyn_cast<GetElementPtrInst>(I)) {
      SmallVector<Value *, 4> idx(GEP->indices());
      Value *N = B.CreateGEP(GEP->getSourceElementType(), to, idx, "",
                             GEP->getNoWrapFlags());
      N->takeName(GEP);
      reroot(GEP, N);
      GEP->eraseFromParent();
    } else if (auto *MT = dyn_cast<MemTransferInst>(I)) {
      CallInst *N = isa<MemMoveInst>(MT)
                        ? B.CreateMemMove(MT->getRawDest(), MT->getDestAlign(),
                                          to, MT->getSourceAlign(),
                                          MT->getLength(), MT->isVolatile())
                        : B.CreateMemCpy(MT->getRawDest(), MT->getDestAlign(),
                                         to, MT->getSourceAlign(),
                                         MT->getLength(), MT->isVolatile());
      N->copyMetadata(*MT);
      MT->eraseFromParent();
    } else if (isa<AddrSpaceCastInst>(I) && I->getType() == to->getType()) {
      I->replaceAllUsesWith(to);
      I->eraseFromParent();
    } else {
      U.set(to);
    }
  }
}

/// Moves the code of `F` into `B`, a new function of as many parameters, with
/// `rewire` making each use of a parameter of F one of B's, and makes F a
/// wrapper that calls B with what `passArg` makes of each of its parameters.
/// Returns that call.
static CallInst *
outlineInto(Function &F, Function &B,
            function_ref<void(Argument &From, Argument &To)> rewire,
            function_ref<Value *(IRBuilder<> &, Argument &)> passArg) {
  B.setSubprogram(F.getSubprogram());
  F.setSubprogram(nullptr);
  B.splice(B.begin(), &F);
  for (auto [From, To] : zip(F.args(), B.args())) {
    To.setName(From.getName());
    rewire(From, To);
  }
  IRBuilder<> Builder(BasicBlock::Create(F.getContext(), "entry", &F));
  SmallVector<Value *, 8> args;
  for (Argument &A : F.args())
    args.push_back(passArg(Builder, A));
  CallInst *call = Builder.CreateCall(&B, args);
  call->setCallingConv(B.getCallingConv());
  if (F.getReturnType()->isVoidTy())
    Builder.CreateRetVoid();
  else
    Builder.CreateRet(call);
  F.addFnAttr(kSiteLowered);
  // Folded back, the body's code still names the subprogram it took along;
  // without it again the wrapper compiles with no line table, and ptxas then
  // allocates a kernel differently.
  F.addFnAttr(kSiteWrapper);
  return call;
}

/// Make the kernel a wrapper around its own body so that both phases see the
/// structure they already handle: a function holding the annotated computation,
/// reached through one call whose arguments are the site's inputs. Returns that
/// call; an error, before any change, if the kernel cannot be a site.
Expected<CallInst *> outlineKernelBody(SiteRecords &T, Function &K,
                                       Function *&bodyOut) {
  if (K.isVarArg() || !K.getReturnType()->isVoidTy())
    return userError(
        K, "an annotated kernel must be a void, non-variadic function");
  Module &M = *K.getParent();
  // An AMDGPU kernel takes an aggregate by reference into its constant
  // argument segment; the body takes it through a generic pointer, which the
  // profiled kernel can point at a copy that holds the shadows.
  SmallVector<Type *, 8> params(K.getFunctionType()->params());
  for (Argument &A : K.args())
    if (A.hasByRefAttr()) {
      if (!reroots(&A))
        return userError(K, displayName(K) + " uses its aggregate parameter " +
                                Twine(A.getArgNo() + 1) +
                                " otherwise than by loading or copying from "
                                "it, so it cannot be a Poseidon site; copy the "
                                "parameter into a local object first");
      params[A.getArgNo()] = getInt8PtrTy(M.getContext());
    }
  // Named alike with and without -fgpu-rdc, which profile generation adds and
  // a solve need not: the profile is found by this name.
  Function *B = Function::Create(
      FunctionType::get(K.getReturnType(), params, false),
      GlobalValue::InternalLinkage,
      withoutExternalizedPostfix(K.getName()) + kBodySuffix, &M);
  B->setAttributes(K.getAttributes());
  noteSiteIdentity(T, *B, functionIdentity(K));
  // The body reads the kernel's own copy of a by-value aggregate. Left byval,
  // folding the body back would copy the aggregate a second time.
  for (Argument &A : B->args())
    if (A.hasByValAttr() || A.hasByRefAttr()) {
      A.removeAttr(Attribute::ByVal);
      A.removeAttr(Attribute::ByRef);
      A.addAttr(Attribute::get(M.getContext(), kByValueParam));
    }
  // The body is an ordinary device function; only the kernel stays an entry.
  B->setCallingConv(CallingConv::C);
  CallInst *call = outlineInto(
      K, *B,
      [](Argument &From, Argument &To) {
        if (From.getType() != To.getType())
          reroot(&From, &To);
        else
          From.replaceAllUsesWith(&To);
      },
      [&](IRBuilder<> &Builder, Argument &A) -> Value * {
        return A.hasByRefAttr() ? Builder.CreateAddrSpaceCast(
                                      &A, B->getArg(A.getArgNo())->getType())
                                : &A;
      });
  bodyOut = B;
  return call;
}

/// Make `F` a wrapper around its own body, as outlineKernelBody does for a
/// kernel, but before the inliner runs: every call of F, direct, indirect or
/// from another translation unit, reaches the body through F, and a caller the
/// inliner puts F into calls the body itself. Until the site is lowered the
/// body is kept from being inlined and from interprocedural rewriting, so that
/// it keeps the signature and the code F was written with.
void outlineHostFunction(SiteRecords &T, Function &F,
                         std::optional<double> tau) {
  Module &M = *F.getParent();
  Function *B =
      Function::Create(F.getFunctionType(), GlobalValue::InternalLinkage,
                       F.getName() + kBodySuffix, &M);
  B->setAttributes(F.getAttributes());
  B->setCallingConv(F.getCallingConv());
  if (F.hasPersonalityFn())
    B->setPersonalityFn(F.getPersonalityFn());
  B->removeFnAttr(Attribute::AlwaysInline);
  B->removeFnAttr(Attribute::InlineHint);
  B->addFnAttr(Attribute::NoInline);
  B->addFnAttr(kSiteBody, tau ? exactDouble(*tau) : "");
  noteSiteIdentity(T, *B, functionIdentity(F));
  CallInst *call = outlineInto(
      F, *B, [](Argument &From, Argument &To) { From.replaceAllUsesWith(&To); },
      [](IRBuilder<> &, Argument &A) -> Value * { return &A; });
  for (BasicBlock &BB : *B)
    if (auto *R = dyn_cast<ReturnInst>(BB.getTerminator())) {
      call->getNextNode()->setDebugLoc(R->getDebugLoc());
      break;
    }
  appendToCompilerUsed(M, {B});
}

SmallVector<CallBase *, 4> callsOf(Value &F) {
  SmallVector<CallBase *, 4> out;
  for (User *U : F.users())
    if (auto *CB = dyn_cast<CallBase>(U); CB && CB->getCalledOperand() == &F)
      out.push_back(CB);
  return out;
}

void releaseHostBody(Function &B) {
  B.removeFnAttr(kSiteBody);
  removeFromUsedLists(*B.getParent(), [&](Constant *C) {
    return C->stripPointerCasts() == &B;
  });
}

/// Undo outlineKernelBody: a filtered site has to leave the kernel exactly as
/// it was found.
void inlineKernelBody(CallInst *call, Function *body) {
  InlineFunctionInfo IFI;
  (void)InlineFunction(*call, IFI);
  if (body->use_empty())
    body->eraseFromParent();
}

} // namespace poseidon
