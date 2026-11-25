//===- CallSites.cpp - calls of __poseidon_optimize -----------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallSites.h"

#include "Annotations.h"
#include "Core/Options.h"
#include "Enzyme/Utils.h"
#include "IR/Utils.h"
#include "Markers.h"
#include "Site.h"
#include "SiteIdentity.h"
#include "Targets/OffloadRuntime.h"
#include "Targets/Target.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/bit.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include <optional>

using namespace llvm;

namespace poseidon {

/// The name of a caller with the accuracy targets of its per-call sites left
/// out. A target the caller takes as a template argument gives it one
/// instantiation, and one name, per target, while the profile of a call does
/// not depend on its target. In an Itanium name a target is a literal, L, the
/// type, the value, E: a floating-point literal spells the hex of its bits, and
/// a 64-bit unsigned integer that carries the bits of the double, the form a
/// C++17 program uses, spells them in decimal. A spelling runs from its L to
/// its E, so it matches whole literals only; it becomes the external name
/// `tau`.
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

/// The accuracy target the identity call `H` carries for its call, 0 when it
/// carries none; none, with an error, when it is no accuracy target.
static std::optional<double> identityTarget(CallInst *H, CallBase *call,
                                            const Function &fn) {
  if (H->arg_size() < 2)
    return 0.0;
  return readTarget(call, H->getArgOperand(1), fn);
}

/// `call` made the marker call over `fn` with `tau`, unless 0, as its
/// poseidon_tau, then enzyme_primal_return for a value, then `list` with the
/// parameter attributes `listAttrs`. The call's value is fn's, in the profiling
/// run as well.
static CallInst *emitMarkerCall(CallBase *call, Function *fn, double tau,
                                ArrayRef<Value *> list,
                                ArrayRef<AttributeSet> listAttrs) {
  LLVMContext &C = call->getContext();
  SmallVector<Value *, 16> args{fn};
  if (tau > 0.0) {
    args.push_back(activityArg(C, "poseidon_tau"));
    args.push_back(ConstantFP::get(Type::getDoubleTy(C), tau));
  }
  if (!call->getType()->isVoidTy())
    args.push_back(activityArg(C, "enzyme_primal_return"));
  SmallVector<AttributeSet, 16> params(args.size());
  append_range(params, listAttrs);
  append_range(args, list);
  auto *FT = FunctionType::get(call->getType(), {getInt8PtrTy(C)},
                               /*isVarArg=*/true);
  FunctionCallee marker =
      call->getModule()->getOrInsertFunction(kMarkerName, FT);
  IRBuilder<> Builder(call);
  CallInst *CI = Builder.CreateCall(FT, marker.getCallee(), args);
  CI->setDebugLoc(call->getDebugLoc());
  AttributeList old = call->getAttributes();
  CI->setAttributes(
      AttributeList::get(C, old.getFnAttrs(), old.getRetAttrs(), params));
  call->replaceAllUsesWith(CI);
  call->eraseFromParent();
  return CI;
}

/// `marker`, a marker call of `fn`, made the call of fn with the primal values
/// of its arguments, or erased when its list cannot be read.
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

/// `call`, a marker call of `fn` whose list `list` reads as automatic, made the
/// call __poseidon_optimize or,
/// with the list's target, __poseidon_optimize_tau writes: fn's own arguments,
/// as given, through the plain identity, whose target is added to the caller's
/// `targets`. The host half of a CUDA or HIP compilation holds no site, so
/// there it is the call of fn. Erased, with an error at `call`, when the
/// arguments are not of fn's parameter types or the call's value is used as
/// another type than fn returns.
static void asPlainCall(CallBase *call, Function *fn, const CallArguments &list,
                        bool hostHalf,
                        DenseMap<Function *, SmallVector<double, 2>> &targets) {
  const std::string name = displayName(*fn);
  auto types = map_range(list.own, [](Value *A) { return A->getType(); });
  std::string why;
  if (call->hasStructRetAttr() ||
      !llvm::equal(fn->getFunctionType()->params(), types))
    why = "this call passes " + name +
          " arguments of other types than it takes, and none of them is an "
          "activity marker; pass its own arguments";
  else if (!call->use_empty() && call->getType() != fn->getReturnType())
    why = "this call uses the value of " + name +
          " as another type than it returns; give the call " + name +
          "'s return type";
  if (!why.empty()) {
    fail(call, why);
    if (!call->use_empty())
      call->replaceAllUsesWith(PoisonValue::get(call->getType()));
    call->eraseFromParent();
    return;
  }
  Module &M = *call->getModule();
  IRBuilder<> Builder(call);
  Value *callee = fn;
  if (!hostHalf) {
    LLVMContext &C = M.getContext();
    Type *Ptr = getInt8PtrTy(C);
    CallInst *identity;
    if (list.tau) {
      Type *Double = Type::getDoubleTy(C);
      FunctionCallee F = M.getOrInsertFunction(
          kPlainIdentityTau, FunctionType::get(Ptr, {Ptr, Double}, false));
      identity =
          Builder.CreateCall(F, {fn, ConstantFP::get(Double, *list.tau)});
      targets[call->getFunction()].push_back(*list.tau);
    } else {
      FunctionCallee F = M.getOrInsertFunction(
          kPlainIdentity, FunctionType::get(Ptr, {Ptr}, false));
      identity = Builder.CreateCall(F, {fn});
    }
    identity->setDebugLoc(call->getDebugLoc());
    callee = identity;
  }
  CallInst *direct =
      Builder.CreateCall(fn->getFunctionType(), callee, list.own);
  direct->setCallingConv(fn->getCallingConv());
  direct->setDebugLoc(call->getDebugLoc());
  if (!call->use_empty())
    call->replaceAllUsesWith(direct);
  call->eraseFromParent();
}

/// The calls of the module's functions named `names`, in module order.
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

/// __poseidon_optimize with an explicit list: the call through the explicit
/// identity is the marker call over fn, and the marker's lowering makes it the
/// site. The host half of a CUDA or HIP compilation holds no site, so there the
/// call runs fn with the primal values of its arguments.
static bool lowerExplicitCalls(Module &M) {
  SmallVector<CallInst *, 4> identities =
      callsOfAny(M, {kExplicitIdentity, kExplicitIdentityTau});
  if (identities.empty())
    return false;
  const bool hostHalf = isOffloadHostHalf(M);
  MapVector<Function *, std::optional<double>> annotated =
      annotatedFunctions(M);
  for (CallInst *H : identities) {
    auto *fn = dyn_cast<Function>(H->getArgOperand(0)->stripPointerCasts());
    SmallVector<CallBase *, 4> calls = callsOf(*H);
    if (!fn || calls.size() != 1 || !H->hasOneUse()) {
      fail(H, "__poseidon_optimize needs a function name and a call of it");
      H->replaceAllUsesWith(PoisonValue::get(H->getType()));
      H->eraseFromParent();
      continue;
    }
    CallBase *call = calls.front();
    const std::string name = displayName(*fn);
    std::optional<double> tau = identityTarget(H, call, *fn);
    // Whatever the outcome, the call stops going through the identity, which
    // nothing defines.
    call->setCalledOperand(fn);
    H->eraseFromParent();
    if (!tau)
      continue;
    if (!readArgumentList(call, *fn, Identity::Explicit))
      continue;
    if (annotated.count(fn)) {
      reportAnnotatedCall(call, name);
      continue;
    }
    SmallVector<Value *, 8> args(call->args());
    SmallVector<AttributeSet, 8> argAttrs;
    for (unsigned i = 0; i < call->arg_size(); ++i)
      argAttrs.push_back(call->getAttributes().getParamAttrs(i));
    CallInst *marker = emitMarkerCall(call, fn, *tau, args, argAttrs);
    if (hostHalf)
      callAsWritten(marker, fn);
  }
  return true;
}

/// A marker call whose list is automatic and passes a pointer among fn's own
/// arguments is the call form of __poseidon_optimize, through the plain
/// identity, so that the pass makes its shadows as it makes that form's. One
/// whose own arguments hold no pointer needs no shadow and stays the marker
/// call it is, as does one with an explicit list. In the host half of a CUDA or
/// HIP compilation, which holds no site, every marker call runs fn with the
/// primal values of its arguments.
static bool
routeAutomaticMarkers(Module &M,
                      DenseMap<Function *, SmallVector<double, 2>> &targets) {
  SmallVector<CallInst *, 4> markers;
  collectMarkerCalls(M, markers);
  const bool hostHalf = isOffloadHostHalf(M);
  bool changed = false;
  for (CallInst *CI : markers) {
    Function *fn = markerTargetOrNull(CI);
    if (!fn)
      continue;
    std::optional<CallArguments> list =
        readArgumentList(CI, *fn, Identity::None);
    if (list && list->kind == ArgumentList::Explicit) {
      if (hostHalf) {
        callAsWritten(CI, fn);
        changed = true;
      }
      continue;
    }
    if (list && !hostHalf && !CI->hasStructRetAttr() &&
        llvm::none_of(list->own,
                      [](Value *A) { return A->getType()->isPointerTy(); }))
      continue;
    changed = true;
    if (!list) {
      if (!CI->use_empty())
        CI->replaceAllUsesWith(PoisonValue::get(CI->getType()));
      CI->eraseFromParent();
      continue;
    }
    asPlainCall(CI, fn, *list, hostHalf, targets);
  }
  return changed;
}

/// The accuracy targets each function's calls through the identities with a
/// target give, which its per-call copies leave out of their names.
static DenseMap<Function *, SmallVector<double, 2>> identityTargets(Module &M) {
  DenseMap<Function *, SmallVector<double, 2>> targets;
  for (CallInst *H : callsOfAny(M, {kPlainIdentityTau, kExplicitIdentityTau}))
    if (H->arg_size() > 1)
      if (auto *C = dyn_cast<ConstantFP>(H->getArgOperand(1)))
        targets[H->getFunction()].push_back(C->getValueAPF().convertToDouble());
  return targets;
}

/// Why `call` of `fn` cannot be a per-call site; none if it can.
static std::optional<std::string> perCallRefusal(CallBase &call, Function &fn) {
  if (call.getFunctionType() != fn.getFunctionType())
    return "is called with another signature";
  if (fn.isDeclaration())
    return "has no body in this translation unit";
  return hostSiteRefusal(fn);
}

/// The copy of `fn` made for its `k`-th call in `caller`, whose name without
/// accuracy targets is `callerKey`: a host body with the call's accuracy
/// target `tau`, recorded in `T` with the identity of that call.
static Function *makePerCallCopy(SiteRecords &T, Function &fn, Function &caller,
                                 StringRef callerKey, unsigned k, double tau) {
  ValueToValueMapTy VMap;
  Function *B = CloneFunction(&fn, VMap);
  B->setName(fn.getName() + kPerCallInfix + callerKey + "." + Twine(k));
  B->setLinkage(GlobalValue::InternalLinkage);
  B->setVisibility(GlobalValue::DefaultVisibility);
  B->setComdat(nullptr);
  B->removeFnAttr(Attribute::AlwaysInline);
  B->removeFnAttr(Attribute::InlineHint);
  B->addFnAttr(Attribute::NoInline);
  B->addFnAttr(kHostBody, tau > 0.0 ? exactDouble(tau) : "");
  resetClonedGUID(B);
  noteSiteIdentity(T, *B,
                   (functionIdentity(fn) + kPerCallInfix +
                    functionIdentity(caller, callerKey) + "." + Twine(k))
                       .str());
  appendToCompilerUsed(*fn.getParent(), {B});
  return B;
}

/// __poseidon_optimize: the call through the plain identity becomes a call of
/// a copy of the function made for that call alone, which is then a host site
/// like an outlined body. The copy is named after the caller, without the
/// targets (targetFreeName), and the call's position among the caller's
/// per-call sites, taken before inlining, so that the generate and use compiles
/// name it alike at any -O level and for any target. The host half of a CUDA or
/// HIP compilation holds no site: the compilation has one device and one cost
/// model, and the call of a host-device function is the device half's site.
/// There the call through the identity is left for the inliner to fold. The
/// calls through the explicit identity and the marker calls are read first
/// (lowerExplicitCalls, routeAutomaticMarkers).
bool lowerPerCallSites(const Options &O, SiteRecords &T, Module &M) {
  // The targets of the explicit calls are left out of their callers' names
  // too, so they are read before those calls become marker calls; the
  // targets poseidon_tau gives in marker calls are added as those calls
  // are routed through the plain identity.
  DenseMap<Function *, SmallVector<double, 2>> targets = identityTargets(M);
  bool changed = lowerExplicitCalls(M);
  changed |= routeAutomaticMarkers(M, targets);
  if (isOffloadHostHalf(M))
    return changed;
  SmallVector<CallInst *, 4> helpers =
      callsOfAny(M, {kPlainIdentity, kPlainIdentityTau});
  if (helpers.empty())
    return changed;
  const bool active = O.Profiling.profileAction();
  MapVector<Function *, std::optional<double>> annotated =
      annotatedFunctions(M);
  // Two callers whose names differ in targets alone number their calls on
  // from each other, so that they stay two sites.
  StringMap<unsigned> ordinal;

  for (CallInst *H : helpers) {
    Function *caller = H->getFunction();
    const std::string callerKey =
        targetFreeName(caller->getName(), targets.lookup(caller));
    unsigned k = ordinal[callerKey]++;
    auto *fn = dyn_cast<Function>(H->getArgOperand(0)->stripPointerCasts());
    SmallVector<CallBase *, 4> calls = callsOf(*H);
    // Whatever the outcome, the call stops going through the helper, which
    // returns its argument, so a later run of the pass does not see it again.
    auto asWritten = [&] {
      H->replaceAllUsesWith(H->getArgOperand(0));
      H->eraseFromParent();
    };
    if (!fn || calls.size() != 1 || !H->hasOneUse()) {
      if (active)
        fail(H, "__poseidon_optimize needs a function name and a call of it");
      else
        warn(H, "__poseidon_optimize needs a function name and a call of it; "
                "the call runs as written");
      asWritten();
      continue;
    }
    CallBase *call = calls.front();
    const std::string name = displayName(*fn);
    std::optional<double> tau = identityTarget(H, call, *fn);
    if (!tau || !readArgumentList(call, *fn, Identity::Plain)) {
      asWritten();
      continue;
    }
    if (annotated.count(fn)) {
      reportAnnotatedCall(call, name);
      asWritten();
      continue;
    }
    std::optional<std::string> why = perCallRefusal(*call, *fn);
    if (why && active)
      fail(call,
           name + " " + *why + ", so this call cannot be a Poseidon site");
    else if (why)
      warn(call, name + " " + *why +
                     ", so this call is not a Poseidon site and runs as "
                     "written");
    else if (!active)
      warn(call, kNoProfileAction);
    if (why || !active) {
      asWritten();
      continue;
    }

    call->setCalledOperand(
        makePerCallCopy(T, *fn, *caller, callerKey, k, *tau));
    H->eraseFromParent();
    if (O.Diagnostics.Print && *tau > 0.0)
      llvm::errs() << "[poseidon] site " << name << " at its call in "
                   << displayName(*caller) << ": accuracy target "
                   << shortDouble(*tau) << '\n';
  }
  return true;
}

Value *localObject(Value *p, const Function *caller) {
  Value *obj = getUnderlyingObject(p, /*MaxLookup=*/0);
  if (auto *AI = dyn_cast<AllocaInst>(obj))
    return AI->getFunction() == caller && AI->isStaticAlloca() ? AI : nullptr;
  if (auto *A = dyn_cast<Argument>(obj))
    return A->getParent() == caller && A->hasByValAttr() ? A : nullptr;
  return nullptr;
}

bool pointsNowhere(Value *p) {
  return isa<ConstantPointerNull>(p) || isa<UndefValue>(p);
}

namespace {

/// accessedExtent: every use of the pointer, through constant address
/// arithmetic and into the functions it is passed to.
struct ExtentWalk {
  const DataLayout &DL;
  PointerExtent ext;
  bool anyAccess = false, anyWrite = false;
  std::string why;
  SmallSet<std::pair<const Argument *, int64_t>, 4> entered;
  /// The parameters the pointer is in on the way down.
  SmallPtrSet<const Argument *, 4> active;

  bool fail(const Twine &reason) {
    why = reason.str();
    return false;
  }

  void claim(int64_t off, MaybeAlign align) {
    if (align && align->value() > ext.align) {
      ext.align = align->value();
      ext.alignedAt = off;
    }
  }

  void access(int64_t off, uint64_t bytes, bool write) {
    if (!bytes)
      return;
    const int64_t end = off + (int64_t)bytes;
    ext.lo = anyAccess ? std::min(ext.lo, off) : off;
    ext.hi = anyAccess ? std::max(ext.hi, end) : end;
    anyAccess = true;
    if (!write)
      return;
    ext.writeLo = anyWrite ? std::min(ext.writeLo, off) : off;
    ext.writeHi = anyWrite ? std::max(ext.writeHi, end) : end;
    anyWrite = true;
  }

  bool memoryAccess(int64_t off, Type *T, Align align, bool write) {
    TypeSize bytes = DL.getTypeStoreSize(T);
    if (bytes.isScalable())
      return fail("accesses a scalable vector through");
    access(off, bytes.getFixedValue(), write);
    claim(off, align);
    return true;
  }

  /// An optimizer may load anywhere a parameter's attributes call
  /// dereferenceable, also after the extent is taken, so the extent covers it.
  /// A function that receives the pointer again at another offset while it is
  /// being walked reaches no constant range through it.
  bool enter(Argument &A, int64_t off) {
    if (!entered.insert({&A, off}).second)
      return true;
    if (!active.insert(&A).second)
      return fail("passes to " + displayName(*A.getParent()) +
                  ", which passes it on to itself at another offset");
    access(off,
           std::max(A.getDereferenceableBytes(),
                    A.getDereferenceableOrNullBytes()),
           false);
    claim(off, A.getParamAlign());
    const bool ok = walk(&A, off);
    active.erase(&A);
    return ok;
  }

  bool walk(Value *P, int64_t off) {
    for (Use &U : P->uses()) {
      auto *I = dyn_cast<Instruction>(U.getUser());
      if (!I)
        return fail("uses otherwise than to load and store through");
      if (auto *GEP = dyn_cast<GetElementPtrInst>(I)) {
        APInt delta(DL.getIndexTypeSizeInBits(GEP->getType()), 0);
        if (!GEP->accumulateConstantOffset(DL, delta))
          return fail("indexes with a value that is not a constant");
        if (!walk(GEP, off + delta.getSExtValue()))
          return false;
      } else if (isa<BitCastInst, AddrSpaceCastInst>(I)) {
        if (!walk(I, off))
          return false;
      } else if (auto *LI = dyn_cast<LoadInst>(I)) {
        if (!memoryAccess(off, LI->getType(), LI->getAlign(), false))
          return false;
      } else if (auto *SI = dyn_cast<StoreInst>(I)) {
        if (U.getOperandNo() != SI->getPointerOperandIndex())
          return fail("stores to memory");
        if (!memoryAccess(off, SI->getValueOperand()->getType(), SI->getAlign(),
                          true))
          return false;
      } else if (auto *RMW = dyn_cast<AtomicRMWInst>(I)) {
        if (U.getOperandNo() != RMW->getPointerOperandIndex())
          return fail("stores to memory");
        if (!memoryAccess(off, RMW->getValOperand()->getType(), RMW->getAlign(),
                          true))
          return false;
      } else if (auto *CX = dyn_cast<AtomicCmpXchgInst>(I)) {
        if (U.getOperandNo() != CX->getPointerOperandIndex())
          return fail("stores to memory");
        if (!memoryAccess(off, CX->getNewValOperand()->getType(),
                          CX->getAlign(), true))
          return false;
      } else if (isa<MemTransferInst, MemSetInst>(I)) {
        auto *MI = cast<MemIntrinsic>(I);
        auto *len = dyn_cast<ConstantInt>(MI->getLength());
        if (!len)
          return fail("copies or sets through with a length that is not a "
                      "constant");
        const bool dest = U.getOperandNo() == 0;
        access(off, len->getZExtValue(), dest);
        claim(off, dest ? MI->getDestAlign()
                        : cast<MemTransferInst>(MI)->getSourceAlign());
      } else if (auto *II = dyn_cast<IntrinsicInst>(I);
                 II &&
                 (II->isLifetimeStartOrEnd() || II->isAssumeLikeIntrinsic())) {
        continue;
      } else if (isa<ICmpInst>(I)) {
        continue;
      } else if (auto *CB = dyn_cast<CallBase>(I)) {
        if (!CB->isArgOperand(&U))
          return fail("calls as a function");
        const unsigned no = CB->getArgOperandNo(&U);
        if (isReadNone(CB, no) && isNoCapture(CB, no))
          continue;
        if (Type *T = CB->getParamByValType(no)) {
          if (!memoryAccess(off, T, CB->getParamAlign(no).valueOrOne(), false))
            return false;
          continue;
        }
        Function *callee = CB->getCalledFunction();
        if (!callee || callee->isDeclaration() || no >= callee->arg_size())
          return fail("passes to " +
                      (callee ? displayName(*callee)
                              : std::string("an indirect call")) +
                      ", whose body the pass cannot see");
        if (!enter(*callee->getArg(no), off))
          return false;
      } else if (isa<PHINode, SelectInst>(I)) {
        return fail("merges with another pointer");
      } else if (isa<PtrToIntInst>(I)) {
        return fail("converts to an integer");
      } else if (isa<ReturnInst>(I)) {
        return fail("returns");
      } else {
        return fail("uses otherwise than to load and store through");
      }
    }
    return true;
  }
};

} // namespace

Expected<PointerExtent> accessedExtent(Argument &A) {
  ExtentWalk W{A.getParent()->getParent()->getDataLayout(), {}};
  if (!W.enter(A, 0))
    return createStringError(inconvertibleErrorCode(), W.why);
  return W.ext;
}

/// __poseidon_optimize in device code: every call of the copy lowerPerCallSites
/// made becomes a marker over that copy, so that profiling, deferral and the
/// solve are the marker's. Its pointers are marked poseidon_shadow for
/// profileSite, which makes the shadow of a pointer into an object of the
/// caller whose extent it knows (localObject; inlining has put the objects a
/// call passes there by now) or else of the constant range the call reaches
/// through it (accessedExtent).
bool lowerDeviceCallSites(const Options &O, Module &M) {
  if (!Target::of(M).isDevice())
    return false;
  SmallVector<Site, 4> bodies = collectSites(O, M, {Site::Form::HostBody});
  if (bodies.empty())
    return false;
  LLVMContext &C = M.getContext();
  for (Site &site : bodies) {
    Function *B = site.fn;
    double tau = site.target.value_or(0.0);
    releaseHostBody(*B);
    for (CallBase *CB : callsOf(*B)) {
      if (!isa<CallInst>(CB)) {
        fail(CB, perCallSiteName(B->getName()) +
                     " is reached through an invoke, so this call cannot be a "
                     "Poseidon site");
        markOutlineTransparent(B);
        continue;
      }
      std::string why;
      for (Use &U : CB->args()) {
        if (!U->getType()->isPointerTy() || pointsNowhere(U.get()) ||
            localObject(U.get(), CB->getFunction()))
          continue;
        Expected<PointerExtent> ext =
            accessedExtent(*B->getArg(U.getOperandNo()));
        if (ext)
          continue;
        why += (Twine(why.empty() ? "" : "; ") + "argument " +
                Twine(U.getOperandNo() + 1) + " is a pointer the call " +
                toString(ext.takeError()))
                   .str();
      }
      if (!why.empty()) {
        fail(CB, perCallSiteName(B->getName()) +
                     " cannot be a Poseidon site: " + why +
                     ". The pass makes the shadow of a pointer into a local "
                     "object of the caller or an aggregate it takes by value, "
                     "or else of the constant range of memory the call "
                     "reaches through the pointer. Copy such an argument into "
                     "a local object, or pass its shadow in the call: "
                     "enzyme_dup before the pointer and the shadow after it.");
        markOutlineTransparent(B);
        continue;
      }
      SmallVector<Value *, 16> list;
      for (Value *a : CB->args()) {
        Type *T = a->getType();
        list.push_back(activityArg(C, T->isPointerTy() ? "poseidon_shadow"
                                      : T->isFloatingPointTy()
                                          ? "enzyme_out"
                                          : "enzyme_const"));
        list.push_back(a);
      }
      emitMarkerCall(CB, B, tau, list, {});
    }
  }
  return true;
}

} // namespace poseidon
