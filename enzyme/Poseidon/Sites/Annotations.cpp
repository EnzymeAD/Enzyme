//===- Annotations.cpp - functions and kernels marked POSEIDON_OPTIMIZE ---===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"

#include "Canonicalize.h"
#include "Core/Options.h"
#include "Enzyme/Utils.h"
#include "IR/Utils.h"
#include "LaunchDescriptors.h"
#include "Markers.h"
#include "Pass/Session.h"
#include "Site.h"
#include "SiteIdentity.h"
#include "Targets/MathLibrary.h"
#include "Targets/OffloadRuntime.h"
#include "Targets/Target.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

namespace poseidon {

// Attribute sites: a function or kernel carrying POSEIDON_OPTIMIZE
// (__attribute__((annotate("poseidon", 0)))) is a site whose annotated
// computation is its whole body. POSEIDON_OPTIMIZE_TAU(t) passes the site's
// accuracy target as the annotation's one argument instead.

static constexpr StringLiteral kAnnotation = "poseidon";

void diagnoseSite(Function &F, const Twine &message,
                  DiagnosticSeverity severity) {
  if (F.hasFnAttribute(kNotASite))
    return;
  F.addFnAttr(kNotASite);
  diagnose(F, F.getSubprogram(), message, severity);
}

namespace {

/// What the argument global of a Poseidon annotation says: POSEIDON_OPTIMIZE's
/// int 0, POSEIDON_OPTIMIZE_TAU's one double, or neither.
struct AnnotationArgument {
  enum { NoTarget, Target, Malformed } Kind;
  double Tau = 0.0;
};

} // namespace

static AnnotationArgument readAnnotationArgument(Constant *args) {
  if (!args || args->isNullValue())
    return {AnnotationArgument::NoTarget};
  auto *GV = dyn_cast<GlobalVariable>(args->stripPointerCasts());
  auto *ST = GV && GV->hasInitializer()
                 ? dyn_cast<StructType>(GV->getValueType())
                 : nullptr;
  if (!ST || ST->getNumElements() != 1)
    return {AnnotationArgument::Malformed};
  if (ST->getElementType(0)->isIntegerTy() &&
      GV->getInitializer()->isNullValue())
    return {AnnotationArgument::NoTarget};
  auto *value = ST->getElementType(0)->isDoubleTy()
                    ? dyn_cast_or_null<ConstantFP>(
                          GV->getInitializer()->getAggregateElement(0u))
                    : nullptr;
  if (!value)
    return {AnnotationArgument::Malformed};
  return {AnnotationArgument::Target, value->getValueAPF().convertToDouble()};
}

/// Replaces the module's llvm.global.annotations, `GA`, by an array of
/// `entries`.
static void replaceAnnotations(Module &M, GlobalVariable *GA,
                               ArrayRef<Constant *> entries) {
  auto *arr = cast<ConstantArray>(GA->getInitializer());
  Constant *init = ConstantArray::get(
      ArrayType::get(arr->getType()->getElementType(), entries.size()),
      entries);
  auto *NG = new GlobalVariable(
      M, init->getType(), GA->isConstant(), GA->getLinkage(), init, "", GA,
      GA->getThreadLocalMode(), GA->getAddressSpace());
  NG->setSection(GA->getSection());
  NG->takeName(GA);
  GA->eraseFromParent();
}

/// The entries of the module's llvm.global.annotations, `GA`; null when the
/// module has none.
static ConstantArray *globalAnnotations(Module &M, GlobalVariable *&GA) {
  GA = M.getGlobalVariable("llvm.global.annotations");
  return GA ? dyn_cast_or_null<ConstantArray>(GA->getInitializer()) : nullptr;
}

/// The annotation string of an llvm.global.annotations entry: none unless the
/// entry is a struct of at least two fields whose second names a C string.
static std::optional<StringRef> entryAnnotation(Value *op) {
  auto *entry = dyn_cast<ConstantStruct>(op);
  if (!entry || entry->getNumOperands() < 2)
    return std::nullopt;
  auto *text =
      dyn_cast<GlobalVariable>(entry->getOperand(1)->stripPointerCasts());
  auto *str = text && text->hasInitializer()
                  ? dyn_cast<ConstantDataArray>(text->getInitializer())
                  : nullptr;
  if (!str || !str->isCString())
    return std::nullopt;
  return str->getAsCString();
}

/// The annotated functions of this module with the accuracy target each
/// annotation carries, in module order so that the site numbering does not
/// depend on the order a container happened to hash them in. A Poseidon
/// annotation that cannot be read as written is a compile error on its function
/// rather than a site without the target its source asked for.
MapVector<Function *, std::optional<double>> annotatedFunctions(Module &M) {
  GlobalVariable *GA;
  ConstantArray *arr = globalAnnotations(M, GA);
  if (!arr)
    return {};
  DenseMap<Function *, std::optional<double>> found;
  for (Value *op : arr->operands()) {
    std::optional<StringRef> text = entryAnnotation(op);
    if (!text)
      continue;
    auto *entry = cast<ConstantStruct>(op);
    auto *fn = dyn_cast<Function>(entry->getOperand(0)->stripPointerCasts());
    if (!fn || fn->isDeclaration())
      continue;
    if (*text != kAnnotation)
      continue;
    std::optional<double> tau;
    AnnotationArgument arg = readAnnotationArgument(
        entry->getNumOperands() > 4 ? entry->getOperand(4) : nullptr);
    if (arg.Kind == AnnotationArgument::Malformed) {
      diagnoseSite(*fn,
                   displayName(*fn) +
                       " carries a Poseidon annotation whose arguments are "
                       "not one accuracy target",
                   DS_Error);
      continue;
    }
    if (arg.Kind == AnnotationArgument::Target) {
      if (!isAccuracyTarget(arg.Tau)) {
        diagnoseSite(*fn,
                     displayName(*fn) + " carries the accuracy target " +
                         shortDouble(arg.Tau) +
                         ", which is not a positive finite number",
                     DS_Error);
        continue;
      }
      tau = arg.Tau;
    }
    auto [it, inserted] = found.try_emplace(fn, tau);
    if (inserted || !tau)
      continue;
    if (it->second && *it->second != *tau)
      diagnoseSite(*fn,
                   displayName(*fn) + " carries two different accuracy targets",
                   DS_Error);
    it->second = tau;
  }

  MapVector<Function *, std::optional<double>> out;
  for (Function &F : M) {
    auto it = found.find(&F);
    if (it != found.end() && !F.hasFnAttribute(kNotASite))
      out.insert({&F, it->second});
  }
  return out;
}

void noteSiteTarget(const Options &O, const Function &F,
                    std::optional<double> tau) {
  if (O.Diagnostics.Print && tau)
    llvm::errs() << "[poseidon] site " << displayName(F) << ": accuracy target "
                 << shortDouble(*tau) << '\n';
}

// Device-function sites: POSEIDON_OPTIMIZE on a function a kernel calls (a
// lambda a generic launch helper runs, say) makes that kernel a site.

/// What `K` reaches through direct calls of defined functions, not looking
/// inside annotated ones: `wrappers` is K and everything reached that way,
/// `reached` the annotated device functions it calls.
static void
kernelCallees(Function &K,
              const MapVector<Function *, std::optional<double>> &annotated,
              SetVector<Function *> &wrappers, SetVector<Function *> &reached) {
  const Target &T = Target::of(K);
  wrappers.insert(&K);
  for (size_t w = 0; w < wrappers.size(); ++w)
    for (Instruction &I : instructions(*wrappers[w]))
      if (auto *CB = dyn_cast<CallBase>(&I))
        if (Function *callee = CB->getCalledFunction();
            callee && !callee->isDeclaration()) {
          if (annotated.count(callee) && !T.isKernel(*callee))
            reached.insert(callee);
          else
            wrappers.insert(callee);
        }
}

/// Why `K` cannot be the site of the annotated functions it calls, empty if it
/// can. A kernel is the site of an annotated device function only when the rest
/// of what it runs is a wrapper around the call (index arithmetic, a bounds
/// check, a loop over the call, further such wrappers); the whole kernel is
/// what gets profiled and rewritten, so floating-point work outside the
/// annotated function would be rewritten without having been marked.
static std::string deviceSiteRefusal(ArrayRef<Function *> wrappers) {
  for (Function *W : wrappers)
    for (Instruction &I : instructions(*W)) {
      if (isOptimizable(I))
        return "does floating-point work outside them (in " + displayName(*W) +
               ")";
      if (auto *CB = dyn_cast<CallBase>(&I);
          CB && !CB->getCalledFunction() && !CB->isInlineAsm())
        return "makes an indirect call (in " + displayName(*W) + ")";
    }
  return "";
}

/// Gives `K` the annotation `fn` carries, target included, so that every later
/// phase sees an annotated kernel.
static void annotateLike(Function &K, Function &fn) {
  Module &M = *K.getParent();
  GlobalVariable *GA;
  ConstantArray *arr = globalAnnotations(M, GA);
  assert(arr && "an annotated function has an llvm.global.annotations entry");
  SmallVector<Constant *, 8> entries;
  Constant *copy = nullptr;
  for (Value *op : arr->operands()) {
    auto *entry = cast<Constant>(op);
    entries.push_back(entry);
    std::optional<StringRef> str = entryAnnotation(entry);
    if (!str || *str != kAnnotation)
      continue;
    auto *ST = cast<ConstantStruct>(entry);
    if (ST->getOperand(0)->stripPointerCasts() != &fn)
      continue;
    SmallVector<Constant *, 5> ops;
    for (Use &U : ST->operands())
      ops.push_back(cast<Constant>(U.get()));
    ops[0] = ConstantExpr::getPointerCast(&K, ops[0]->getType());
    copy = ConstantStruct::get(ST->getType(), ops);
  }
  if (!copy)
    return;
  entries.push_back(copy);
  replaceAnnotations(M, GA, entries);
}

/// Before inlining, on the device: every kernel that calls an annotated device
/// function becomes a site with that function's target. A kernel that cannot be
/// the site is an error in a compile that generates or uses a profile, and is
/// left as written with a warning in any other; an annotated kernel keeps its
/// own annotation.
bool propagateDeviceSites(const Options &O, Module &M) {
  const Target &T = Target::of(M);
  if (!T.isDevice())
    return false;
  MapVector<Function *, std::optional<double>> annotated =
      annotatedFunctions(M);
  if (llvm::none_of(annotated,
                    [&](auto &kv) { return !T.isKernel(*kv.first); }))
    return false;

  SmallPtrSet<Function *, 4> called;
  SmallVector<std::pair<Function *, Function *>, 4> sites;
  for (Function &K : M) {
    if (K.isDeclaration() || !T.isKernel(K) || K.hasFnAttribute(kSiteLowered))
      continue;
    SetVector<Function *> wrappers, reached;
    kernelCallees(K, annotated, wrappers, reached);
    if (reached.empty())
      continue;
    called.insert(reached.begin(), reached.end());
    std::string names;
    std::optional<double> tau;
    Function *governing = reached.front();
    std::string why = deviceSiteRefusal(wrappers.getArrayRef());
    for (Function *fn : reached) {
      names += (names.empty() ? "" : ", ") + displayName(*fn);
      std::optional<double> t = annotated.lookup(fn);
      if (t && tau && *t != *tau)
        why = "calls annotated functions with different accuracy targets";
      if (t && !tau) {
        tau = t;
        governing = fn;
      }
    }
    if (annotated.count(&K)) {
      std::optional<double> own = annotated.lookup(&K);
      if (tau && (!own || *own != *tau))
        diagnose(K, K.getSubprogram(),
                 displayName(K) + " carries POSEIDON_OPTIMIZE and calls " +
                     names +
                     ", which carry it too; the kernel's own annotation and "
                     "target apply",
                 DS_Warning);
      continue;
    }
    if (!why.empty()) {
      const bool active = O.Profiling.profileAction();
      diagnoseSite(K,
                   displayName(K) + " calls " + names +
                       ", which carry POSEIDON_OPTIMIZE, and " + why +
                       "; a kernel is the site of an annotated function only "
                       "when it is a wrapper around the call" +
                       (active ? ". Move that work out of the kernel, or "
                                 "annotate the kernel instead."
                               : ". Emitting the original computation."),
                   active ? DS_Error : DS_Warning);
      continue;
    }
    for (Function *W : wrappers)
      if (W != &K)
        W->addFnAttr(kSiteCallee);
    for (Function *fn : reached)
      fn->addFnAttr(kSiteCallee);
    sites.push_back({&K, governing});
  }
  for (auto &[K, fn] : sites)
    annotateLike(*K, *fn);
  // Its kernels are the sites now; the function itself is not one.
  for (Function *fn : called)
    fn->addFnAttr(kNotASite);
  return !called.empty();
}

/// Whether `CB` calls code a device site folds in: a site's computation is
/// everything its kernel runs on the device, whatever the inliner decided, but
/// the math library, whose calls the solve prices as operations.
static bool foldsIntoSite(const Target &T, CallBase &CB) {
  Function *callee = CB.getCalledFunction();
  return callee && !callee->isDeclaration() && !T.isKernel(*callee) &&
         deviceMathName(callee->getName()).empty();
}

/// At OptimizerEarly: the kernel is the site, so what the inliner left of the
/// calls it makes (to the annotated function, the wrappers on the way to it,
/// and what those call) is folded into it here. Inlining is not forced earlier:
/// a site the solve leaves alone has to compile exactly as the kernel would
/// without the annotation.
bool foldDeviceSiteCallees(const Options &O, Module &M) {
  const Target &T = Target::of(M);
  if (!T.isDevice())
    return false;
  bool changed = false;
  for (Site &site : collectSites(O, M, {Site::Form::Kernel})) {
    Function *K = site.fn;
    changed |= inlineCallsWhile(
                   *K, [&](CallBase &CB) { return foldsIntoSite(T, CB); }) != 0;
    // A recursive callee is still called after the bounded rounds; what it
    // computes is outside the site.
    for (Instruction &I : instructions(*K))
      if (auto *CB = dyn_cast<CallBase>(&I); CB && foldsIntoSite(T, *CB))
        diagnose(*K, CB->getDebugLoc(),
                 displayName(*K) + " calls " +
                     displayName(*CB->getCalledFunction()) +
                     ", which the pass cannot fold into the site; the "
                     "floating-point work it does is neither profiled nor "
                     "rewritten",
                 DS_Warning);
  }
  return changed;
}

void reportAnnotatedCall(CallBase *call, StringRef name) {
  fail(call, name + " carries POSEIDON_OPTIMIZE, so every call of it is "
                    "already a site; remove this __poseidon_optimize or the "
                    "annotation");
}

void warnSitesWithoutProfile(Session &S, Module &M) {
  if (S.WarnedWithoutProfile)
    return;
  S.WarnedWithoutProfile = true;
  for (Site &site : collectSites(S.Opts, M, {Site::Form::Kernel}))
    diagnose(*site.fn, site.fn->getSubprogram(),
             Twine(kNoProfileAction) + " Emitting the original computation.",
             DS_Warning);
}

/// The module's POSEIDON_OPTIMIZE annotations removed, with what only they
/// referred to.
bool dropSiteAnnotations(Module &M) {
  GlobalVariable *GA;
  ConstantArray *arr = globalAnnotations(M, GA);
  if (!arr)
    return false;
  SmallVector<Constant *, 8> kept;
  SmallSetVector<GlobalVariable *, 8> maybeDead;
  for (Value *op : arr->operands()) {
    auto *entry = cast<Constant>(op);
    std::optional<StringRef> str = entryAnnotation(entry);
    if (!str || *str != kAnnotation) {
      kept.push_back(entry);
      continue;
    }
    for (Use &U : cast<ConstantStruct>(entry)->operands())
      if (auto *G = dyn_cast<GlobalVariable>(U.get()->stripPointerCasts()))
        maybeDead.insert(G);
  }
  if (kept.size() == arr->getNumOperands())
    return false;
  if (kept.empty())
    GA->eraseFromParent();
  else
    replaceAnnotations(M, GA, kept);
  for (GlobalVariable *G : maybeDead) {
    G->removeDeadConstantUsers();
    if (G->use_empty() && G->isDiscardableIfUnused())
      G->eraseFromParent();
  }
  return true;
}

/// At PipelineStart without a profile action, once the sites are known: they
/// are reported and their annotations dropped. An annotation is a use of the
/// function it names, which the inliner counts as a second caller (no last-call
/// bonus, no deletion once inlined), so the module then compiles as its source
/// does without them.
bool leaveSitesAsWritten(Session &S, Module &M) {
  warnSitesWithoutProfile(S, M);

  bool changed = false;
  for (Function &F : M)
    for (StringRef attr : {kSiteCallee, kNotASite})
      if (F.hasFnAttribute(attr)) {
        F.removeFnAttr(attr);
        changed = true;
      }
  return dropSiteAnnotations(M) || changed;
}

// Outlined sites: a site's computation moved into a body of its own that the
// function or kernel carrying the annotation calls.

/// Whether `F` is the call operator of a lambda. A closure type demangles as
/// 'lambda'(...), or as $_<n> where clang needs no mangling number for it (a
/// lambda in a function of a host-only compile).
static bool isLambdaCallOperator(const Function &F) {
  std::optional<DemangledFunction> D = demangleFunction(F.getName());
  return D && D->BaseName == "operator()" &&
         (StringRef(D->Context).contains("'lambda") ||
          StringRef(D->Context).rsplit("::").second.starts_with("$_"));
}

// Host sites: a host function carrying POSEIDON_OPTIMIZE.

static bool isHostScalar(Type *T) {
  return T->isIntegerTy() || T->isHalfTy() || T->isBFloatTy() ||
         T->isFloatTy() || T->isDoubleTy();
}

/// Why `F` cannot be a host site, empty if it can. The profiled clone is
/// differentiated with one activity per parameter, which a pointer, a
/// floating-point scalar and an integer have, and its value comes back only as
/// one of the latter two.
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
      if (getFunctionFromCall(CB) == &F)
        return "calls itself";
  return "";
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
  F.addFnAttr(kHostWrapper);
  return call;
}

/// Make the kernel a wrapper around its own body so that both phases see the
/// structure they already handle: a function holding the annotated computation,
/// reached through one call whose arguments are the site's inputs. Returns that
/// call, or null if the kernel cannot be a site.
CallInst *outlineKernelBody(Session &S, Function &K, Function *&bodyOut) {
  if (K.isVarArg() || !K.getReturnType()->isVoidTy()) {
    diagnoseSite(K, "an annotated kernel must be a void, non-variadic function",
                 DS_Error);
    return nullptr;
  }
  Module &M = *K.getParent();
  // An AMDGPU kernel takes an aggregate by reference into its constant
  // argument segment; the body takes it through a generic pointer, which the
  // profiled kernel can point at a copy that holds the shadows.
  SmallVector<Type *, 8> params(K.getFunctionType()->params());
  for (Argument &A : K.args())
    if (A.hasByRefAttr()) {
      if (!reroots(&A)) {
        diagnoseSite(K,
                     displayName(K) + " uses its aggregate parameter " +
                         Twine(A.getArgNo() + 1) +
                         " otherwise than by loading or copying from it, so "
                         "it cannot be a Poseidon site; copy the parameter "
                         "into a local object first",
                     DS_Error);
        return nullptr;
      }
      params[A.getArgNo()] = getInt8PtrTy(M.getContext());
    }
  // Named alike with and without -fgpu-rdc, which profile generation adds and
  // a solve need not: the profile is found by this name.
  Function *B = Function::Create(
      FunctionType::get(K.getReturnType(), params, false),
      GlobalValue::InternalLinkage,
      withoutExternalizedPostfix(K.getName()) + kBodySuffix, &M);
  B->setAttributes(K.getAttributes());
  noteSiteIdentity(S, *B, functionIdentity(K));
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
static void outlineHostFunction(Session &S, Function &F,
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
  B->addFnAttr(kHostBody, tau ? exactDouble(*tau) : "");
  noteSiteIdentity(S, *B, functionIdentity(F));
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

/// As lowerPerCallSites, the host half of a CUDA or HIP compilation holds no
/// site: a function carrying POSEIDON_OPTIMIZE there compiles as written.
bool outlineHostSites(Session &S, Module &M) {
  const Options &O = S.Opts;
  if (Target::of(M).isDevice() || isOffloadHostHalf(M))
    return false;
  MapVector<Function *, std::optional<double>> annotated =
      annotatedFunctions(M);
  if (annotated.empty())
    return false;
  const bool active = O.Profiling.profileAction();

  SmallVector<CallInst *, 4> markers;
  collectMarkerCalls(M, markers);

  bool changed = false;
  for (auto &[F, tau] : annotated) {
    if (F->hasFnAttribute(kSiteLowered))
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
      diagnoseSite(
          *F, Twine(kNoProfileAction) + " Emitting the original computation.",
          DS_Warning);
      continue;
    }
    auto both = llvm::find_if(
        markers, [F = F](CallInst *CI) { return markerTargetOrNull(CI) == F; });
    if (both != markers.end()) {
      reportAnnotatedCall(*both, displayName(*F));
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
    outlineHostFunction(S, *F, tau);
    changed = true;
  }
  return changed;
}

SmallVector<CallBase *, 4> callsOf(Value &F) {
  SmallVector<CallBase *, 4> out;
  for (User *U : F.users())
    if (auto *CB = dyn_cast<CallBase>(U); CB && CB->getCalledOperand() == &F)
      out.push_back(CB);
  return out;
}

void releaseHostBody(Function &B) {
  B.removeFnAttr(kHostBody);
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
