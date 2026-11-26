//===- Annotations.cpp - functions and kernels marked POSEIDON_OPTIMIZE ---===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"

#include "Core/FPNode.h"
#include "Markers.h"
#include "Site.h"
#include "SiteIdentity.h"
#include "Support/Diagnostics.h"
#include "Support/Options.h"
#include "Support/Utils.h"
#include "Targets/MathLibrary.h"
#include "Targets/OffloadRuntime.h"
#include "Targets/Target.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace poseidon {

// Attribute sites: a function or kernel carrying POSEIDON_OPTIMIZE
// (__attribute__((annotate("poseidon", 0)))) is a site whose annotated
// computation is its whole body. POSEIDON_OPTIMIZE_TAU(t) passes the site's
// accuracy target as the annotation's one argument instead.

static constexpr StringLiteral kAnnotation = "poseidon";

void warnSite(Function &F, const Twine &message) {
  if (F.hasFnAttribute(kNotASite))
    return;
  F.addFnAttr(kNotASite);
  diagnose(F, message, DS_Warning);
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
/// depend on the order a container happened to hash them in. A function whose
/// Poseidon annotation cannot be read as written is left out, and the mistake
/// added to `errors`: a compile error on the function rather than a site
/// without the target its source asked for.
static MapVector<Function *, std::optional<double>>
readAnnotations(Module &M, Error &errors) {
  GlobalVariable *GA;
  ConstantArray *arr = globalAnnotations(M, GA);
  if (!arr)
    return {};
  DenseMap<Function *, std::optional<double>> found;
  SmallPtrSet<Function *, 2> unreadable;
  auto refuse = [&](Function &fn, const Twine &message) {
    if (unreadable.insert(&fn).second)
      errors = joinErrors(std::move(errors), userError(fn, message));
  };
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
      refuse(*fn, displayName(*fn) +
                      " carries a Poseidon annotation whose arguments are "
                      "not one accuracy target");
      continue;
    }
    if (arg.Kind == AnnotationArgument::Target) {
      if (!isAccuracyTarget(arg.Tau)) {
        refuse(*fn, displayName(*fn) + " carries the accuracy target " +
                        shortDouble(arg.Tau) +
                        ", which is not a positive finite number");
        continue;
      }
      tau = arg.Tau;
    }
    auto [it, inserted] = found.try_emplace(fn, tau);
    if (inserted || !tau)
      continue;
    if (it->second && *it->second != *tau)
      refuse(*fn, displayName(*fn) + " carries two different accuracy targets");
    it->second = tau;
  }

  MapVector<Function *, std::optional<double>> out;
  for (Function &F : M) {
    auto it = found.find(&F);
    if (it != found.end() && !F.hasFnAttribute(kNotASite) &&
        !unreadable.count(&F))
      out.insert({&F, it->second});
  }
  return out;
}

MapVector<Function *, std::optional<double>> annotatedFunctions(Module &M) {
  Error errors = Error::success();
  MapVector<Function *, std::optional<double>> out = readAnnotations(M, errors);
  // checkSiteAnnotations reports them.
  consumeError(std::move(errors));
  return out;
}

Error checkSiteAnnotations(Module &M) {
  if (isOffloadHostHalf(M))
    return Error::success();
  Error errors = Error::success();
  readAnnotations(M, errors);
  return errors;
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
Expected<bool> propagateDeviceSites(const Options &O, Module &M) {
  const Target &T = Target::of(M);
  if (!T.isDevice())
    return false;
  MapVector<Function *, std::optional<double>> annotated =
      annotatedFunctions(M);
  if (llvm::none_of(annotated,
                    [&](auto &kv) { return !T.isKernel(*kv.first); }))
    return false;

  Error errors = Error::success();
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
        diagnose(K,
                 displayName(K) + " carries POSEIDON_OPTIMIZE and calls " +
                     names +
                     ", which carry it too; the kernel's own annotation and "
                     "target apply",
                 DS_Warning);
      continue;
    }
    if (!why.empty()) {
      const std::string message =
          displayName(K) + " calls " + names +
          ", which carry POSEIDON_OPTIMIZE, and " + why +
          ", so it is no wrapper around the call and not their site; move "
          "that work out of the kernel, or annotate the kernel instead";
      if (O.Profiling.profileAction())
        errors = joinErrors(std::move(errors), userError(K, message));
      else
        warnSite(K, message);
      continue;
    }
    for (Function *W : wrappers)
      if (W != &K)
        W->addFnAttr(kSiteCallee);
    for (Function *fn : reached)
      fn->addFnAttr(kSiteCallee);
    sites.push_back({&K, governing});
  }
  if (errors)
    return std::move(errors);
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
         !deviceMathName(callee->getName());
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
        diagnose(*CB,
                 displayName(*K) + " calls " +
                     displayName(*CB->getCalledFunction()) +
                     ", which the pass cannot fold into the site; the "
                     "floating-point work it does is neither profiled nor "
                     "rewritten",
                 DS_Warning);
  }
  return changed;
}

Error annotatedCallError(CallBase &call, StringRef name) {
  return userError(call, name + " carries POSEIDON_OPTIMIZE, so every call of "
                                "it is already a site; remove this "
                                "__poseidon_optimize or the annotation");
}

void warnSitesWithoutProfile(const Options &O, SiteRecords &T, Module &M) {
  if (T.WarnedWithoutProfile)
    return;
  T.WarnedWithoutProfile = true;
  for (Site &site : collectSites(O, M, {Site::Form::Kernel}))
    diagnose(*site.fn, kNoProfileAction, DS_Warning);
}

/// The module's POSEIDON_OPTIMIZE annotations removed, with what only they
/// referred to.
bool dropSiteAnnotations(Module &M) {
  GlobalVariable *GA;
  ConstantArray *arr = globalAnnotations(M, GA);
  if (!arr)
    return false;
  SmallVector<Constant *, 8> kept;
  SmallVector<WeakVH, 8> maybeDead;
  for (Value *op : arr->operands()) {
    auto *entry = cast<Constant>(op);
    std::optional<StringRef> str = entryAnnotation(entry);
    if (!str || *str != kAnnotation) {
      kept.push_back(entry);
      continue;
    }
    for (Use &U : cast<ConstantStruct>(entry)->operands())
      if (auto *G = dyn_cast<GlobalVariable>(U.get()->stripPointerCasts()))
        maybeDead.push_back(G);
  }
  if (kept.size() == arr->getNumOperands())
    return false;
  if (kept.empty())
    GA->eraseFromParent();
  else
    replaceAnnotations(M, GA, kept);
  eraseDeadGlobals(maybeDead, /*declarations=*/false);
  return true;
}

/// At PipelineStart without a profile action, once the sites are known: they
/// are reported and their annotations dropped. An annotation is a use of the
/// function it names, which the inliner counts as a second caller (no last-call
/// bonus, no deletion once inlined), so the module then compiles as its source
/// does without them.
bool leaveSitesAsWritten(const Options &O, SiteRecords &T, Module &M) {
  warnSitesWithoutProfile(O, T, M);

  bool changed = false;
  for (Function &F : M)
    for (StringRef attr : {kSiteCallee, kNotASite})
      if (F.hasFnAttribute(attr)) {
        F.removeFnAttr(attr);
        changed = true;
      }
  return dropSiteAnnotations(M) || changed;
}

// Host sites: a host function carrying POSEIDON_OPTIMIZE.

/// Whether `F` is the call operator of a lambda. A closure type demangles as
/// 'lambda'(...), or as $_<n> where clang needs no mangling number for it (a
/// lambda in a function of a host-only compile).
static bool isLambdaCallOperator(const Function &F) {
  std::optional<DemangledFunction> D = demangleFunction(F.getName());
  return D && D->BaseName == "operator()" &&
         (StringRef(D->Context).contains("'lambda") ||
          StringRef(D->Context).rsplit("::").second.starts_with("$_"));
}

/// As lowerPerCallSites, the host half of a CUDA or HIP compilation holds no
/// site: a function carrying POSEIDON_OPTIMIZE there compiles as written.
Expected<bool> outlineHostSites(const Options &O, SiteRecords &T, Module &M) {
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
  Error errors = Error::success();
  for (auto &[F, tau] : annotated) {
    if (F->hasFnAttribute(kSiteLowered))
      continue;
    // An annotated lambda marks the kernels that call it, which only a CUDA
    // or HIP compilation has. Its closure is no pointer the profiler can
    // shadow, so a host-only build leaves it alone.
    if (isLambdaCallOperator(*F)) {
      warnSite(*F, displayName(*F) +
                       " is a lambda and this compilation has no GPU kernel "
                       "that calls it, so it is left as written; "
                       "POSEIDON_OPTIMIZE on a lambda marks the kernels that "
                       "call it in a CUDA or HIP compilation");
      continue;
    }
    if (!active) {
      warnSite(*F, kNoProfileAction);
      continue;
    }
    auto both = llvm::find_if(
        markers, [F = F](CallInst *CI) { return markerTargetOrNull(CI) == F; });
    if (both != markers.end()) {
      errors = joinErrors(std::move(errors),
                          annotatedCallError(**both, displayName(*F)));
      continue;
    }
    if (std::optional<std::string> why = hostSiteRefusal(*F)) {
      errors =
          joinErrors(std::move(errors),
                     userError(*F, displayName(*F) + " " + *why +
                                       ", so it cannot be a Poseidon site"));
      continue;
    }
    noteSiteTarget(O, *F, tau);
    outlineHostFunction(T, *F, tau);
    changed = true;
  }
  if (errors)
    return std::move(errors);
  return changed;
}

} // namespace poseidon
