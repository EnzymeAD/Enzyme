#include "Annotations.h"

#include "Canonicalize.h"
#include "Core/Options.h"
#include "Targets/OffloadRuntime.h"
#include "Targets/Target.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <cmath>
#include <cstdio>

using namespace llvm;

namespace poseidon {

std::string shortDouble(double v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%g", v);
  return buf;
}

// --------------------------------------------------------------------------
// Attribute sites: a function or kernel carrying POSEIDON_OPTIMIZE
// (__attribute__((annotate("poseidon")))) is a site whose annotated
// computation is its whole body. POSEIDON_OPTIMIZE_TAU(t) passes the site's
// accuracy target as the annotation's one argument.
// --------------------------------------------------------------------------

namespace {

constexpr StringLiteral kAnnotation = "poseidon";

} // namespace

std::string displayName(const Function &F) {
  return llvm::demangle(F.getName().str());
}

void diagnoseSite(Function &F, const Twine &message,
                  DiagnosticSeverity severity) {
  if (F.hasFnAttribute(kNotASite))
    return;
  F.addFnAttr(kNotASite);
  F.getContext().diagnose(DiagnosticInfoUnsupported(
      F, "Poseidon: " + message.str(), F.getSubprogram(), severity));
}

namespace {

// The value of an annotation's argument global when it is one double.
std::optional<double> annotationTarget(Constant *args) {
  auto *GV = dyn_cast<GlobalVariable>(args->stripPointerCasts());
  if (!GV || !GV->hasInitializer())
    return std::nullopt;
  auto *ST = dyn_cast<StructType>(GV->getValueType());
  if (!ST || ST->getNumElements() != 1 || !ST->getElementType(0)->isDoubleTy())
    return std::nullopt;
  auto *value = dyn_cast_or_null<ConstantFP>(
      GV->getInitializer()->getAggregateElement(0u));
  if (!value)
    return std::nullopt;
  return value->getValueAPF().convertToDouble();
}

// The C string an annotation entry's second field names; none when that is not
// a global with a C-string initializer.
std::optional<StringRef> annotationString(ConstantStruct &entry) {
  auto *text =
      dyn_cast<GlobalVariable>(entry.getOperand(1)->stripPointerCasts());
  auto *str = text && text->hasInitializer()
                  ? dyn_cast<ConstantDataArray>(text->getInitializer())
                  : nullptr;
  if (!str || !str->isCString())
    return std::nullopt;
  return str->getAsCString();
}

// Replaces the module's llvm.global.annotations, `GA`, by an array of
// `entries`.
void replaceAnnotations(Module &M, GlobalVariable *GA,
                        ArrayRef<Constant *> entries) {
  auto *arr = cast<ConstantArray>(GA->getInitializer());
  auto *init = ConstantArray::get(
      ArrayType::get(arr->getType()->getElementType(), entries.size()),
      entries);
  auto *NG = new GlobalVariable(
      M, init->getType(), GA->isConstant(), GA->getLinkage(), init, "", GA,
      GA->getThreadLocalMode(), GA->getAddressSpace());
  NG->setSection(GA->getSection());
  NG->takeName(GA);
  GA->eraseFromParent();
}

} // namespace

// The annotated functions of this module with the accuracy target each
// annotation carries, in module order so that the site numbering does not
// depend on the order a container happened to hash them in. A Poseidon
// annotation that cannot be read as written is a compile error on its function
// rather than a site without the target its source asked for.
MapVector<Function *, std::optional<double>> annotatedFunctions(Module &M) {
  auto *GA = M.getGlobalVariable("llvm.global.annotations");
  auto *arr =
      GA ? dyn_cast_or_null<ConstantArray>(GA->getInitializer()) : nullptr;
  if (!arr)
    return {};
  DenseMap<Function *, std::optional<double>> found;
  for (Value *op : arr->operands()) {
    auto *entry = dyn_cast<ConstantStruct>(op);
    if (!entry || entry->getNumOperands() < 2)
      continue;
    auto *fn = dyn_cast<Function>(entry->getOperand(0)->stripPointerCasts());
    if (!fn || fn->isDeclaration())
      continue;
    std::optional<StringRef> text = annotationString(*entry);
    if (!text)
      continue;
    StringRef str = *text;
    if (str != kAnnotation) {
      if (str.starts_with("poseidon;"))
        diagnoseSite(*fn,
                     displayName(*fn) + " carries the annotation \"" + str +
                         "\"; an accuracy target is written "
                         "POSEIDON_OPTIMIZE_TAU(t)",
                     DS_Error);
      continue;
    }
    std::optional<double> tau;
    Constant *args =
        entry->getNumOperands() > 4 ? entry->getOperand(4) : nullptr;
    if (args && !args->isNullValue()) {
      tau = annotationTarget(args);
      if (!tau) {
        diagnoseSite(*fn,
                     displayName(*fn) +
                         " carries a Poseidon annotation whose arguments are "
                         "not one accuracy target",
                     DS_Error);
        continue;
      }
      if (!(*tau > 0.0) || !std::isfinite(*tau)) {
        diagnoseSite(*fn,
                     displayName(*fn) + " carries the accuracy target " +
                         shortDouble(*tau) +
                         ", which is not a positive finite number",
                     DS_Error);
        continue;
      }
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
                 << shortDouble(*tau) << "\n";
}

// The kernels of this module that are sites, in module order. `byRegex`, when
// given, receives the sites -poseidon-kernels named and the attribute did not:
// those are the ones the cost-share filter applies to. `tauOut`, when given,
// receives the accuracy target of each site whose annotation carried one.
void collectAnnotatedSites(const Options &O, Module &M,
                           SmallVectorImpl<Function *> &out,
                           SmallPtrSetImpl<Function *> *byRegex,
                           DenseMap<Function *, double> *tauOut) {
  const Target &T = Target::of(M);
  SmallPtrSet<Function *, 8> found;

  for (auto &[fn, tau] : annotatedFunctions(M)) {
    // In a CUDA host compilation the same annotation lands on the launch
    // stub; the site itself lives in the device module.
    if (isLaunchStub(*fn) || fn->hasFnAttribute(kSiteLowered))
      continue;
    // A host function is a site of its own (outlineHostSites); a device
    // function makes the kernels that call it sites (propagateDeviceSites),
    // and what is left here no kernel of this translation unit calls.
    if (!T.isKernel(*fn)) {
      if (T.isDevice())
        diagnoseSite(*fn,
                     displayName(*fn) +
                         " is a device function no kernel of this translation "
                         "unit calls; on the device, POSEIDON_OPTIMIZE marks a "
                         "kernel or a function a kernel calls. Emitting the "
                         "original computation.",
                     DS_Warning);
      continue;
    }
    found.insert(fn);
    if (tau && tauOut)
      (*tauOut)[fn] = *tau;
  }

  if (!O.Profiling.Kernels.empty()) {
    bool all = O.Profiling.Kernels == "all";
    Regex re(O.Profiling.Kernels);
    std::string err;
    if (!all && !re.isValid(err))
      report_fatal_error(Twine("-poseidon-kernels=") + O.Profiling.Kernels +
                         " is not a valid regular expression: " + err);
    for (Function &F : M)
      if (!F.isDeclaration() && !F.hasFnAttribute(kSiteLowered) &&
          T.isKernel(F) && (all || re.match(F.getName())))
        if (found.insert(&F).second && byRegex)
          byRegex->insert(&F);
  }

  for (Function &F : M)
    if (found.count(&F))
      out.push_back(&F);
}

// --------------------------------------------------------------------------
// Device-function sites: POSEIDON_OPTIMIZE on a function a kernel calls (the
// lambda an mfem::forall body is written as) makes that kernel a site.
// --------------------------------------------------------------------------

namespace {

// What `K` reaches through direct calls of defined functions, not looking
// inside annotated ones: `wrappers` is K and everything reached that way,
// `reached` the annotated device functions it calls.
void kernelCallees(
    Function &K, const MapVector<Function *, std::optional<double>> &annotated,
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

// Why `K` cannot be the site of the annotated functions it calls, empty if it
// can. A kernel is the site of an annotated device function only when the rest
// of what it runs is a wrapper around the call (index arithmetic, a bounds
// check, a loop over the call, further such wrappers); the whole kernel is what
// gets profiled and rewritten, so floating-point work outside the annotated
// function would be rewritten without having been marked.
std::string deviceSiteRefusal(ArrayRef<Function *> wrappers) {
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

// Gives `K` the annotation `fn` carries, target included, so that every later
// phase sees an annotated kernel.
void annotateLike(Function &K, Function &fn) {
  Module &M = *K.getParent();
  auto *GA = M.getGlobalVariable("llvm.global.annotations");
  auto *arr = cast<ConstantArray>(GA->getInitializer());
  SmallVector<Constant *, 8> entries;
  Constant *copy = nullptr;
  for (Value *op : arr->operands()) {
    auto *entry = cast<Constant>(op);
    entries.push_back(entry);
    auto *ST = dyn_cast<ConstantStruct>(entry);
    if (!ST || ST->getNumOperands() < 2 ||
        ST->getOperand(0)->stripPointerCasts() != &fn)
      continue;
    std::optional<StringRef> str = annotationString(*ST);
    if (!str || *str != kAnnotation)
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

} // namespace

// Before inlining, on the device: every kernel that calls an annotated device
// function becomes a site with that function's target. A kernel that cannot be
// the site is left as written with a warning; an annotated kernel keeps its own
// annotation.
bool propagateDeviceSites(Module &M) {
  const Target &T = Target::of(M);
  if (!T.isDevice())
    return false;
  auto annotated = annotatedFunctions(M);
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
        K.getContext().diagnose(DiagnosticInfoUnsupported(
            K,
            "Poseidon: " + displayName(K) +
                " carries POSEIDON_OPTIMIZE and calls " + names +
                ", which carry it too; the kernel's own annotation and target "
                "apply",
            K.getSubprogram(), DS_Warning));
      continue;
    }
    if (!why.empty()) {
      diagnoseSite(K,
                   displayName(K) + " calls " + names +
                       ", which carry POSEIDON_OPTIMIZE, and " + why +
                       "; a kernel is the site of an annotated function only "
                       "when it is a wrapper around the call. Emitting the "
                       "original computation.",
                   DS_Warning);
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

// At OptimizerEarly: the kernel is the site, so what the inliner left of the
// call to the annotated function is folded into it here. Inlining is not forced
// earlier: a site the solve leaves alone has to compile exactly as the kernel
// would without the annotation.
bool foldDeviceSiteCallees(Module &M) {
  const Target &T = Target::of(M);
  if (!T.isDevice())
    return false;
  bool changed = false;
  for (auto &[K, tau] : annotatedFunctions(M)) {
    if (!T.isKernel(*K) || K->hasFnAttribute(kSiteLowered))
      continue;
    for (unsigned round = 0; round < 8; ++round) {
      SmallVector<CallBase *, 4> calls;
      for (Instruction &I : instructions(*K))
        if (auto *CB = dyn_cast<CallBase>(&I))
          if (Function *callee = CB->getCalledFunction();
              callee && !callee->isDeclaration() &&
              callee->hasFnAttribute(kSiteCallee))
            calls.push_back(CB);
      if (calls.empty())
        break;
      for (CallBase *CB : calls) {
        InlineFunctionInfo IFI;
        changed |= InlineFunction(*CB, IFI).isSuccess();
      }
    }
  }
  return changed;
}

void warnWithoutProfile(Function &K) {
  K.getContext().diagnose(DiagnosticInfoUnsupported(
      K,
      "Poseidon: a Poseidon site was compiled without "
      "-poseidon-profile-generate or -poseidon-profile-use=<dir>. Emitting "
      "the original computation.",
      K.getSubprogram(), DS_Warning));
}

// The module's POSEIDON_OPTIMIZE annotations removed, with what only they
// referred to.
bool dropSiteAnnotations(Module &M) {
  auto *GA = M.getGlobalVariable("llvm.global.annotations");
  auto *arr =
      GA ? dyn_cast_or_null<ConstantArray>(GA->getInitializer()) : nullptr;
  if (!arr)
    return false;
  SmallVector<Constant *, 8> kept;
  SmallSetVector<GlobalVariable *, 8> maybeDead;
  for (Value *op : arr->operands()) {
    auto *entry = cast<Constant>(op);
    auto *ST = dyn_cast<ConstantStruct>(entry);
    std::optional<StringRef> str;
    if (ST && ST->getNumOperands() > 1)
      str = annotationString(*ST);
    if (!str || *str != kAnnotation) {
      kept.push_back(entry);
      continue;
    }
    for (Use &U : ST->operands())
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

// At PipelineStart without a profile action, once the sites are known: they
// are reported and their annotations dropped. An annotation is a use of the
// function it names, which the inliner counts as a second caller (no last-call
// bonus, no deletion once inlined), so the module then compiles as its source
// does without them.
bool leaveSitesAsWritten(const Options &O, Module &M) {
  SmallVector<Function *, 4> sites;
  SmallPtrSet<Function *, 4> byRegex;
  collectAnnotatedSites(O, M, sites, &byRegex);
  for (Function *K : sites)
    if (!byRegex.count(K))
      warnWithoutProfile(*K);

  bool changed = false;
  for (Function &F : M)
    for (StringRef attr : {kSiteCallee, kNotASite})
      if (F.hasFnAttribute(attr)) {
        F.removeFnAttr(attr);
        changed = true;
      }
  return dropSiteAnnotations(M) || changed;
}

} // namespace poseidon
