//===- Markers.cpp - marker calls and the sites they defer ----------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Markers.h"

#include "Enzyme/EnzymeCallMarkers.h"
#include "Enzyme/Utils.h"
#include "Support/Diagnostics.h"
#include "Support/Options.h"
#include "Support/Utils.h"

#include "llvm/ADT/STLExtras.h"
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

#include <cmath>
#include <optional>
#include <string>

using namespace llvm;

namespace poseidon {

/// A substring match, because a C++ frontend mangles the marker's template
/// instantiation.
bool isMarkerCall(StringRef calleeName) {
  return calleeName.contains(kMarkerName) ||
         calleeName.contains("__enzyme_fp_optimize");
}

void warn(CallBase *CI, const Twine &message) {
  diagnose(*CI, message, DS_Warning);
}

/// The body a marker names, or null, without diagnostics: a malformed marker
/// is reported once, by the handler that lowers it. An empty body is returned
/// as found; only markerTarget refuses it.
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

Value *activityArg(LLVMContext &C, StringRef name) {
  assert((name.starts_with("poseidon_") ||
          enzyme_markers::lookupEnzymeMarker(name)) &&
         "not an activity marker Enzyme reads");
  return MetadataAsValue::get(C, MDString::get(C, name));
}

Expected<Function *> markerTarget(CallInst *CI) {
  Function *F = markerTargetOrNull(CI);
  if (!F) {
    std::string buf;
    llvm::raw_string_ostream os(buf);
    os << *CI->getArgOperand(CI->hasStructRetAttr() ? 1 : 0);
    return userError(*CI, "failed to find the annotated function in " + buf);
  }
  if (F->empty())
    return userError(*CI,
                     "the annotated function " + F->getName() + " has no body");
  return F;
}

// Deferred sites: a marked body that is still an __enzyme_* request.

/// A marked body whose content is a derivative request holds no arithmetic
/// until Enzyme has lowered it, so at OptimizerEarly there is nothing to
/// profile and nothing to rewrite. Such a site is deferred: its marker call is
/// left in place and the body is marked for the late run, after Enzyme.
static constexpr StringLiteral kSiteDeferred = "poseidon-deferred";

/// The Enzyme requests whose result is the body's content. Matched as
/// substrings, as Enzyme matches them, because a C++ frontend mangles the
/// request's template instantiation.
static bool isEnzymeRequestName(StringRef name) {
  static constexpr StringLiteral kRequests[] = {
      "__enzyme_autodiff",   "__enzyme_fwddiff",  "__enzyme_fwdsplit",
      "__enzyme_augmentfwd", "__enzyme_reverse",  "__enzyme_virtualreverse",
      "__enzyme_batch",      "__enzyme_truncate", "__enzyme_error_estimate",
      "__enzyme_trace"};
  return any_of(kRequests, [&](StringRef req) { return name.contains(req); });
}

/// Enzyme names what it generates after the function it differentiated, with
/// one of these prefixes.
static bool isEnzymeGeneratedName(StringRef name) {
  return name.starts_with("fwddiffe") || name.starts_with("diffe") ||
         name.starts_with("augmented_") || name.starts_with("fwdsplit") ||
         name.starts_with("batch_");
}

/// Fold the code Enzyme generated for the request into the deferred body. What
/// Enzyme leaves behind is a body holding one call to the generated function,
/// and nothing in the -O2/-O3 pipeline inlines after OptimizerEarly, so without
/// this the site is still a call and still has no arithmetic. Only the
/// generated functions are folded; a libdevice call the body makes stays a
/// call, so the solve prices it exactly as it prices one in a primal site.
unsigned foldEnzymeGenerated(Function &body) {
  return inlineCallsWhile(body, [&](CallBase &CB) {
    auto *CI = dyn_cast<CallInst>(&CB);
    Function *callee = CI ? getFunctionFromCall(CI) : nullptr;
    return callee && !callee->isDeclaration() && callee != &body &&
           isEnzymeGeneratedName(callee->getName());
  });
}

/// A copy of a deferred body as the pipeline left it, with the functions Enzyme
/// generated for it, for a site the solve leaves alone. Nothing inlines after
/// Enzyme in a plugin-off build, so the always-inline functions Enzyme's code
/// calls (libdevice's __nv_*) stay calls there; in the copy those calls are
/// noinline, so that the late run's always-inliners leave them too and a
/// refused site is not optimized further than the code as written (a primal
/// and a tangent sqrt would merge into one).
Function *asWrittenCopy(Function &body,
                        DenseMap<Function *, Function *> &copies) {
  if (Function *copy = copies.lookup(&body))
    return copy;
  ValueToValueMapTy VMap;
  Function *copy = CloneFunction(&body, VMap);
  copy->setName(body.getName() + ".left");
  // NVPTX gives only a local symbol a name PTX can spell.
  copy->setLinkage(GlobalValue::InternalLinkage);
  copy->removeFnAttr(kSiteDeferred);
  resetClonedGUID(copy);
  copies[&body] = copy;
  for (Instruction &I : instructions(*copy))
    if (auto *CB = dyn_cast<CallBase>(&I))
      if (Function *callee = CB->getCalledFunction()) {
        if (!callee->isDeclaration() &&
            isEnzymeGeneratedName(callee->getName()))
          CB->setCalledFunction(asWrittenCopy(*callee, copies));
        if (callee->hasFnAttribute(Attribute::AlwaysInline))
          CB->addFnAttr(Attribute::NoInline);
      }
  return copy;
}

/// Whether `F`, or anything it reaches through a direct call, still asks for a
/// derivative. The walk goes through the always-inline wrappers a frontend puts
/// between the marked body and the request.
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
      Function *callee = getFunctionFromCall(CI);
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

/// True when this site belongs to the late run. Marks the body the first time,
/// so that both phases and both passes agree without re-deciding.
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
        if (Function *callee = getFunctionFromCall(CI))
          if (isMarkerCall(callee->getName()))
            out.push_back(CI);
  }
}

bool isDeferredBody(const Function &F) {
  return F.hasFnAttribute(kSiteDeferred);
}

unsigned getNextSiteId(Module &M) {
  NamedMDNode *NMD = M.getNamedMetadata(kNextSiteIdMD);
  if (!NMD || NMD->getNumOperands() == 0)
    return 0;
  auto *C = mdconst::extract<ConstantInt>(NMD->getOperand(0)->getOperand(0));
  return (unsigned)C->getZExtValue();
}

void setNextSiteId(Module &M, unsigned next) {
  NamedMDNode *NMD = M.getOrInsertNamedMetadata(kNextSiteIdMD);
  NMD->clearOperands();
  NMD->addOperand(MDNode::get(M.getContext(),
                              ConstantAsMetadata::get(ConstantInt::get(
                                  Type::getInt32Ty(M.getContext()), next))));
}

/// The name an Enzyme activity marker carries, however the frontend spelled it:
/// a metadata string in hand-written IR, a load of the __device__ int global in
/// CUDA sources.
static std::optional<StringRef> activityMarkerName(Value *res) {
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

/// Everything forwarded to the derivative request is spelled enzyme_*, as
/// Enzyme reads its own requests; poseidon_* is Poseidon's own.
static bool isMarkerName(StringRef name) {
  return name.starts_with("enzyme_") || name.starts_with("poseidon_");
}

bool isActivityMarker(Value *V, StringRef name) {
  return activityMarkerName(V) == name;
}

/// What a mistaken argument list is to become, said after what is wrong with
/// it.
static std::string listRemedy(StringRef name) {
  return ("; pass " + name +
          "'s own arguments, or an explicit list that puts enzyme_dup "
          "before each pointer and its shadow after it, or enzyme_const "
          "before it")
      .str();
}

bool isAccuracyTarget(double tau) { return tau > 0.0 && std::isfinite(tau); }

Expected<double> readTarget(CallBase *call, Value *value, const Function &fn) {
  if (!value)
    return userError(*call, "poseidon_tau without a value");
  auto *CFP = dyn_cast<ConstantFP>(value);
  if (!CFP)
    return userError(*call,
                     "the accuracy target of " + displayName(fn) +
                         " must be a compile-time floating-point constant (a "
                         "literal, a macro or a constant expression)");
  double tau = CFP->getValueAPF().convertToDouble();
  if (!isAccuracyTarget(tau))
    return userError(*call, "the accuracy target " + shortDouble(tau) + " of " +
                                displayName(fn) +
                                " is not a positive finite number");
  return tau;
}

Expected<CallArguments> readArgumentList(CallBase *call, const Function &fn,
                                         Identity identity) {
  const bool marker = identity == Identity::None;
  const bool sret = marker && (call->hasStructRetAttr() ||
                               fn.hasParamAttribute(0, Attribute::StructRet));
  SmallVector<Value *, 8> args(drop_begin(call->args(), marker ? 1 + sret : 0));
  std::optional<StringRef> activity;
  bool hasTau = false;
  for (Value *A : args)
    if (std::optional<StringRef> name = activityMarkerName(A);
        name && isMarkerName(*name)) {
      if (*name == "poseidon_tau")
        hasTau = true;
      else if (!activity)
        activity = name;
    }
  const std::string name = displayName(fn);
  if (hasTau && identity != Identity::None)
    return userError(
        *call, "poseidon_tau among the arguments of this call of " + name +
                   " gives the accuracy target only of a marker call; give "
                   "the target as the first argument of "
                   "__poseidon_optimize_tau");
  if (activity && identity == Identity::Plain)
    return userError(
        *call, name + " takes the arguments of this call as written, but " +
                   *activity + " among them is an activity marker, which " +
                   "makes them an explicit list" + listRemedy(name));
  if (activity)
    return CallArguments{ArgumentList::Explicit, {}, std::nullopt};
  if (identity == Identity::Explicit)
    return userError(
        *call, name +
                   " does not take the arguments of this call as written, "
                   "and none of them is an activity marker" +
                   listRemedy(name));
  CallArguments out{ArgumentList::Automatic, {}, std::nullopt};
  // A marker call without sret passes fn's sret pointer before the list.
  const bool resultPointer =
      sret && !call->hasStructRetAttr() && call->arg_size() > 1;
  if (resultPointer)
    out.own.push_back(call->getArgOperand(1));
  for (size_t i = 0; i < args.size(); ++i) {
    if (!isActivityMarker(args[i], "poseidon_tau")) {
      out.own.push_back(args[i]);
      continue;
    }
    if (out.tau)
      return userError(*call,
                       "this call of " + name +
                           " gives its accuracy target twice, each time after "
                           "poseidon_tau; give it once");
    ++i;
    Expected<double> tau =
        readTarget(call, i < args.size() ? args[i] : nullptr, fn);
    if (!tau)
      return tau.takeError();
    out.tau = *tau;
  }
  // A call through the plain identity is checked against fn's signature where
  // it is lowered, the same with and without a profile.
  const size_t passed = out.own.size() - resultPointer,
               params = fn.arg_size() - sret;
  if (identity != Identity::Plain && passed != params)
    return userError(*call, "this call passes " + Twine(passed) +
                                " argument(s) to " + name + ", which takes " +
                                Twine(params) +
                                ", and none of them is an activity marker" +
                                listRemedy(name));
  return out;
}

namespace {
/// What the activity markers before one argument of a marker call say: its
/// activity when they give one, the marker argument of a poseidon_shadow, and
/// whether they consumed the argument (a target, a call flag's operand).
struct ArgumentMarkers {
  std::optional<bool> dup;
  std::optional<unsigned> madeShadow;
  bool consumed = false;
};
} // namespace

/// Reads the activity markers from argument `i` of the marker call `CI` of
/// `F`, leaving `i` at the argument they precede; a poseidon_tau's target goes
/// to `out`. An error if they cannot be read.
static Expected<ArgumentMarkers>
readArgumentMarkers(CallInst *CI, Function *F, size_t &i, MarkerArgs &out) {
  ArgumentMarkers markers;
  Value *res = CI->getArgOperand(i);
  while (std::optional<StringRef> name = activityMarkerName(res)) {
    if (!isMarkerName(*name))
      break;
    unsigned skip = 0;
    if (*name == "poseidon_shadow") {
      markers.madeShadow = i;
      markers.dup = false;
    } else if (*name == "poseidon_tau") {
      out.forward[i] = false;
      ++i;
      Expected<double> tau = readTarget(
          CI, i < CI->arg_size() ? CI->getArgOperand(i) : nullptr, *F);
      if (!tau)
        return tau.takeError();
      out.tau = *tau;
      out.forward[i] = false;
      markers.consumed = true;
      return markers;
    } else if (std::optional<enzyme_markers::MarkerInfo> info =
                   enzyme_markers::lookupEnzymeMarker(*name);
               info && info->role != enzyme_markers::MarkerRole::Interleave) {
      // A modifier is read with the marker that follows it, as Enzyme's
      // handleArguments reads it, or the activity it prefixes is lost.
      if (info->role == enzyme_markers::MarkerRole::Activity)
        markers.dup =
            enzyme_markers::markerActivityTakesShadow(*info->activity);
      skip = info->extraOperands;
      if (info->role == enzyme_markers::MarkerRole::CallFlag) {
        i += skip;
        if (i >= CI->arg_size())
          return userError(*CI, *name + " without its operand");
        markers.consumed = true;
        return markers;
      }
    } else {
      return userError(*CI,
                       "unsupported Poseidon annotation argument " + *name);
    }
    i += 1 + skip;
    if (i >= CI->arg_size())
      return userError(*CI, "activity marker without an argument");
    res = CI->getArgOperand(i);
  }
  return markers;
}

/// Records argument `i` of the marker call `CI` as the caller value of the
/// next parameter of `F` (after an sret one when `sret`), with the activity
/// `markers` give it or its type implies, stepping `i` over its shadow. An
/// error if the argument fits no parameter or needs a marker.
static Error readArgument(CallInst *CI, Function *F, bool sret, size_t &i,
                          ArgumentMarkers markers, MarkerArgs &out) {
  size_t param = out.primalArgs.size() + sret;
  if (param >= F->getFunctionType()->getNumParams())
    return userError(*CI,
                     "more arguments than " + F->getName() + " has parameters");
  if (markers.madeShadow) {
    if (!F->getFunctionType()->getParamType(param)->isPointerTy())
      return userError(*CI, "poseidon_shadow on parameter " + Twine(param) +
                                " of " + F->getName() +
                                ", which is not a pointer");
    out.madeShadows.push_back(
        {*markers.madeShadow, (unsigned)i, (unsigned)param});
  }
  if (!markers.dup) {
    // Whether a pointer without a marker is followed by its shadow or not is
    // what the markers of an explicit list say, so it needs one.
    Type *PTy = F->getFunctionType()->getParamType(param);
    if (PTy->isPointerTy()) {
      return userError(
          *CI, "parameter " + Twine(param) + " of " + F->getName() +
                   " is a pointer without an activity marker in an "
                   "explicit list; put enzyme_dup before it and its shadow "
                   "after it, or enzyme_const before it");
    }
    if (!PTy->isFloatingPointTy() && !PTy->isIntegerTy())
      return userError(*CI, "parameter " + Twine(param) + " of " +
                                F->getName() +
                                " needs an explicit activity marker");
    markers.dup = false;
  }
  out.primalArgs.push_back(CI->getArgOperand(i));
  if (*markers.dup)
    ++i;
  return Error::success();
}

Expected<MarkerArgs> parseMarker(CallInst *CI, Function *F) {
  MarkerArgs out;
  bool sret =
      CI->hasStructRetAttr() || F->hasParamAttribute(0, Attribute::StructRet);
  out.forward.assign(CI->arg_size(), true);

  for (size_t i = 1 + sret; i < CI->arg_size(); ++i) {
    Expected<ArgumentMarkers> markers = readArgumentMarkers(CI, F, i, out);
    if (!markers)
      return markers.takeError();
    if (markers->consumed)
      continue;
    if (Error E = readArgument(CI, F, sret, i, *markers, out))
      return E;
  }

  size_t expected = F->getFunctionType()->getNumParams() - sret;
  if (out.primalArgs.size() != expected)
    return userError(*CI, "decoded " + Twine(out.primalArgs.size()) +
                              " argument(s) for " + F->getName() +
                              ", which takes " + Twine(expected));
  return out;
}

} // namespace poseidon
