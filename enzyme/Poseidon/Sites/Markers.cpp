//===- Markers.cpp - marker calls and the sites they defer ----------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Markers.h"

#include "Annotations.h"
#include "Core/Options.h"
#include "Enzyme/EnzymeCallMarkers.h"
#include "IR/Utils.h"

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

// A substring match, because a C++ frontend mangles the marker's template
// instantiation.
bool isMarkerCall(StringRef calleeName) {
  return calleeName.contains(kMarkerName) ||
         calleeName.contains("__enzyme_fp_optimize");
}

void warn(CallBase *CI, const Twine &message) {
  diagnose(*CI->getFunction(), CI->getDebugLoc(), message, DS_Warning);
}

void fail(CallBase *CI, const Twine &message) {
  diagnose(*CI->getFunction(), CI->getDebugLoc(), message, DS_Error);
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

// The body a marker names, or null, without diagnostics: a malformed marker
// is reported once, by the handler that lowers it. An empty body is returned
// as found; only markerTarget refuses it.
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
  return MetadataAsValue::get(C, MDString::get(C, name));
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

// Deferred sites: a marked body that is still an __enzyme_* request.

// A marked body whose content is a derivative request holds no arithmetic
// until Enzyme has lowered it, so at OptimizerEarly there is nothing to
// profile and nothing to rewrite. Such a site is deferred: its marker call is
// left in place and the body is marked for the late run, after Enzyme.
static constexpr StringLiteral kSiteDeferred = "poseidon-deferred";

// The Enzyme requests whose result is the body's content. Matched as
// substrings, as Enzyme matches them, because a C++ frontend mangles the
// request's template instantiation.
static bool isEnzymeRequestName(StringRef name) {
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

// Enzyme names what it generates after the function it differentiated, with
// one of these prefixes.
static bool isEnzymeGeneratedName(StringRef name) {
  return name.starts_with("fwddiffe") || name.starts_with("diffe") ||
         name.starts_with("augmented_") || name.starts_with("fwdsplit") ||
         name.starts_with("batch_");
}

// Fold the code Enzyme generated for the request into the deferred body. What
// Enzyme leaves behind is a body holding one call to the generated function,
// and nothing in the -O2/-O3 pipeline inlines after OptimizerEarly, so without
// this the site is still a call and still has no arithmetic. Only the generated
// functions are folded; a libdevice call the body makes stays a call, so the
// solve prices it exactly as it prices one in a primal site.
unsigned foldEnzymeGenerated(Function &body) {
  return inlineCallsWhile(body, [&](CallBase &CB) {
    auto *CI = dyn_cast<CallInst>(&CB);
    Function *callee = CI ? calleeOf(CI) : nullptr;
    return callee && !callee->isDeclaration() && callee != &body &&
           isEnzymeGeneratedName(callee->getName());
  });
}

// A copy of a deferred body as the pipeline left it, with the functions Enzyme
// generated for it, for a site the solve leaves alone. Nothing inlines after
// Enzyme in a plugin-off build, so the always-inline functions Enzyme's code
// calls (libdevice's __nv_*) stay calls there; in the copy those calls are
// noinline, so that the late run's always-inliners leave them too and a
// refused site is not optimized further than the code as written (a primal
// and a tangent sqrt would merge into one).
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

bool isDeferredBody(const Function &F) {
  return F.hasFnAttribute(kSiteDeferred);
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

// The name an Enzyme activity marker carries, however the frontend spelled it:
// a metadata string in hand-written IR, a load of the __device__ int global in
// CUDA sources.
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

// Everything forwarded to the derivative request is spelled enzyme_*, as Enzyme
// reads its own requests; poseidon_* is Poseidon's own.
static bool isMarkerName(StringRef name) {
  return name.starts_with("enzyme_") || name.starts_with("poseidon_");
}

bool isActivityMarker(Value *V, StringRef name) {
  return activityMarkerName(V) == name;
}

// What a mistaken argument list is to become, said after what is wrong with it.
static std::string listRemedy(StringRef name) {
  return ("; pass " + name +
          "'s own arguments, or an explicit list that puts enzyme_dup "
          "before each pointer and its shadow after it, or enzyme_const "
          "before it")
      .str();
}

bool isAccuracyTarget(double tau) { return tau > 0.0 && std::isfinite(tau); }

std::optional<double> readTarget(CallBase *call, Value *value,
                                 const Function &fn) {
  if (!value) {
    fail(call, "poseidon_tau without a value");
    return std::nullopt;
  }
  auto *CFP = dyn_cast<ConstantFP>(value);
  if (!CFP) {
    fail(call, "the accuracy target of " + displayName(fn) +
                   " must be a compile-time floating-point constant (a "
                   "literal, a macro or a constant expression)");
    return std::nullopt;
  }
  double tau = CFP->getValueAPF().convertToDouble();
  if (!isAccuracyTarget(tau)) {
    fail(call, "the accuracy target " + shortDouble(tau) + " of " +
                   displayName(fn) + " is not a positive finite number");
    return std::nullopt;
  }
  return tau;
}

std::optional<CallArguments>
readArgumentList(CallBase *call, const Function &fn, Identity identity) {
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
  if (hasTau && identity != Identity::None) {
    fail(call, "poseidon_tau among the arguments of this call of " + name +
                   " gives the accuracy target only of a marker call; give "
                   "the target as the first argument of "
                   "__poseidon_optimize_tau");
    return std::nullopt;
  }
  if (activity && identity == Identity::Plain) {
    fail(call, name + " takes the arguments of this call as written, but " +
                   *activity + " among them is an activity marker, which " +
                   "makes them an explicit list" + listRemedy(name));
    return std::nullopt;
  }
  if (activity)
    return CallArguments{ArgumentList::Explicit, {}, std::nullopt};
  if (identity == Identity::Explicit) {
    fail(call, name +
                   " does not take the arguments of this call as written, "
                   "and none of them is an activity marker" +
                   listRemedy(name));
    return std::nullopt;
  }
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
    if (out.tau) {
      fail(call, "this call of " + name +
                     " gives its accuracy target twice, each time after "
                     "poseidon_tau; give it once");
      return std::nullopt;
    }
    ++i;
    out.tau = readTarget(call, i < args.size() ? args[i] : nullptr, fn);
    if (!out.tau)
      return std::nullopt;
  }
  // A call through the plain identity is checked against fn's signature where
  // it is lowered, the same with and without a profile.
  const size_t passed = out.own.size() - resultPointer,
               params = fn.arg_size() - sret;
  if (identity != Identity::Plain && passed != params) {
    fail(call, "this call passes " + Twine(passed) + " argument(s) to " + name +
                   ", which takes " + Twine(params) +
                   ", and none of them is an activity marker" +
                   listRemedy(name));
    return std::nullopt;
  }
  return out;
}

std::optional<MarkerArgs> parseMarker(CallInst *CI, Function *F) {
  MarkerArgs out;
  bool sret =
      CI->hasStructRetAttr() || F->hasParamAttribute(0, Attribute::StructRet);
  out.forward.assign(CI->arg_size(), true);

  for (size_t i = 1 + sret; i < CI->arg_size(); ++i) {
    Value *res = CI->getArgOperand(i);
    std::optional<bool> dup;
    std::optional<unsigned> madeShadow;
    bool skipArg = false;

    while (auto name = activityMarkerName(res)) {
      if (!isMarkerName(*name))
        break;
      unsigned skip = 0;
      if (*name == "poseidon_shadow") {
        madeShadow = i;
        dup = false;
      } else if (*name == "poseidon_tau") {
        out.forward[i] = false;
        ++i;
        std::optional<double> tau = readTarget(
            CI, i < CI->arg_size() ? CI->getArgOperand(i) : nullptr, *F);
        if (!tau)
          return std::nullopt;
        out.errTol = *tau;
        out.forward[i] = false;
        skipArg = true;
        break;
      } else if (std::optional<enzyme_markers::MarkerInfo> info =
                     enzyme_markers::lookupEnzymeMarker(*name);
                 info && info->role != enzyme_markers::MarkerRole::Interleave) {
        // A modifier is read with the marker that follows it, as Enzyme's
        // handleArguments reads it, or the activity it prefixes is lost.
        if (info->role == enzyme_markers::MarkerRole::Activity)
          dup = enzyme_markers::markerActivityTakesShadow(*info->activity);
        skip = info->extraOperands;
        if (info->role == enzyme_markers::MarkerRole::CallFlag) {
          i += skip;
          if (i >= CI->arg_size()) {
            fail(CI, *name + " without its operand");
            return std::nullopt;
          }
          skipArg = true;
          break;
        }
      } else {
        fail(CI, "unsupported Poseidon annotation argument " + *name);
        return std::nullopt;
      }
      i += 1 + skip;
      if (i >= CI->arg_size()) {
        fail(CI, "activity marker without an argument");
        return std::nullopt;
      }
      res = CI->getArgOperand(i);
    }
    if (skipArg)
      continue;

    size_t param = out.primalArgs.size() + sret;
    if (param >= F->getFunctionType()->getNumParams()) {
      fail(CI, "more arguments than " + F->getName() + " has parameters");
      return std::nullopt;
    }
    if (madeShadow) {
      if (!F->getFunctionType()->getParamType(param)->isPointerTy()) {
        fail(CI, "poseidon_shadow on parameter " + Twine(param) + " of " +
                     F->getName() + ", which is not a pointer");
        return std::nullopt;
      }
      out.madeShadows.push_back({*madeShadow, (unsigned)i, (unsigned)param});
    }
    if (!dup) {
      // Whether a pointer without a marker is followed by its shadow or not is
      // what the markers of an explicit list say, so it needs one.
      Type *PTy = F->getFunctionType()->getParamType(param);
      if (PTy->isPointerTy()) {
        fail(CI, "parameter " + Twine(param) + " of " + F->getName() +
                     " is a pointer without an activity marker in an "
                     "explicit list; put enzyme_dup before it and its shadow "
                     "after it, or enzyme_const before it");
        return std::nullopt;
      }
      if (PTy->isFloatingPointTy() || PTy->isIntegerTy())
        dup = false;
      else {
        fail(CI, "parameter " + Twine(param) + " of " + F->getName() +
                     " needs an explicit activity marker");
        return std::nullopt;
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
    return std::nullopt;
  }
  return out;
}

} // namespace poseidon
