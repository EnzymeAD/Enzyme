//===- SiteIdentity.cpp - what a site clone is and was cut from -----------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SiteIdentity.h"

#include "Annotations.h"
#include "Markers.h"
#include "Pass/Session.h"
#include "Targets/OffloadRuntime.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

#include "runtime/fpprofiler/FPProfileName.h"

using namespace llvm;

namespace poseidon {

void noteSiteClone(Session &S, Function *clone, Function *origin,
                   ArrayRef<Value *> primalArgs) {
  Session::SiteClone &info = S.SiteCloneOf[clone];
  info.Origin = origin;
  info.Args.assign(primalArgs.begin(), primalArgs.end());
}

const Function *siteOriginOf(const Session &S, const Function &clone) {
  auto it = S.SiteCloneOf.find(&clone);
  return it == S.SiteCloneOf.end() ? nullptr : it->second.Origin;
}

// Whether the Itanium mangling of `symbol` names an entity of internal linkage
// or one local to such an entity: an 'L' before a namespace-scope or local
// name, or an anonymous namespace. Clang gives a static or anonymous kernel
// external linkage in a device compile without -fgpu-rdc, since the host
// runtime looks it up by name, so its linkage no longer says so.
static bool mangledAsInternal(StringRef symbol) {
  static const Regex internal("(^_Z|Z)L[0-9]|_GLOBAL__N_");
  return symbol.starts_with("_Z") && internal.match(symbol);
}

std::string functionIdentity(const Function &F, StringRef name) {
  StringRef symbol = withoutExternalizedPostfix(name);
  const bool local = F.hasLocalLinkage() || symbol.size() != name.size() ||
                     mangledAsInternal(symbol);
  // The file name alone, as -static-func-strip-dirname-prefix makes PGO's: a
  // profile has to name the same site wherever the sources are checked out
  // and however the build spells their path.
  return GlobalValue::getGlobalIdentifier(
      symbol,
      local ? GlobalValue::InternalLinkage : GlobalValue::ExternalLinkage,
      sys::path::filename(F.getParent()->getSourceFileName()));
}

std::string functionIdentity(const Function &F) {
  return functionIdentity(F, F.getName());
}

void noteSiteIdentity(Session &S, Function &body, std::string identity) {
  S.SiteIdentities[&body] = std::move(identity);
}

std::string siteIdentity(const Session &S, const Function &body) {
  auto it = S.SiteIdentities.find(const_cast<Function *>(&body));
  return it != S.SiteIdentities.end() ? it->second : functionIdentity(body);
}

void recordMarkerIdentities(Session &S, Module &M) {
  SmallVector<CallInst *, 4> markers;
  collectMarkerCalls(M, markers);
  for (CallInst *CI : markers)
    if (Function *target = markerTargetOrNull(CI))
      S.SiteIdentities.insert({target, functionIdentity(*target)});
}

std::string fileStem(StringRef name, StringRef identity) {
  // The longest file named after a stem, "launches_" + stem + "_" + 16 hex
  // digits + ".json", stays under NAME_MAX = 255.
  constexpr size_t kVerbatim = 200, kKept = 160;
  auto [file, symbol] = identity.split(GlobalIdentifierDelimiter);
  const bool local = !symbol.empty();
  if (name.size() <= kVerbatim && !local)
    return name.str();
  const std::string named = local
                                ? GlobalValue::getGlobalIdentifier(
                                      name, GlobalValue::InternalLinkage, file)
                                : name.str();
  return (name.take_front(kKept) + "~" +
          utohexstr(MD5Hash(named), /*LowerCase=*/true, /*Width=*/16))
      .str();
}

std::string bodyProfileStem(const Session &S, const Function &body) {
  return fileStem((kCanonicalPrefix + body.getName()).str(),
                  siteIdentity(S, body));
}

std::string bodyProfileFile(const Session &S, const Function &body) {
  return bodyProfileStem(S, body) + kProfileFileSuffix;
}

std::string perCallSiteName(StringRef symbol) {
  auto [fnSymbol, where] = symbol.split(kPerCallInfix);
  std::string name = demangle(fnSymbol.str());
  if (!where.empty()) {
    auto [callerSymbol, k] = where.rsplit('.');
    name += " at call " + k.str() + " in " + demangle(callerSymbol.str());
  }
  return name;
}

std::optional<DemangledFunction> demangleFunction(StringRef mangled) {
  // The demangler's nodes point into the name it was given.
  std::string name = mangled.str();
  ItaniumPartialDemangler D;
  if (D.partialDemangle(name.c_str()) || !D.isFunction())
    return std::nullopt;
  char *base = D.getFunctionBaseName(nullptr, nullptr);
  char *context = D.getFunctionDeclContextName(nullptr, nullptr);
  std::optional<DemangledFunction> out;
  if (base)
    out = DemangledFunction{base, context ? context : ""};
  std::free(base);
  std::free(context);
  return out;
}

// Keyed on the source body, not on the per-call clone: profiling shares one
// instrumented clone between a body's markers, so the run writes one record
// per body, while here a body reached from several marker calls is cloned once
// per call and LLVM uniques the second and later clone names.
static std::string siteProfileStem(const Session &S, const Function &clone) {
  if (const Function *origin = siteOriginOf(S, clone))
    return bodyProfileStem(S, *origin);
  return profileNameStem(clone.getName().str());
}

std::string siteProfileFile(const Session &S, const Function &clone) {
  return siteProfileStem(S, clone) + kProfileFileSuffix;
}

std::string siteKeyOfBody(const Function &body) {
  StringRef symbol = body.getName();
  symbol.consume_back(kBodySuffix);
  return symbol.str();
}

std::string siteKey(const Session &S, const Function &clone) {
  const Function *origin = siteOriginOf(S, clone);
  return siteKeyOfBody(origin ? *origin : clone);
}

StringRef siteSymbol(StringRef key) { return key.split(kPerCallInfix).first; }

SmallVector<std::string, 2> siteAliases(StringRef key) {
  SmallVector<std::string, 2> aliases;
  StringRef fn = siteSymbol(key);
  if (fn != key)
    aliases.push_back(fn.str());
  if (std::optional<DemangledFunction> D = demangleFunction(fn))
    if (key != D->BaseName && !is_contained(aliases, D->BaseName))
      aliases.push_back(D->BaseName);
  return aliases;
}

bool isSiteClone(const Session &S, const Function *F) {
  return S.SiteCloneOf.count(F);
}

Value *siteArg(const Session &S, const Function *clone, unsigned i) {
  auto it = S.SiteCloneOf.find(clone);
  if (it == S.SiteCloneOf.end() || i >= it->second.Args.size())
    return nullptr;
  const WeakTrackingVH &arg = it->second.Args[i];
  return arg.pointsToAliveValue() ? (Value *)arg : nullptr;
}

bool redirectNoopSite(Session &S, Function *clone) {
  auto it = S.SiteCloneOf.find(clone);
  if (it == S.SiteCloneOf.end() || !it->second.Origin)
    return false;
  Function *orig = it->second.Origin;
  if (orig == clone)
    return false;
  SmallVector<CallBase *, 4> callers = callsOf(*clone);
  for (CallBase *CB : callers)
    CB->setCalledFunction(orig);
  if (!callers.empty() && S.Opts.Diagnostics.Print)
    llvm::errs() << "[poseidon] no rewrite applied for " << clone->getName()
                 << "; wrapper call restored to original body "
                 << orig->getName() << " (bit-transparent no-op)\n";
  return !callers.empty();
}

} // namespace poseidon
