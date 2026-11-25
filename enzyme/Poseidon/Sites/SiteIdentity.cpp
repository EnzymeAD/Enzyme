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

#include "runtime/profile/ProfileFormat.h"

using namespace llvm;

namespace poseidon {

void noteSiteClone(SiteRecords &T, Function *clone, Function *origin,
                   ArrayRef<Value *> primalArgs) {
  SiteClone &info = T.Clones[clone];
  info.Origin = origin;
  info.Args.assign(primalArgs.begin(), primalArgs.end());
}

const Function *siteOriginOf(const SiteRecords &T, const Function &clone) {
  auto it = T.Clones.find(&clone);
  return it == T.Clones.end() ? nullptr : it->second.Origin;
}

/// Whether the Itanium mangling of `symbol` says it names an entity of internal
/// linkage, or one local to such an entity, by either of the two forms the ABI
/// gives it: an 'L' before an unscoped or local name (`_ZL4stepPdi`, and
/// `Z...L` inside a mangled template argument), or the anonymous namespace
/// `_GLOBAL__N_`. The linkage cannot be read instead: a device compile without
/// -fgpu-rdc gives a static or anonymous-namespace kernel external linkage
/// under its own name, since the host runtime looks a kernel up by name. Nor
/// can the demangler: ItaniumPartialDemangler consumes the 'L' and keeps no
/// node for it.
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

void noteSiteIdentity(SiteRecords &T, Function &body, std::string identity) {
  T.Identities[&body] = std::move(identity);
}

std::string siteIdentity(const SiteRecords &T, const Function &body) {
  auto it = T.Identities.find(&body);
  return it != T.Identities.end() ? it->second : functionIdentity(body);
}

void recordMarkerIdentities(SiteRecords &T, Module &M) {
  SmallVector<CallInst *, 4> markers;
  collectMarkerCalls(M, markers);
  for (CallInst *CI : markers)
    if (Function *target = markerTargetOrNull(CI))
      T.Identities.insert({target, functionIdentity(*target)});
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

std::string bodyProfileStem(const SiteRecords &T, const Function &body) {
  return fileStem(siteKeyOfBody(body), siteIdentity(T, body));
}

std::string bodyProfileFile(const SiteRecords &T, const Function &body) {
  return bodyProfileStem(T, body) + kProfileFileSuffix;
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

/// Keyed on the source body, not on the per-call clone: profiling shares one
/// instrumented clone between a body's markers, so the run writes one record
/// per body, while here a body reached from several marker calls is cloned once
/// per call and LLVM uniques the second and later clone names.
static std::string siteProfileStem(const SiteRecords &T,
                                   const Function &clone) {
  if (const Function *origin = siteOriginOf(T, clone))
    return bodyProfileStem(T, *origin);
  StringRef key = clone.getName();
  key.consume_front(kCanonicalPrefix);
  key.consume_back(kBodySuffix);
  return profileNameStem(key);
}

std::string siteFileStem(const SiteRecords &T, const Function &F) {
  const Function *origin = siteOriginOf(T, F);
  if (!origin)
    return fileStem(siteKeyOfBody(F), siteIdentity(T, F));
  // A second clone of one body (another marker of it) is told apart by the
  // suffix LLVM gave its name.
  StringRef unique = F.getName();
  unique.consume_front(kCanonicalPrefix);
  unique.consume_front(origin->getName());
  return fileStem(siteKeyOfBody(*origin) + unique.str(),
                  siteIdentity(T, *origin));
}

std::string siteProfileFile(const SiteRecords &T, const Function &clone) {
  return siteProfileStem(T, clone) + kProfileFileSuffix;
}

std::string siteKeyOfBody(const Function &body) {
  StringRef symbol = body.getName();
  symbol.consume_back(kBodySuffix);
  return symbol.str();
}

std::string siteKey(const SiteRecords &T, const Function &clone) {
  const Function *origin = siteOriginOf(T, clone);
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

bool isSiteClone(const SiteRecords &T, const Function *F) {
  return T.Clones.count(F);
}

Value *siteArg(const SiteRecords &T, const Function *clone, unsigned i) {
  auto it = T.Clones.find(clone);
  if (it == T.Clones.end() || i >= it->second.Args.size())
    return nullptr;
  const WeakTrackingVH &arg = it->second.Args[i];
  return arg.pointsToAliveValue() ? (Value *)arg : nullptr;
}

bool redirectNoopSite(SiteRecords &T, Function *clone, bool print) {
  auto it = T.Clones.find(clone);
  if (it == T.Clones.end() || !it->second.Origin)
    return false;
  Function *orig = it->second.Origin;
  if (orig == clone)
    return false;
  SmallVector<CallBase *, 4> callers = callsOf(*clone);
  for (CallBase *CB : callers)
    CB->setCalledFunction(orig);
  if (!callers.empty() && print)
    llvm::errs() << "[poseidon] no rewrite applied for " << clone->getName()
                 << "; wrapper call restored to original body "
                 << orig->getName() << " (bit-transparent no-op)\n";
  return !callers.empty();
}

} // namespace poseidon
