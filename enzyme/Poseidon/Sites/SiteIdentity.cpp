//=- SiteIdentity.cpp - what a site clone is and was cut from ---------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SiteIdentity.h"

#include "Pass/Session.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

// The profile file name rule, shared with both FP profiler runtimes.
#include "FPProfileName.h"

using namespace llvm;

namespace poseidon {

void noteSiteOrigin(Session &S, Function *clone, Function *orig) {
  S.SiteOrigins[clone] = orig;
}
const Function *siteOriginOf(const Session &S, const Function &clone) {
  auto it = S.SiteOrigins.find(&clone);
  return it == S.SiteOrigins.end() ? nullptr : it->second;
}

std::string bodyProfileStem(const Function &body) {
  return profileNameStem(("preprocess_" + body.getName()).str());
}

// Keyed on the source body, not on the per-call clone: profiling shares one
// instrumented clone between a body's markers, so the run writes one record
// per body, while here a body reached from several marker calls is cloned once
// per call and LLVM uniques the second and later clone names.
static std::string siteProfileStem(const Session &S, const Function &clone) {
  auto it = S.SiteOrigins.find(&clone);
  if (it != S.SiteOrigins.end() && it->second)
    return bodyProfileStem(*it->second);
  return profileNameStem(clone.getName().str());
}

std::string siteProfileFile(const Session &S, const Function &clone) {
  return siteProfileStem(S, clone) + ".fpprofile";
}

std::string siteKeyOfBody(const Function &body) {
  StringRef symbol = body.getName();
  symbol.consume_back("_poseidon_body");
  return symbol.str();
}

std::string siteKey(const Session &S, const Function &clone) {
  const Function *origin = siteOriginOf(S, clone);
  return siteKeyOfBody(origin ? *origin : clone);
}

SmallVector<std::string, 2> siteAliases(StringRef key) {
  SmallVector<std::string, 2> aliases;
  StringRef fn = key.split(".poseidon.").first;
  if (fn != key)
    aliases.push_back(fn.str());
  // The demangler's nodes point into the name it was given.
  std::string mangled = fn.str();
  ItaniumPartialDemangler D;
  if (!D.partialDemangle(mangled.c_str()) && D.isFunction())
    if (char *base = D.getFunctionBaseName(nullptr, nullptr)) {
      if (key != base && !is_contained(aliases, base))
        aliases.push_back(base);
      std::free(base);
    }
  return aliases;
}

void noteSiteArgs(Session &S, Function *clone, ArrayRef<Value *> primalArgs) {
  auto &v = S.SiteArgs[clone];
  v.clear();
  for (Value *A : primalArgs)
    v.emplace_back(A);
}
bool isSiteClone(const Session &S, const Function *F) {
  return S.SiteArgs.count(F);
}
Value *siteArg(const Session &S, const Function *clone, unsigned i) {
  auto it = S.SiteArgs.find(clone);
  if (it == S.SiteArgs.end() || i >= it->second.size())
    return nullptr;
  return it->second[i].pointsToAliveValue() ? (Value *)it->second[i] : nullptr;
}

bool redirectNoopSite(Session &S, Function *clone) {
  auto it = S.SiteOrigins.find(clone);
  if (it == S.SiteOrigins.end() || !it->second)
    return false;
  Function *orig = it->second;
  if (orig == clone)
    return false;
  SmallVector<CallBase *, 4> callers;
  for (User *U : clone->users())
    if (auto *CB = dyn_cast<CallBase>(U))
      if (CB->getCalledFunction() == clone)
        callers.push_back(CB);
  for (CallBase *CB : callers)
    CB->setCalledFunction(orig);
  if (!callers.empty())
    llvm::errs() << "[poseidon] no rewrite applied for " << clone->getName()
                 << "; wrapper call restored to original body "
                 << orig->getName() << " (bit-transparent no-op)\n";
  return !callers.empty();
}

} // namespace poseidon
