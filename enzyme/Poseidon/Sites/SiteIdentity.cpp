//===- SiteIdentity.cpp - what a site clone is and was cut from ----------===//
#include "SiteIdentity.h"

#include "Pass/Session.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

// The one filename rule, shared verbatim with both FP profiler runtimes that
// write what this file reads.
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

// Keyed on the SOURCE body, not on the per-call clone: profileSite shares one
// instrumented clone between a body's markers, so the run writes one record
// per body, while here a body reached from several marker calls is cloned once
// per call and LLVM uniques the second and later clone names. dFEM inlines one
// integrator's marker into every registered Q1D specialization, so a body with
// many marker calls is the normal case.
std::string siteProfileStem(const Session &S, const Function &clone) {
  auto it = S.SiteOrigins.find(&clone);
  if (it != S.SiteOrigins.end() && it->second)
    return profileNameStem(("preprocess_" + it->second->getName()).str());
  return profileNameStem(clone.getName().str());
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
