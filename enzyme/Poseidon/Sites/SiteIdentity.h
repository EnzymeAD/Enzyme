//===- SiteIdentity.h - what a site clone is and was cut from -------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A site is solved on a clone of its body. Per clone, the body it was cut from
// and the caller values of its parameters, and the names its profile and its
// -poseidon-apply-rewrites items are found by.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_SITE_IDENTITY_H
#define POSEIDON_SITE_IDENTITY_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <optional>
#include <string>

namespace llvm {
class Function;
class Value;
} // namespace llvm

namespace poseidon {

class Session;

// The body outlined from a site's function or kernel is named after it with
// this suffix.
constexpr llvm::StringLiteral kBodySuffix = "_poseidon_body";
// The copy made for one call of __poseidon_optimize is named
// <function><infix><caller>.<ordinal>.
constexpr llvm::StringLiteral kPerCallInfix = ".poseidon.";
// A site's canonical clone is named <body> after this prefix.
constexpr llvm::StringLiteral kCanonicalPrefix = "preprocess_";

// A site's name for a diagnostic: the demangled function, and for a per-call
// copy the call and caller it was made for.
std::string perCallSiteName(llvm::StringRef symbol);

// The base name and the context of the function an Itanium name demangles to.
struct DemangledFunction {
  std::string BaseName, Context;
};
std::optional<DemangledFunction> demangleFunction(llvm::StringRef mangled);

void noteSiteOrigin(Session &S, llvm::Function *clone, llvm::Function *orig);
const llvm::Function *siteOriginOf(const Session &S,
                                   const llvm::Function &clone);
// The profile stem and file name of the site cut from `body`.
std::string bodyProfileStem(const llvm::Function &body);
std::string bodyProfileFile(const llvm::Function &body);
// The ".fpprofile" file name of a site clone. The profiling run writes one
// record per marked body, so every clone of one body reads the same file.
std::string siteProfileFile(const Session &S, const llvm::Function &clone);
// The name -poseidon-apply-rewrites gives a site by: the symbol its body was
// cut from (an annotated kernel or host function, a marker's target, the copy
// made for one __poseidon_optimize call), the one its profile is named after.
std::string siteKeyOfBody(const llvm::Function &body);
std::string siteKey(const Session &S, const llvm::Function &clone);
// The other names a site answers to when no other site of the module does:
// the function a per-call copy was made of, and the unqualified name the
// symbol demangles to.
llvm::SmallVector<std::string, 2> siteAliases(llvm::StringRef key);
// The symbol of the function a site was cut from: its key without the
// suffixes of a per-call copy.
llvm::StringRef siteSymbol(llvm::StringRef key);
// Caller value of parameter i of a site clone, as recorded by
// noteSiteArgs; null when unknown.
llvm::Value *siteArg(const Session &S, const llvm::Function *clone, unsigned i);
// Whether noteSiteArgs recorded F: a site clone, not a pricing clone.
bool isSiteClone(const Session &S, const llvm::Function *F);
void noteSiteArgs(Session &S, llvm::Function *clone,
                  llvm::ArrayRef<llvm::Value *> primalArgs);
bool redirectNoopSite(Session &S, llvm::Function *clone);

} // namespace poseidon
#endif // POSEIDON_SITE_IDENTITY_H
