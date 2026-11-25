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
class Module;
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

// The name LLVM's PGO knows `F` by (GlobalValue::getGlobalIdentifier):
// its symbol, after the name of its source file and ';' when it has local
// linkage as written. A static or anonymous kernel, which a device compile
// gives external linkage (and with -fgpu-rdc a renamed symbol), is local.
std::string functionIdentity(const llvm::Function &F);
// The same for the name `name` of `F`'s linkage.
std::string functionIdentity(const llvm::Function &F, llvm::StringRef name);
// Records `identity` as that of the site whose body is `body`.
void noteSiteIdentity(Session &S, llvm::Function &body, std::string identity);
// The identity of the site whose body is `body`: what noteSiteIdentity
// recorded, or else the body's own, a marker's target being a function as
// written.
std::string siteIdentity(const Session &S, const llvm::Function &body);
// Records the identity of each marker's target as written, before the
// optimizer can change its linkage.
void recordMarkerIdentities(Session &S, llvm::Module &M);

// The stem of a file named after `name`: `name` itself, unless it is longer
// than a file name can hold or `identity`, the function the file stands for,
// has local linkage; then its first 160 characters, '~' and the GUID (the
// 64-bit MD5 hash PGO keys a function by) of `name` as PGO would name a
// function of that name and linkage in that function's source file.
std::string fileStem(llvm::StringRef name, llvm::StringRef identity);
inline std::string profileNameStem(llvm::StringRef name) {
  return fileStem(name, name);
}

// A site's name for a diagnostic: the demangled function, and for a per-call
// copy the call and caller it was made for.
std::string perCallSiteName(llvm::StringRef symbol);

// The base name and the context of the function an Itanium name demangles to.
struct DemangledFunction {
  std::string BaseName, Context;
};
std::optional<DemangledFunction> demangleFunction(llvm::StringRef mangled);

// Records that `clone` was cut from `origin` for a call that passes it
// `primalArgs`.
void noteSiteClone(Session &S, llvm::Function *clone, llvm::Function *origin,
                   llvm::ArrayRef<llvm::Value *> primalArgs);
const llvm::Function *siteOriginOf(const Session &S,
                                   const llvm::Function &clone);
// The profile stem and file name of the site cut from `body`.
std::string bodyProfileStem(const Session &S, const llvm::Function &body);
std::string bodyProfileFile(const Session &S, const llvm::Function &body);
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
// noteSiteClone; null when unknown.
llvm::Value *siteArg(const Session &S, const llvm::Function *clone, unsigned i);
// Whether noteSiteClone recorded F: a site clone, not a pricing clone.
bool isSiteClone(const Session &S, const llvm::Function *F);
bool redirectNoopSite(Session &S, llvm::Function *clone);

} // namespace poseidon
#endif // POSEIDON_SITE_IDENTITY_H
