//===- SiteIdentity.h - what a site clone is and was cut from -------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A site is solved on a clone of its body. Per clone, the body it was cut from
// and the caller values of its parameters, and the names its profile and its
// -poseidon-pin items are found by.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_SITES_SITE_IDENTITY_H
#define POSEIDON_SITES_SITE_IDENTITY_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/IR/ValueMap.h"

#include <map>
#include <optional>
#include <string>

namespace llvm {
class Function;
class Module;
class Value;
} // namespace llvm

namespace poseidon {

/// What a site clone was cut from: the body, and the caller values of its
/// parameters, which the clone has no callers to give while it is optimized.
struct SiteClone {
  /// The body the clone was cut from.
  llvm::Function *Origin;
  /// The caller value of each parameter of the clone, null once the value is
  /// deleted.
  llvm::SmallVector<llvm::WeakTrackingVH, 12> Args;
};

/// The sites of one module: each site clone's record, and the identity of
/// each body the pass made for a site and of each marker's target.
struct SiteRecords {
  static inline char ID;
  /// The record of each site clone, by clone.
  std::map<const llvm::Function *, SiteClone> Clones;
  /// Keyed by value, not function: a later pass may replace a function by an
  /// alias or a constant.
  llvm::ValueMap<const llvm::Value *, std::string> Identities;
  /// Whether the module's sites were warned about as compiled without a
  /// profile action.
  bool WarnedWithoutProfile = false;
};

/// The body outlined from a site's function or kernel is named after it with
/// this suffix.
constexpr llvm::StringLiteral kBodySuffix = "_poseidon_body";
/// The copy made for one call of __poseidon_optimize is named
/// <function><infix><caller>.<ordinal>.
constexpr llvm::StringLiteral kPerCallInfix = ".poseidon.";
/// A site's canonical clone is named <body> after this prefix.
constexpr llvm::StringLiteral kCanonicalPrefix = "preprocess_";

/// The name LLVM's PGO knows `F` by (GlobalValue::getGlobalIdentifier):
/// its symbol, after the name of its source file and ';' when it has local
/// linkage as written. A static or anonymous kernel, which a device compile
/// gives external linkage (and with -fgpu-rdc a renamed symbol), is local.
std::string functionIdentity(const llvm::Function &F);
/// The same for the name `name` of `F`'s linkage.
std::string functionIdentity(const llvm::Function &F, llvm::StringRef name);
/// Records `identity` as that of the site whose body is `body`.
void noteSiteIdentity(SiteRecords &T, llvm::Function &body,
                      std::string identity);
/// Records the identity of each marker's target as written, before the
/// optimizer can change its linkage.
void recordMarkerIdentities(SiteRecords &T, llvm::Module &M);

/// The stem of a file named after `name`, a function name that is its own
/// identity: `name` itself, unless it is longer than a file name can hold; then
/// its first 160 characters, '~' and the GUID PGO keys a function of that name
/// by.
std::string profileNameStem(llvm::StringRef name);

/// A site's name for a diagnostic: the demangled function, and for a per-call
/// copy the call and caller it was made for.
std::string perCallSiteName(llvm::StringRef symbol);

/// The base name and the context of the function an Itanium name demangles to.
struct DemangledFunction {
  /// The unqualified name, and the scope it is declared in.
  std::string BaseName, Context;
};
/// The base name and context `mangled` demangles to; none if it names no
/// function.
std::optional<DemangledFunction> demangleFunction(llvm::StringRef mangled);

/// Records that `clone` was cut from `origin` for a call that passes it
/// `primalArgs`.
void noteSiteClone(SiteRecords &T, llvm::Function *clone,
                   llvm::Function *origin,
                   llvm::ArrayRef<llvm::Value *> primalArgs);
/// The body the site clone `clone` was cut from; null if it was not.
const llvm::Function *siteOriginOf(const SiteRecords &T,
                                   const llvm::Function &clone);
/// The stem of the profile file of the site cut from `body`.
std::string bodyProfileStem(const SiteRecords &T, const llvm::Function &body);
/// The file name of the profile of the site cut from `body`: its stem and
/// ".fpprofile".
std::string bodyProfileFile(const SiteRecords &T, const llvm::Function &body);
/// The stem of a file named after the site clone or body `F`, which is told
/// apart from a same-named function of another source file as its profile is.
std::string siteFileStem(const SiteRecords &T, const llvm::Function &F);
/// The ".fpprofile" file name of a site clone. The profiling run writes one
/// record per marked body, so every clone of one body reads the same file.
std::string siteProfileFile(const SiteRecords &T, const llvm::Function &clone);
/// The name -poseidon-pin gives a site by: the symbol its body was
/// cut from (an annotated kernel or host function, a marker's target, the copy
/// made for one __poseidon_optimize call), the one its profile is named after.
std::string siteKeyOfBody(const llvm::Function &body);
/// The key of the site `clone` was cut for, as siteKeyOfBody gives it.
std::string siteKey(const SiteRecords &T, const llvm::Function &clone);
/// The other names a site answers to when no other site of the module does:
/// the function a per-call copy was made of, and the unqualified name the
/// symbol demangles to.
llvm::SmallVector<std::string, 2> siteAliases(llvm::StringRef key);
/// The symbol of the function a site was cut from: its key without the
/// suffixes of a per-call copy.
llvm::StringRef siteSymbol(llvm::StringRef key);
/// Caller value of parameter i of a site clone, as recorded by
/// noteSiteClone; null when unknown.
llvm::Value *siteArg(const SiteRecords &T, const llvm::Function *clone,
                     unsigned i);
/// Whether noteSiteClone recorded F: a site clone, not a pricing clone.
bool isSiteClone(const SiteRecords &T, const llvm::Function *F);

} // namespace poseidon
#endif // POSEIDON_SITES_SITE_IDENTITY_H
