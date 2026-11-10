//===- SiteIdentity.h - what a site clone is and was cut from ------------===//
//
// A site is solved on a clone of its body. Per clone, the body it was cut from
// and the caller values of its parameters, and the names its profile and its
// -poseidon-apply-rewrites items are found by.
//===---------------------------------------------------------------------===//
#ifndef POSEIDON_SITE_IDENTITY_H
#define POSEIDON_SITE_IDENTITY_H

#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
class Function;
class Value;
} // namespace llvm

namespace poseidon {

class Session;

void noteSiteOrigin(Session &S, llvm::Function *clone, llvm::Function *orig);
const llvm::Function *siteOriginOf(const Session &S,
                                   const llvm::Function &clone);
// The ".fpprofile" stem of a site clone. The profiling run writes one record
// per marked BODY, so every clone of one body reads the same file.
std::string siteProfileStem(const Session &S, const llvm::Function &clone);
// The name -poseidon-apply-rewrites gives a site by: the symbol its body was
// cut from (an annotated kernel or host function, a marker's target, the copy
// made for one __poseidon_optimize call), the one its profile is named after.
std::string siteKeyOfBody(const llvm::Function &body);
std::string siteKey(const Session &S, const llvm::Function &clone);
// The other names a site answers to when no other site of the module does:
// the function a per-call copy was made of, and the unqualified name the
// symbol demangles to.
llvm::SmallVector<std::string, 2> siteAliases(llvm::StringRef key);
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
