//===- Annotations.h - functions and kernels marked POSEIDON_OPTIMIZE -----===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A module's annotated functions, the kernels that become sites through the
// device functions they call, the attributes a site is marked with, and the
// body a site's computation is outlined into.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_SITES_ANNOTATIONS_H
#define POSEIDON_SITES_ANNOTATIONS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <string>

namespace llvm {
class CallBase;
class CallInst;
class Function;
class Module;
class Value;
} // namespace llvm

namespace poseidon {

struct Options;
struct SiteRecords;

/// The attribute stays on the kernel after its site has been lowered, and the
/// pass reaches a module more than once, so a lowered kernel has to say so or
/// its wrapper body is outlined again as a second site.
constexpr llvm::StringLiteral kSiteLowered = "poseidon-site-lowered";

/// An annotated function that is not a site, diagnosed once.
constexpr llvm::StringLiteral kNotASite = "poseidon-not-a-site";

/// The function a site's body was cut from (a host site's function, an
/// annotated kernel), until finalizeModule has given it its debug info back.
constexpr llvm::StringLiteral kSiteWrapper = "poseidon-site-wrapper";

/// The annotated device functions whose kernels are sites, and the wrappers on
/// the way to them.
constexpr llvm::StringLiteral kSiteCallee = "poseidon-site-callee";

/// The outlined body of a host site. Its value is the site's accuracy target,
/// empty when the annotation carried none.
constexpr llvm::StringLiteral kSiteBody = "poseidon-site-body";

/// What a site compiled without a profile action is warned with.
constexpr llvm::StringLiteral kNoProfileAction =
    "a Poseidon site was compiled without -poseidon-profile-generate or "
    "-poseidon-profile-use=<dir>, so it is left as written; compile with one "
    "of the two to profile or optimize it";

/// Warns with `message` at the annotated function `F`, once, and marks it as no
/// site.
void warnSite(llvm::Function &F, const llvm::Twine &message);
/// The functions the module's Poseidon annotations name, in module order, with
/// the accuracy target each carries; one whose annotation cannot be read is
/// left out.
llvm::MapVector<llvm::Function *, std::optional<double>>
annotatedFunctions(llvm::Module &M);
/// The mistakes in the module's Poseidon annotations, none in the host half of
/// a CUDA or HIP compilation, which holds no site.
llvm::Error checkSiteAnnotations(llvm::Module &M);
/// Prints, under -poseidon-print, the accuracy target site `F` carries.
void noteSiteTarget(const Options &O, const llvm::Function &F,
                    std::optional<double> tau);
/// Makes each kernel that calls an annotated device function a site, as if
/// it carried the annotation itself.
llvm::Expected<bool> propagateDeviceSites(const Options &O, llvm::Module &M);
/// Inlines into each device site kernel what it calls besides the math
/// library, warning about a call that cannot be folded.
bool foldDeviceSiteCallees(const Options &O, llvm::Module &M);
/// The error at `call`, a __poseidon_optimize of the annotated function `name`.
llvm::Error annotatedCallError(llvm::CallBase &call, llvm::StringRef name);

/// Warns once per module about every annotated kernel of a compile without a
/// profile action.
void warnSitesWithoutProfile(const Options &O, SiteRecords &T, llvm::Module &M);
/// Removes the module's POSEIDON_OPTIMIZE annotations and what only they
/// referred to; whether any was removed.
bool dropSiteAnnotations(llvm::Module &M);
/// For a compile without a profile action: warns about the sites once and
/// removes their annotations and marks, so the module compiles as written.
bool leaveSitesAsWritten(const Options &O, SiteRecords &T, llvm::Module &M);

/// Why `F` cannot be a host site; none if it can.
std::optional<std::string> hostSiteRefusal(llvm::Function &F);
/// Makes each annotated host function a wrapper around an outlined body that
/// carries the site's accuracy target, recording the site's identity in `T`.
llvm::Expected<bool> outlineHostSites(const Options &O, SiteRecords &T,
                                      llvm::Module &M);
/// The calls whose callee is `F` itself.
llvm::SmallVector<llvm::CallBase *, 4> callsOf(llvm::Value &F);
/// The host body `B` handed back to the optimizer: out of llvm.compiler.used
/// and without the mark its clones would inherit.
void releaseHostBody(llvm::Function &B);

/// Lets the inliner fold `body` back into its caller.
void markOutlineTransparent(llvm::Function *body);
/// Moves the code of kernel `K` into a new body, recorded in `T` with K's
/// identity, that K calls; that call, or an error, before any change, if K
/// cannot be a site.
llvm::Expected<llvm::CallInst *>
outlineKernelBody(SiteRecords &T, llvm::Function &K, llvm::Function *&bodyOut);
/// Inlines `body` back at `call`, undoing outlineKernelBody.
void inlineKernelBody(llvm::CallInst *call, llvm::Function *body);

} // namespace poseidon

#endif // POSEIDON_SITES_ANNOTATIONS_H
