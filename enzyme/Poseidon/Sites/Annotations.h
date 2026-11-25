//===- Annotations.h - functions and kernels marked POSEIDON_OPTIMIZE -----===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
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

#ifndef POSEIDON_ANNOTATIONS_H
#define POSEIDON_ANNOTATIONS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/DiagnosticInfo.h"

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
constexpr llvm::StringLiteral kHostWrapper = "poseidon-site-wrapper";

/// The annotated device functions whose kernels are sites, and the wrappers on
/// the way to them.
constexpr llvm::StringLiteral kSiteCallee = "poseidon-site-callee";

/// The outlined body of a host site. Its value is the site's accuracy target,
/// empty when the annotation carried none.
constexpr llvm::StringLiteral kHostBody = "poseidon-site-body";

/// What a site compiled without a profile action is warned with, before what
/// becomes of it.
constexpr llvm::StringLiteral kNoProfileAction =
    "a Poseidon site was compiled without -poseidon-profile-generate or "
    "-poseidon-profile-use=<dir>.";

void diagnoseSite(llvm::Function &F, const llvm::Twine &message,
                  llvm::DiagnosticSeverity severity);
llvm::MapVector<llvm::Function *, std::optional<double>>
annotatedFunctions(llvm::Module &M);
void noteSiteTarget(const Options &O, const llvm::Function &F,
                    std::optional<double> tau);
bool propagateDeviceSites(const Options &O, llvm::Module &M);
bool foldDeviceSiteCallees(const Options &O, llvm::Module &M);
/// An error at `call`, a __poseidon_optimize of the annotated function `name`.
void reportAnnotatedCall(llvm::CallBase *call, llvm::StringRef name);

/// Warns once per module about every annotated kernel of a compile without a
/// profile action.
void warnSitesWithoutProfile(const Options &O, SiteRecords &T, llvm::Module &M);
bool dropSiteAnnotations(llvm::Module &M);
bool leaveSitesAsWritten(const Options &O, SiteRecords &T, llvm::Module &M);

/// Why `F` cannot be a host site, empty if it can.
std::string hostSiteRefusal(llvm::Function &F);
bool outlineHostSites(const Options &O, SiteRecords &T, llvm::Module &M);
/// The calls whose callee is `F` itself.
llvm::SmallVector<llvm::CallBase *, 4> callsOf(llvm::Value &F);
/// The host body `B` handed back to the optimizer: out of llvm.compiler.used
/// and without the mark its clones would inherit.
void releaseHostBody(llvm::Function &B);

void markOutlineTransparent(llvm::Function *body);
llvm::CallInst *outlineKernelBody(SiteRecords &T, llvm::Function &K,
                                  llvm::Function *&bodyOut);
void inlineKernelBody(llvm::CallInst *call, llvm::Function *body);

} // namespace poseidon

#endif // POSEIDON_ANNOTATIONS_H
