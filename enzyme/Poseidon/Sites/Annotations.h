//===- Annotations.h - functions and kernels marked POSEIDON_OPTIMIZE -----===//
// A module's annotated functions, the kernels that become sites through the
// device functions they call, and the attributes a site is marked with.
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
class Function;
class Module;
} // namespace llvm

namespace poseidon {

struct Options;

// The attribute stays on the kernel after its site has been lowered, and the
// pass reaches a module more than once, so a lowered kernel has to say so or
// its wrapper body is outlined again as a second site.
constexpr llvm::StringLiteral kSiteLowered = "poseidon-site-lowered";

// An annotated function that is not a site, diagnosed once.
constexpr llvm::StringLiteral kNotASite = "poseidon-not-a-site";

// The function a site's body was cut from (a host site's function, an
// annotated kernel), until finalizeModule has given it its debug info back.
constexpr llvm::StringLiteral kHostWrapper = "poseidon-site-wrapper";

// The annotated device functions whose kernels are sites, and the wrappers on
// the way to them: what foldDeviceSiteCallees folds into a site kernel the
// inliner left calling them.
constexpr llvm::StringLiteral kSiteCallee = "poseidon-site-callee";

std::string shortDouble(double v);
std::string displayName(const llvm::Function &F);
void diagnoseSite(llvm::Function &F, const llvm::Twine &message,
                  llvm::DiagnosticSeverity severity);
llvm::MapVector<llvm::Function *, std::optional<double>>
annotatedFunctions(llvm::Module &M);
void noteSiteTarget(const Options &O, const llvm::Function &F,
                    std::optional<double> tau);
void collectAnnotatedSites(
    const Options &O, llvm::Module &M,
    llvm::SmallVectorImpl<llvm::Function *> &out,
    llvm::SmallPtrSetImpl<llvm::Function *> *byRegex = nullptr,
    llvm::DenseMap<llvm::Function *, double> *tauOut = nullptr);
bool propagateDeviceSites(llvm::Module &M);
bool foldDeviceSiteCallees(llvm::Module &M);
void warnWithoutProfile(llvm::Function &K);
bool dropSiteAnnotations(llvm::Module &M);
bool leaveSitesAsWritten(const Options &O, llvm::Module &M);

} // namespace poseidon

#endif // POSEIDON_ANNOTATIONS_H
