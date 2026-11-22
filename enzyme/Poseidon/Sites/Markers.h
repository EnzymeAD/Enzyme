//=- Markers.h - marker calls and the sites they defer ----------------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A marker call names a site's body and the activity of its arguments; a body
// that still holds an Enzyme request is deferred to the late run.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_MARKERS_H
#define POSEIDON_MARKERS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <optional>

namespace llvm {
class CallBase;
class CallInst;
class Function;
class LLVMContext;
class Module;
class Value;
} // namespace llvm

namespace poseidon {

struct Options;

// Whether a called function is the Poseidon annotation, under either of its
// names, __poseidon_fp_optimize and __enzyme_fp_optimize.
bool isMarkerCall(llvm::StringRef calleeName);

void warn(llvm::CallBase *CI, const llvm::Twine &message);
void fail(llvm::CallBase *CI, const llvm::Twine &message);
llvm::Function *calleeOf(llvm::CallInst *CI);
llvm::Function *markerTargetOrNull(llvm::CallInst *CI);
llvm::Function *markerTarget(llvm::CallInst *CI);
// The argument of a marker call or an Enzyme request that says what the
// argument after it is, as enzyme_dup does.
llvm::Value *activityArg(llvm::LLVMContext &C, llvm::StringRef name);

// Whether `V`, an argument of a marker call or of a call through an identity,
// is the activity marker `name`.
bool isActivityMarker(llvm::Value *V, llvm::StringRef name);

// The identity of poseidon.h a call of __poseidon_optimize goes through: the
// plain one when fn takes the arguments as written, the explicit one when it
// does not. A marker call goes through none.
enum class Identity { Plain, Explicit, None };

// The names of the identities, without and with an accuracy target. The
// explicit ones are declared and never defined, so that a build without the
// plugin does not link a call fn does not take.
constexpr llvm::StringLiteral kPlainIdentity = "__poseidon_site";
constexpr llvm::StringLiteral kPlainIdentityTau = "__poseidon_site_tau";
constexpr llvm::StringLiteral kExplicitIdentity = "__poseidon_site_explicit";
constexpr llvm::StringLiteral kExplicitIdentityTau =
    "__poseidon_site_explicit_tau";

constexpr llvm::StringLiteral kMarkerName = "__poseidon_fp_optimize";

// How the arguments of a call of `fn` are read: as fn's own (automatic), or
// with activity markers and shadows, as the marker reads them (explicit).
enum class ArgumentList { Automatic, Explicit };

// The one rule for every call form: the arguments `call` passes for the
// parameters of `fn` (all of a call through an identity, those after the
// function of a marker call) are an explicit list when one of them is an
// activity marker (an enzyme_* or poseidon_* name), and an automatic list
// otherwise. None, with an error at `call`, when the list is both or neither:
// a list `identity` says fn takes as written that carries a marker, a list it
// says fn does not take that carries none, or an automatic list of a marker
// call whose length is not fn's number of parameters.
std::optional<ArgumentList> readArgumentList(llvm::CallBase *call,
                                             const llvm::Function &fn,
                                             Identity identity);

// Inlines the calls of `F` that `pick` selects, in instruction order, and again
// in what that inlined, for a bounded number of rounds; returns how many calls
// it inlined.
unsigned inlineCallsWhile(llvm::Function &F,
                          llvm::function_ref<bool(llvm::CallBase &)> pick);
unsigned foldEnzymeGenerated(llvm::Function &body);
llvm::Function *
asLeftByPipeline(llvm::Function &body,
                 llvm::DenseMap<llvm::Function *, llvm::Function *> &copies);
bool callsEnzymeRequest(llvm::Function &F);
bool deferSite(const Options &O, llvm::CallInst *CI);
// The marker calls of the module, in module order.
void collectMarkerCalls(llvm::Module &M,
                        llvm::SmallVectorImpl<llvm::CallInst *> &out);
void collectDeferredMarkers(llvm::Module &M,
                            llvm::SmallVectorImpl<llvm::CallInst *> &out);

// Site ids number the profiled clones of one module and arm the
// condition-number perturbation one site at a time, so the late run has to
// continue the early run's numbering rather than restart it. Carried on the
// module because the two runs are different pass instances.
constexpr llvm::StringLiteral kNextSiteIdMD = "poseidon.next.site.id";

unsigned getNextSiteId(llvm::Module &M);
void setNextSiteId(llvm::Module &M, unsigned next);

// What the marker call says about one site: the caller value behind each of the
// site's parameters, the accuracy target, and which marker arguments are
// Poseidon's own and must not be forwarded to the derivative request.
struct MarkerArgs {
  llvm::SmallVector<llvm::Value *, 8> primalArgs;
  llvm::SmallVector<bool, 16> forward;
  // The pointers marked poseidon_shadow, whose shadows profileSite makes: the
  // marker argument naming the activity, the pointer's, and its parameter.
  struct MadeShadow {
    unsigned activity, arg, param;
  };
  llvm::SmallVector<MadeShadow, 4> madeShadows;
  double errTol = 0.0;
};

// What the marker call `CI` of body `F` says; none, with an error at `CI`, if
// the marker cannot be read.
std::optional<MarkerArgs> parseMarker(llvm::CallInst *CI, llvm::Function *F);

} // namespace poseidon

#endif // POSEIDON_MARKERS_H
