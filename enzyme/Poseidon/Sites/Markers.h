//===- Markers.h - marker calls and the sites they defer ------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

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

/// Whether a called function is the Poseidon annotation, under either of its
/// names, __poseidon_fp_optimize and __enzyme_fp_optimize.
bool isMarkerCall(llvm::StringRef calleeName);

/// The reverse-mode request profile generation makes of a site, which Enzyme
/// lowers; LLVM uniques a second declaration's name with a suffix.
constexpr llvm::StringLiteral kAutodiffRequest = "__enzyme_autodiff_poseidon";

/// Reports `message` as a warning at the marker call `CI`.
void warn(llvm::CallBase *CI, const llvm::Twine &message);
/// The function the marker call `CI` names, or null, without a diagnostic.
llvm::Function *markerTargetOrNull(llvm::CallInst *CI);
/// The function the marker call `CI` names; an error if it names none or one
/// without a body.
llvm::Expected<llvm::Function *> markerTarget(llvm::CallInst *CI);
/// The argument of a marker call or an Enzyme request that says what the
/// argument after it is, as enzyme_dup does.
llvm::Value *activityArg(llvm::LLVMContext &C, llvm::StringRef name);

/// Whether `V`, an argument of a marker call or of a call through an identity,
/// is the activity marker `name`.
bool isActivityMarker(llvm::Value *V, llvm::StringRef name);

/// The identity of poseidon.h a call of __poseidon_optimize goes through: the
/// plain one when fn takes the arguments as written, the explicit one when it
/// does not. A marker call goes through none.
enum class Identity { Plain, Explicit, None };

/// The names of the identities, without and with an accuracy target. The
/// explicit ones are declared and never defined, so that a build without the
/// plugin does not link a call fn does not take.
constexpr llvm::StringLiteral kPlainIdentity = "__poseidon_site";
constexpr llvm::StringLiteral kPlainIdentityTau = "__poseidon_site_tau";
constexpr llvm::StringLiteral kExplicitIdentity = "__poseidon_site_explicit";
constexpr llvm::StringLiteral kExplicitIdentityTau =
    "__poseidon_site_explicit_tau";

constexpr llvm::StringLiteral kMarkerName = "__poseidon_fp_optimize";

/// How the arguments of a call of `fn` are read: as fn's own (automatic), or
/// with activity markers and shadows, as the marker reads them (explicit).
enum class ArgumentList { Automatic, Explicit };

/// What a call's list is, and of an automatic list, fn's own arguments and the
/// accuracy target its poseidon_tau gives, if it has one.
struct CallArguments {
  ArgumentList kind;
  llvm::SmallVector<llvm::Value *, 8> own;
  std::optional<double> tau;
};

/// The arguments `call` passes for the parameters of `fn` (all of a call
/// through an identity, those after the function of a marker call) are an
/// explicit list when one of them is an activity marker (an enzyme_* name, or a
/// poseidon_* name other than poseidon_tau), and an automatic list otherwise.
/// poseidon_tau and the constant after it give the accuracy target of a marker
/// call; the rest of an automatic list are fn's own arguments. An explicit list
/// is read by parseMarker. An error at `call` when the list cannot be read as
/// either.
llvm::Expected<CallArguments> readArgumentList(llvm::CallBase *call,
                                               const llvm::Function &fn,
                                               Identity identity);

/// Whether `tau` is an accuracy target: positive and finite.
bool isAccuracyTarget(double tau);

/// The accuracy target `value` gives a call of `fn`; an error at `call` unless
/// it is a floating-point constant that isAccuracyTarget.
llvm::Expected<double> readTarget(llvm::CallBase *call, llvm::Value *value,
                                  const llvm::Function &fn);

/// Inlines into the deferred `body` the functions Enzyme generated for it;
/// how many calls were inlined.
unsigned foldEnzymeGenerated(llvm::Function &body);
llvm::Function *
asLeftByPipeline(llvm::Function &body,
                 llvm::DenseMap<llvm::Function *, llvm::Function *> &copies);
/// Whether `F`, or a function it calls, is still an __enzyme_* request.
bool callsEnzymeRequest(llvm::Function &F);
/// Whether the marker call `CI` is the late run's, its body still being a
/// derivative request; marks the body so that both runs agree.
bool deferSite(const Options &O, llvm::CallInst *CI);
/// The marker calls of the module, in module order.
void collectMarkerCalls(llvm::Module &M,
                        llvm::SmallVectorImpl<llvm::CallInst *> &out);
/// Whether deferSite left the marked body `F` to the late run.
bool isDeferredBody(const llvm::Function &F);

/// Site ids number the profiled clones of one module and arm the
/// condition-number perturbation one site at a time, so the late run has to
/// continue the early run's numbering rather than restart it. Carried on the
/// module because the two runs are different pass instances.
constexpr llvm::StringLiteral kNextSiteIdMD = "poseidon.next.site.id";

/// The next site id of `M`, which the early run leaves for the late one.
unsigned getNextSiteId(llvm::Module &M);
/// Records `next` as the next site id of `M`.
void setNextSiteId(llvm::Module &M, unsigned next);

/// What the marker call says about one site: the caller value behind each of
/// the site's parameters, the accuracy target, and which marker arguments are
/// Poseidon's own and must not be forwarded to the derivative request.
struct MarkerArgs {
  llvm::SmallVector<llvm::Value *, 8> primalArgs;
  llvm::SmallVector<bool, 16> forward;
  /// The pointers marked poseidon_shadow, whose shadows profileSite makes: the
  /// marker argument naming the activity, the pointer's, and its parameter.
  struct MadeShadow {
    unsigned activity, arg, param;
  };
  llvm::SmallVector<MadeShadow, 4> madeShadows;
  double errTol = 0.0;
};

/// What the marker call `CI` of body `F` says; an error at `CI` if the marker
/// cannot be read.
llvm::Expected<MarkerArgs> parseMarker(llvm::CallInst *CI, llvm::Function *F);

} // namespace poseidon

#endif // POSEIDON_MARKERS_H
