//===- Markers.h - marker calls and the sites they defer ------------------===//
// A marker call names a site's body and the activity of its arguments; a body
// that still holds an Enzyme request is deferred to the late run.
#ifndef POSEIDON_MARKERS_H
#define POSEIDON_MARKERS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

namespace llvm {
class CallBase;
class CallInst;
class Function;
class Module;
class Value;
} // namespace llvm

namespace poseidon {

struct Options;

// Whether a called function is the Poseidon annotation.
bool isMarkerCall(llvm::StringRef calleeName);

void warn(llvm::CallBase *CI, const llvm::Twine &message);
void fail(llvm::CallBase *CI, const llvm::Twine &message);
llvm::Function *calleeOf(llvm::CallInst *CI);
llvm::Function *markerTargetOrNull(llvm::CallInst *CI);
llvm::Function *markerTarget(llvm::CallInst *CI);

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

bool parseMarker(llvm::CallInst *CI, llvm::Function *F, MarkerArgs &out);

} // namespace poseidon

#endif // POSEIDON_MARKERS_H
