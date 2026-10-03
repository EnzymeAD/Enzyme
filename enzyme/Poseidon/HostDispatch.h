//===- HostDispatch.h - host-side GEMM library dispatch ------------------===//
//
// A raised GEMM materialized as a host library call: the device sub-compilation
// records a descriptor to a side-car file in the Poseidon cache dir, and the
// host sub-compilation (a separate cc1 process) reads it and rewrites the
// kernel launch into a call to the runtime helper.
//===---------------------------------------------------------------------===//
#ifndef POSEIDON_HOST_DISPATCH_H
#define POSEIDON_HOST_DISPATCH_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/CommandLine.h"

#include "OzakiII.h"
#include "Utils.h"

namespace llvm {
class Function;
class Module;
class Value;
} // namespace llvm

namespace poseidon {

struct AbstractMatmul;

// GemmBodyNote roles are the optimized body's parameter indices;
// optimizeSiteBody maps them to the wrapper kernel's launch-arg indices before
// writing the descriptor. All host runtimes take the identical calling
// convention, so the scheme selects only the callee name:
//   OzakiII -> __poseidon_ozaki_dgemm(_ex): INT8 residue/CRT, accuracy set by
//              num_moduli;
//   Tcec    -> __poseidon_tcec_dgemm(_ex): FP16 tensor cores + error
//              correction, F32-class;
//   Direct  -> __poseidon_direct_dgemm(_ex): one rounding of each operand to
//              FP16/BF16/TF32/FP32 (numModuli = 1/2/3/4), FP32 accumulator.
enum class DispatchScheme { OzakiII, Tcec, Direct };

// Descriptor file suffix for host GEMM dispatch.
constexpr llvm::StringLiteral kOzDispatchScheme = ".ozdispatch";

// Descriptor file suffix for profile generation. The device compilation writes
// one per annotated kernel; the host compilation of the same clang invocation
// reads it to rewrite that kernel's launch stub.
//   <kernel> <site id> <original arg count> <pointer arg count>
//            <pointer arg index>...
//            <seed width in bytes of that pointer, 0 if it is not an output>...
constexpr llvm::StringLiteral kProfGenScheme = ".profgen";

struct GemmBodyNote {
  int cParam = -1, aParam = -1, bParam = -1;
  unsigned N = 0; // square dim (== gK); square fast path
  DispatchScheme scheme = DispatchScheme::OzakiII;
  // Scheme parameter: Ozaki-II num_moduli, or the TCEC compute mode.
  unsigned numModuli = kOzakiIIMaxModuli;
  bool standalone = false; // body is a pure GEMM (only the C store)
  bool valid = false;
  // General GEMM geometry, gM x gNcols x gK with per-operand leading dims (in
  // elements) and layouts; a square row-major NN GEMM takes the square entry,
  // anything else the _ex entry.
  unsigned gM = 0, gNcols = 0, gK = 0;
  unsigned lda = 0, ldb = 0, ldc = 0; // leading dims, in ELEMENTS
  bool aColMajor = false, bColMajor = false, cColMajor = false;

  // Deploy extents recovered from the launch geometry: a dimension that does
  // not track matrix order has no sound rescale from a surrogate profile. With
  // `launchCrop` set, gM/gNcols hold the deploy extents from the constants
  // gating the C store, and cropMAxis/cropNAxis name the thread axis each index
  // runs along so the host stub takes extent = min(crop, gridDim * blockDim).
  bool launchCrop = false;
  int cropMAxis = -1, cropNAxis = -1; // 0/1/2 = thread axis x/y/z

  // Runtime geometry (Origin::HostGemmLoopNest). A finite-element apply's
  // product dimensions are kernel ARGUMENTS (M = nDofs, K = d*numPoints,
  // Ncols = d*NE), not compile-time constants, so the six integers above cannot
  // express them: baking the profile-scale numbers in would deploy the
  // surrogate's mesh size, and baking the deploy-scale numbers in would require
  // the compiler to know the mesh. With `runtimeDims` set the host-side stub
  // rewrite EVALUATES each dimension from the launch arguments as
  // `mul*arg[param] + add`, which is what makes profile-small/deploy-large hold
  // for this class.
  //
  // `beta` is part of the descriptor for the same reason: a partial-assembly
  // reduce accumulates into y (`y(i,q,e) += sum`), and a dispatch emitted with
  // a hard-wired beta = 0 would drop the caller's accumulator. Descriptors
  // written before these tokens existed carry neither and read back as false /
  // 0.0.
  struct RtDim {
    int param = -1; // WRAPPER launch-argument index (-1 = use the constant)
    int64_t mul = 1;
    int64_t add = 0;
  };
  bool runtimeDims = false;
  RtDim rM, rNcols, rK, rlda, rldb, rldc;
  double beta = 0.0;
};

// Walk GEP/bitcast/addrspacecast chains back to a Function Argument (-1 if
// none).
int traceToArgIndex(const llvm::Value *v);

// Compute the GemmBodyNote for matmul m inside optimized body F; false
// (note.valid = false) if anything is unsupported.
bool computeGemmBodyNote(llvm::Function &F, const AbstractMatmul &m,
                         GemmBodyNote &note);

// Process-local (device cc1) stash: fpOptimize records the note keyed by the
// optimized body clone; optimizeSiteBody retrieves it after fpOptimize returns.
void noteGemmBody(const llvm::Function *body, const GemmBodyNote &n);
bool getGemmBody(const llvm::Function *body, GemmBodyNote &out);

// Fused-case device transform: replace the GEMM result where the epilogue
// consumes it with a load from the output buffer filled by the prepended host
// dispatch, so the reduction loop dies. Paired with the fused host-side
// prepend.
bool fissionGemmForDispatch(llvm::Function &F, const AbstractMatmul &m);

// Map body params to wrapper launch-arg indices via primalArgs and append a
// descriptor line to <cacheDir>/<wrapper>.ozdispatch.
void writeGemmDescriptor(llvm::Function &wrapper,
                         llvm::ArrayRef<llvm::Value *> primalArgs,
                         const GemmBodyNote &n, llvm::StringRef cacheDir);

// Joint-DP path: the note is produced only in the post-Enzyme joint
// materialize, so addPendingGemmDispatch captures the body-param to wrapper-arg
// mapping at split time and flushPendingGemmDispatches emits the descriptors
// afterwards.
void addPendingGemmDispatch(const llvm::Function *body, llvm::Function &wrapper,
                            llvm::ArrayRef<llvm::Value *> primalArgs,
                            llvm::StringRef cacheDir);
void flushPendingGemmDispatches();

// Read *.ozdispatch in cacheDir and rewrite each matching __device_stub__
// launch stub; must run on the host module before inlining.
bool rewriteGemmStubBodies(llvm::Module &M, llvm::StringRef cacheDir);

// Read *.profgen in cacheDir and rewrite each matching launch stub into a call
// of __poseidon_launch_profiled, which allocates, zeroes and seeds the shadow
// buffers the profiling kernel takes and launches it. Host module only.
bool rewriteProfileStubBodies(llvm::Module &M, llvm::StringRef cacheDir);

// PipelineStart pass wrapper for rewriteGemmStubBodies (host module only).
class HostStubPass : public PassParent<HostStubPass> {
public:
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);
};

} // namespace poseidon
#endif // POSEIDON_HOST_DISPATCH_H
