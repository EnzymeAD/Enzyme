//=- HostDispatch.h - host-side GEMM library dispatch ---------------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A raised GEMM materialized as a host library call: the device sub-compilation
// records a descriptor in the manifest of its translation unit in the Poseidon
// cache dir, and the host sub-compilation (a separate cc1 process) reads it and
// rewrites the kernel launch into a call to the runtime helper.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_HOST_DISPATCH_H
#define POSEIDON_HOST_DISPATCH_H

#include "IR/Utils.h"
#include "Matmul.h"
#include "runtime/ozaki/OzakiII.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/PassManager.h"

#include <string>
#include <vector>

namespace llvm {
class Function;
class Module;
class Value;
} // namespace llvm

namespace poseidon {

struct Options;
class Session;

// The host-dispatch descriptor of a GEMM in an optimized body. Operand roles
// and runtime-dimension factors name parameters of the body until the
// descriptor is written, and launch arguments of the wrapper kernel after.
struct GemmBodyNote {
  int cParam = -1, aParam = -1, bParam = -1;
  // Byte offsets of the three operand pointers inside their parameter, when
  // the parameter is a by-value aggregate; -1 when it is the pointer itself.
  int64_t cOffset = -1, aOffset = -1, bOffset = -1;
  unsigned N = 0; // square dim (== gK); square fast path
  // descriptorScheme and descriptorModulusField of the selected variant.
  DispatchScheme scheme = DispatchScheme::OzakiII;
  unsigned numModuli = kOzakiIIMaxModuli;
  bool standalone = false; // body is a pure GEMM (only the C store)
  bool valid = false;
  // General GEMM geometry, gM x gNcols x gK with per-operand leading dims (in
  // elements) and layouts; a square row-major NN GEMM takes the square entry,
  // anything else the _ex entry.
  unsigned gM = 0, gNcols = 0, gK = 0;
  unsigned lda = 0, ldb = 0, ldc = 0; // leading dims, in elements
  bool aColMajor = false, bColMajor = false, cColMajor = false;

  // Deploy extents recovered from the launch geometry: a dimension that does
  // not track matrix order has no sound rescale from a surrogate profile. With
  // `launchCrop` set, gM/gNcols hold the deploy extents from the constants
  // gating the C store, and cropMAxis/cropNAxis name the thread axis each index
  // runs along so the host stub takes extent = min(crop, gridDim * blockDim).
  bool launchCrop = false;
  int cropMAxis = -1, cropNAxis = -1; // 0/1/2 = thread axis x/y/z

  // Runtime geometry (Origin::HostGemmLoopNest): with `runtimeDims` set the
  // host stub evaluates each dimension from the launch arguments (RuntimeDim),
  // so a build profiled at one problem size runs at another; the six integers
  // above are then the profile-scale values. `beta` = 1 accumulates into C. A
  // descriptor without these tokens reads back as false / 0.0.
  using RtDim = RuntimeDim;
  bool runtimeDims = false;
  RtDim rM, rNcols, rK, rlda, rldb, rldc;
  double beta = 0.0;
  // Launch values the recognition assumed equal (RuntimeGemmHandle); the stub
  // dispatches only when every pair agrees and launches the kernel otherwise.
  llvm::SmallVector<std::pair<LaunchRef, LaunchRef>, 2> assumedEqual;
};

// Compute the GemmBodyNote for matmul m inside optimized body F; false
// (note.valid = false) if anything is unsupported.
bool computeGemmBodyNote(const Options &O, llvm::Function &F,
                         const AbstractMatmul &m, GemmBodyNote &note);

// Record and retrieve the note of an optimized body clone; getGemmBody returns
// false when none was recorded.
void noteGemmBody(Session &S, const llvm::Function *body,
                  const GemmBodyNote &n);
bool getGemmBody(const Session &S, const llvm::Function *body,
                 GemmBodyNote &out);

// Fused-case device transform: replace the GEMM result where the epilogue
// consumes it with a load from the output buffer filled by the prepended host
// dispatch, so the reduction loop dies. Paired with the fused host-side
// prepend. F stores a value that reaches the product to an argument, as it
// does when computeGemmBodyNote gave it a valid note that is not standalone.
void fissionGemmForDispatch(llvm::Function &F, const AbstractMatmul &m);

// Map body params to wrapper launch-arg indices via primalArgs and record the
// wrapper's host dispatch descriptor.
void writeGemmDescriptor(Session &S, llvm::Function &wrapper,
                         llvm::ArrayRef<llvm::Value *> primalArgs,
                         const GemmBodyNote &n);

struct PendingGemm {
  std::string wrapperName;
  std::vector<int> paramToArg; // body-param index -> wrapper launch-arg index
};

// For a joint solve, whose notes exist only after materialization:
// addPendingGemmDispatch records the body-parameter to wrapper-argument map
// when the site is split, and flushPendingGemmDispatches writes the
// descriptors afterwards.
void addPendingGemmDispatch(Session &S, const llvm::Function *body,
                            llvm::Function &wrapper,
                            llvm::ArrayRef<llvm::Value *> primalArgs);
void flushPendingGemmDispatches(Session &S);

// Rewrite each __device_stub__ launch stub that a host dispatch descriptor in
// M's manifest matches; must run on the host module before inlining.
bool rewriteGemmStubBodies(const Options &O, llvm::Module &M);

// Rewrite the launch in each launch stub that a profile generation descriptor
// in M's manifest matches into a call of __poseidon_launch_profiled, which
// allocates, zeroes and seeds the shadow buffers the profiling kernel takes and
// launches it. Host module only.
bool rewriteProfileStubBodies(const Options &O, llvm::Module &M);

// PipelineStart pass wrapper for rewriteGemmStubBodies (host module only).
class HostStubPass : public PassParent<HostStubPass> {
public:
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);
};

} // namespace poseidon
#endif // POSEIDON_HOST_DISPATCH_H
