//===- GemmRuntime.h - what the GEMM dispatch runtimes share --------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The error policy, the per-device state and the launch helpers of the
// runtimes that define the entry points of GemmDispatch.h.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_GEMM_RUNTIME_H
#define POSEIDON_GEMM_RUNTIME_H

#include "GemmDispatch.h"

#ifndef __HIP__
#include <cublas_v2.h>
#endif

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

// A runtime error leaves C unwritten or wrong, so it stops the program.
[[noreturn]] inline void poseidonGemmFatal(const char *call, const char *file,
                                           int line, const char *status) {
  fflush(stdout);
  fprintf(stderr, "Poseidon GEMM runtime: %.*s failed at %s:%d: %s\n",
          (int)strcspn(call, "("), call, file, line, status);
  abort();
}

#define POSEIDON_GEMM_CK(call)                                                 \
  do {                                                                         \
    cudaError_t e_ = (call);                                                   \
    if (e_ != cudaSuccess)                                                     \
      poseidonGemmFatal(#call, __FILE__, __LINE__, cudaGetErrorString(e_));    \
  } while (0)

#define POSEIDON_GEMM_CB(call)                                                 \
  do {                                                                         \
    cublasStatus_t s_ = (call);                                                \
    if (s_ != CUBLAS_STATUS_SUCCESS)                                           \
      poseidonGemmFatal(#call, __FILE__, __LINE__, cublasGetStatusString(s_)); \
  } while (0)

// The extents of a call of `entry`: a negative one stops the program. Whether
// there is a product to compute, all three extents being positive.
inline bool poseidonGemmExtents(const char *entry, int M, int Ncols, int K) {
  if (M >= 0 && Ncols >= 0 && K >= 0)
    return M > 0 && Ncols > 0 && K > 0;
  fflush(stdout);
  fprintf(stderr,
          "Poseidon GEMM runtime: %s with M = %d, Ncols = %d, K = %d: an "
          "extent is negative\n",
          entry, M, Ncols, K);
  abort();
}

// Calls launch(y0, n) for slabs of n <= 65535 blocks in y, the grid's y
// limit, numbered from y0, that together cover `blocks`.
template <class Launch> void poseidonForGridY(unsigned blocks, Launch launch) {
  constexpr unsigned kMaxGridY = 65535;
  for (unsigned y0 = 0; y0 < blocks; y0 += kMaxGridY)
    launch(y0, std::min(blocks - y0, kMaxGridY));
}

// C = beta * C, the product of an empty contraction, over M x Ncols; beta == 0
// writes zeros without reading C. Block (16, 16) over (fast, slow) axes.
static __global__ void poseidonScaleKernel(double *C, int M, int Ncols, int ldc,
                                           int cColMajor, double beta,
                                           unsigned y0) {
  const int fast = cColMajor ? M : Ncols, slow = cColMajor ? Ncols : M;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = (blockIdx.y + y0) * blockDim.y + threadIdx.y;
  if (i >= fast || j >= slow)
    return;
  double &c = C[(size_t)j * ldc + i];
  c = beta == 0.0 ? 0.0 : beta * c;
}

// The result of a call with K == 0 and M, Ncols > 0.
static void poseidonGemmEmptyK(double *C, int M, int Ncols, int ldc,
                               int cColMajor, double beta,
                               cudaStream_t stream) {
  const int fast = cColMajor ? M : Ncols, slow = cColMajor ? Ncols : M;
  poseidonForGridY((unsigned)((slow + 15) / 16), [&](unsigned y0, unsigned n) {
    poseidonScaleKernel<<<dim3((fast + 15) / 16, n), dim3(16, 16), 0, stream>>>(
        C, M, Ncols, ldc, cColMajor, beta, y0);
  });
  POSEIDON_GEMM_CK(cudaGetLastError());
}

// What a runtime keeps on one device. A call enqueues its work holding `lock`,
// so the handle's stream and the scratch stay the call's whatever other host
// threads do. `scratchFree` is recorded on the stream of every call that used
// the scratch, and the next call's stream waits for it, so calls on different
// streams take the scratch in turn.
struct PoseidonDevice {
  std::mutex lock;
  cublasHandle_t blas = nullptr;
  cudaEvent_t scratchFree = nullptr;
};

// The State (a PoseidonDevice) of the current device, one per runtime, created
// on first use and never destroyed.
template <class State> State &poseidonDevice() {
  static std::vector<State> *all = [] {
    int n = 0;
    POSEIDON_GEMM_CK(cudaGetDeviceCount(&n));
    return new std::vector<State>(n);
  }();
  int d = 0;
  POSEIDON_GEMM_CK(cudaGetDevice(&d));
  return (*all)[d];
}

// One call on the current device, for its scope: holds the device's lock
// and, with `scratch`, orders the call's use of the scratch after the previous
// call's.
template <class State> class PoseidonGemmCall {
public:
  PoseidonGemmCall(cudaStream_t stream, bool scratch)
      : dev(poseidonDevice<State>()), hold(dev.lock), stream(stream),
        scratch(scratch) {
    if (scratch && dev.scratchFree)
      POSEIDON_GEMM_CK(cudaStreamWaitEvent(stream, dev.scratchFree, 0));
  }
  // The device's handle, created on first use, on the call's stream.
  cublasHandle_t blas() {
    if (!dev.blas)
      POSEIDON_GEMM_CB(cublasCreate(&dev.blas));
    POSEIDON_GEMM_CB(cublasSetStream(dev.blas, stream));
    return dev.blas;
  }
  ~PoseidonGemmCall() {
    if (!scratch)
      return;
    if (!dev.scratchFree)
      POSEIDON_GEMM_CK(
          cudaEventCreateWithFlags(&dev.scratchFree, cudaEventDisableTiming));
    POSEIDON_GEMM_CK(cudaEventRecord(dev.scratchFree, stream));
  }
  PoseidonGemmCall(const PoseidonGemmCall &) = delete;
  PoseidonGemmCall &operator=(const PoseidonGemmCall &) = delete;

  State &dev;

private:
  std::lock_guard<std::mutex> hold;
  cudaStream_t stream;
  bool scratch;
};

// Waits until no call enqueued before uses the scratch of `dev`, which may
// then be freed.
inline void poseidonDrainScratch(PoseidonDevice &dev) {
  if (dev.scratchFree)
    POSEIDON_GEMM_CK(cudaEventSynchronize(dev.scratchFree));
}

// Grows the scratch buffer *p of *cap bytes of `dev` to at least `bytes`.
inline void poseidonReserve(PoseidonDevice &dev, void **p, size_t *cap,
                            size_t bytes) {
  if (*cap >= bytes)
    return;
  if (*p) {
    poseidonDrainScratch(dev);
    POSEIDON_GEMM_CK(cudaFree(*p));
  }
  POSEIDON_GEMM_CK(cudaMalloc(p, bytes));
  *cap = bytes;
}

#endif // POSEIDON_GEMM_RUNTIME_H
