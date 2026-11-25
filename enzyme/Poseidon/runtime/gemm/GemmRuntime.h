//===- GemmRuntime.h - what the GEMM runtimes share -----------------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// What the runtimes of the entries of GemmDispatch.inc share: the error
// policy, the per-device state a call holds, the body every entry runs, the
// tiling of launches, the column-major form of a call of the library, and the
// operand repacking of the runtimes that run an FP64 product through a
// reduced-precision library call (direct, TCEC).
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_RUNTIME_GEMM_GEMM_RUNTIME_H
#define POSEIDON_RUNTIME_GEMM_GEMM_RUNTIME_H

#include "GemmDispatch.h"

#ifndef __HIP__
#include <cublas_v2.h>
#endif

#include <algorithm>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <type_traits>
#include <vector>

namespace poseidon {
namespace gemm {

//===----------------------------------------------------------------------===//
// Errors
//===----------------------------------------------------------------------===//

// An error leaves C unwritten or wrong, so it stops the program, with the
// message after "[poseidon] ".
[[noreturn]] __attribute__((format(printf, 1, 2))) inline void
fail(const char *format, ...) {
  fflush(stdout);
  fputs("[poseidon] ", stderr);
  va_list args;
  va_start(args, format);
  vfprintf(stderr, format, args);
  va_end(args);
  fputc('\n', stderr);
  abort();
}

// A call of the GPU runtime or the library that returned `status`.
[[noreturn]] inline void failCall(const char *call, const char *file, int line,
                                  const char *status) {
  fail("%.*s failed at %s:%d: %s", (int)strcspn(call, "("), call, file, line,
       status);
}

#define POSEIDON_GEMM_CK(call)                                                 \
  do {                                                                         \
    cudaError_t e_ = (call);                                                   \
    if (e_ != cudaSuccess)                                                     \
      poseidon::gemm::failCall(#call, __FILE__, __LINE__,                      \
                               cudaGetErrorString(e_));                        \
  } while (0)

#define POSEIDON_GEMM_CB(call)                                                 \
  do {                                                                         \
    cublasStatus_t s_ = (call);                                                \
    if (s_ != CUBLAS_STATUS_SUCCESS)                                           \
      poseidon::gemm::failCall(#call, __FILE__, __LINE__,                      \
                               cublasGetStatusString(s_));                     \
  } while (0)

//===----------------------------------------------------------------------===//
// Per-device state
//===----------------------------------------------------------------------===//

// What a runtime keeps on one device. A call enqueues its work holding `lock`,
// so the handle's stream and the scratch stay the call's whatever other host
// threads do. `scratchFree` is recorded on the stream of every call that used
// the scratch, and the next call's stream waits for it, so calls on different
// streams take the scratch in turn.
struct Device {
  std::mutex lock;
  cublasHandle_t blas = nullptr;
  cudaEvent_t scratchFree = nullptr;
};

// The State (a Device) of the current device, one per runtime, created on
// first use and never destroyed.
template <class State> State &deviceState() {
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
template <class State> class Call {
public:
  Call(cudaStream_t stream, bool scratch)
      : dev(deviceState<State>()), stream(stream), hold(dev.lock),
        scratch(scratch) {
    if (scratch && dev.scratchFree)
      POSEIDON_GEMM_CK(cudaStreamWaitEvent(stream, dev.scratchFree, 0));
  }
  ~Call() {
    if (!scratch)
      return;
    if (!dev.scratchFree)
      POSEIDON_GEMM_CK(
          cudaEventCreateWithFlags(&dev.scratchFree, cudaEventDisableTiming));
    POSEIDON_GEMM_CK(cudaEventRecord(dev.scratchFree, stream));
  }
  Call(const Call &) = delete;
  Call &operator=(const Call &) = delete;

  // The device's handle, created on first use, on the call's stream.
  cublasHandle_t blas() {
    if (!dev.blas)
      POSEIDON_GEMM_CB(cublasCreate(&dev.blas));
    POSEIDON_GEMM_CB(cublasSetStream(dev.blas, stream));
    return dev.blas;
  }

  State &dev;
  const cudaStream_t stream;

private:
  std::lock_guard<std::mutex> hold;
  const bool scratch;
};

// Waits until no call enqueued before uses the scratch of `dev`, which may
// then be freed.
inline void drainScratch(Device &dev) {
  if (dev.scratchFree)
    POSEIDON_GEMM_CK(cudaEventSynchronize(dev.scratchFree));
}

// A device buffer of a runtime's scratch, grown to the largest size asked of
// it and kept.
struct ScratchBuffer {
  void *data = nullptr;
  size_t bytes = 0;

  void reserve(Device &dev, size_t want) {
    if (bytes >= want)
      return;
    if (data) {
      drainScratch(dev);
      POSEIDON_GEMM_CK(cudaFree(data));
    }
    POSEIDON_GEMM_CK(cudaMalloc(&data, want));
    bytes = want;
  }
  template <class T> T *as() const { return static_cast<T *>(data); }
};

//===----------------------------------------------------------------------===//
// Launch geometry
//===----------------------------------------------------------------------===//

// The block of an elementwise kernel over a matrix: kTile x kTile threads, x
// along the matrix's contiguous axis.
constexpr unsigned kTile = 16;
// The block of an elementwise kernel over a packed buffer.
constexpr unsigned kFlatBlock = 256;
// The grid's limit in y.
constexpr unsigned kMaxGridY = 65535;

inline unsigned flatBlocks(size_t n) {
  return (unsigned)((n + kFlatBlock - 1) / kFlatBlock);
}

// Calls launch(y0, n) for slabs of n <= kMaxGridY blocks in y, numbered from
// y0, that together cover `blocks`.
template <class Launch> void forGridY(unsigned blocks, Launch launch) {
  for (unsigned y0 = 0; y0 < blocks; y0 += kMaxGridY)
    launch(y0, std::min(blocks - y0, kMaxGridY));
}

// Calls launch(grid, y0) for the grids of kTile x kTile blocks that cover
// fast x slow elements, x along `fast`.
template <class Launch>
void forTiles(unsigned fast, unsigned slow, Launch launch) {
  const unsigned gx = (fast + kTile - 1) / kTile;
  forGridY((slow + kTile - 1) / kTile,
           [&](unsigned y0, unsigned n) { launch(dim3(gx, n), y0); });
}

// The element of a kTile x kTile tile kernel whose y blocks are numbered from
// y0: i along the contiguous axis, j along the other.
__device__ __forceinline__ int tileI() {
  return blockIdx.x * blockDim.x + threadIdx.x;
}
__device__ __forceinline__ int tileJ(unsigned y0) {
  return (blockIdx.y + y0) * blockDim.y + threadIdx.y;
}

// The extent of a rows x cols matrix along its contiguous axis, and along the
// other: element (r, c) is at c * ld + r column-major and r * ld + c
// row-major.
__host__ __device__ inline int contiguousExtent(int rows, int cols,
                                                int colMajor) {
  return colMajor ? rows : cols;
}
__host__ __device__ inline int stridedExtent(int rows, int cols, int colMajor) {
  return colMajor ? cols : rows;
}

//===----------------------------------------------------------------------===//
// The body of an entry
//===----------------------------------------------------------------------===//

// C = beta * C, the product of an empty contraction; Zero (beta == 0) writes
// zeros without reading C.
template <bool Zero>
__global__ void scaleKernel(double *C, int fast, int slow, int ldc, double beta,
                            unsigned y0) {
  const int i = tileI(), j = tileJ(y0);
  if (i >= fast || j >= slow)
    return;
  double &c = C[(size_t)j * ldc + i];
  c = Zero ? 0.0 : beta * c;
}

inline void scaleByBeta(double *C, int M, int Ncols, int ldc, int cColMajor,
                        double beta, cudaStream_t stream) {
  const int fast = contiguousExtent(M, Ncols, cColMajor);
  const int slow = stridedExtent(M, Ncols, cColMajor);
  forTiles(fast, slow, [&](dim3 grid, unsigned y0) {
    if (beta == 0.0)
      scaleKernel<true><<<grid, dim3(kTile, kTile), 0, stream>>>(C, fast, slow,
                                                                 ldc, beta, y0);
    else
      scaleKernel<false><<<grid, dim3(kTile, kTile), 0, stream>>>(
          C, fast, slow, ldc, beta, y0);
  });
}

// What every entry does around its product: a negative extent stops the
// program, M == 0 or Ncols == 0 leaves C alone, K == 0 gives C = beta * C,
// and otherwise multiply(call) enqueues the product under a Call<State>
// (with the scratch ordered when `scratch`).
template <class State, class Multiply>
void runEntry(const char *entry, double *C, int M, int Ncols, int K, int ldc,
              int cColMajor, double beta, cudaStream_t stream, bool scratch,
              Multiply multiply) {
  if (M < 0 || Ncols < 0 || K < 0)
    fail("%s with M = %d, Ncols = %d, K = %d: an extent is negative", entry, M,
         Ncols, K);
  if (M == 0 || Ncols == 0)
    return;
  if (K == 0) {
    scaleByBeta(C, M, Ncols, ldc, cColMajor, beta, stream);
  } else {
    Call<State> call(stream, scratch);
    multiply(call);
  }
  POSEIDON_GEMM_CK(cudaGetLastError());
}

//===----------------------------------------------------------------------===//
// The column-major call of the library
//===----------------------------------------------------------------------===//

// The arguments of the column-major library call that computes C (M x Ncols)
// = A B for A, B and C in the given layouts. A row-major matrix with leading
// dimension ld is the column-major transpose with the same ld, so a
// column-major C is op(A) op(B) with op = T for a row-major operand, and a
// row-major C is computed as C^T = B^T A^T: the operands swap and their ops
// flip.
struct ColumnMajorGemm {
  cublasOperation_t opA, opB;
  int m, n;
  const void *A;
  int lda;
  const void *B;
  int ldb;
};

inline ColumnMajorGemm columnMajorGemm(int M, int Ncols, const void *A, int lda,
                                       int aColMajor, const void *B, int ldb,
                                       int bColMajor, int cColMajor) {
  const cublasOperation_t opA = aColMajor ? CUBLAS_OP_N : CUBLAS_OP_T;
  const cublasOperation_t opB = bColMajor ? CUBLAS_OP_N : CUBLAS_OP_T;
  if (cColMajor)
    return {opA, opB, M, Ncols, A, lda, B, ldb};
  auto flip = [](cublasOperation_t o) {
    return o == CUBLAS_OP_N ? CUBLAS_OP_T : CUBLAS_OP_N;
  };
  return {flip(opB), flip(opA), Ncols, M, B, ldb, A, lda};
}

//===----------------------------------------------------------------------===//
// Operand repacking around a reduced-precision library call
//===----------------------------------------------------------------------===//

// dst = narrow(src) for a rows x cols FP64 matrix in `srcColMajor` layout with
// leading dimension ld, into a packed matrix (leading dimension its contiguous
// extent) in `dstColMajor` layout. Threads walk the source's contiguous axis.
template <class T, class Narrow>
__global__ void packKernel(T *__restrict__ dst, const double *__restrict__ src,
                           int rows, int cols, int ld, int srcColMajor,
                           int dstColMajor, Narrow narrow, unsigned y0) {
  const int i = tileI(), j = tileJ(y0);
  if (i >= contiguousExtent(rows, cols, srcColMajor) ||
      j >= stridedExtent(rows, cols, srcColMajor))
    return;
  const int r = srcColMajor ? i : j, c = srcColMajor ? j : i;
  dst[dstColMajor ? (size_t)c * rows + r : (size_t)r * cols + c] =
      narrow(src[(size_t)j * ld + i]);
}

template <class T, class Narrow>
__global__ void packFlatKernel(T *__restrict__ dst,
                               const double *__restrict__ src, size_t n,
                               Narrow narrow) {
  const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n)
    dst[i] = narrow(src[i]);
}

// A source already packed in the destination's layout is one flat pass.
template <class T, class Narrow>
void pack(T *dst, const double *src, int rows, int cols, int ld,
          int srcColMajor, int dstColMajor, Narrow narrow,
          cudaStream_t stream) {
  const int fast = contiguousExtent(rows, cols, srcColMajor);
  if (srcColMajor == dstColMajor && ld == fast) {
    const size_t n = (size_t)rows * cols;
    packFlatKernel<<<flatBlocks(n), kFlatBlock, 0, stream>>>(dst, src, n,
                                                             narrow);
    return;
  }
  forTiles(fast, stridedExtent(rows, cols, srcColMajor),
           [&](dim3 grid, unsigned y0) {
             packKernel<<<grid, dim3(kTile, kTile), 0, stream>>>(
                 dst, src, rows, cols, ld, srcColMajor, dstColMajor, narrow,
                 y0);
           });
}

// C = alpha * widen(P) + beta * C for C (M x Ncols, leading dimension ldc) and
// the packed FP32 product P in `pColMajor` layout; the FP64 alpha and beta are
// the caller's, which the library never saw. Threads walk C's contiguous axis.
// Accumulate is a template parameter rather than a test of beta: a ternary
// would load C unconditionally, an extra read of the output when beta = 0. The
// fused multiply-add is written out so that its rounding does not depend on
// the compiler's contraction.
template <bool Accumulate, class Widen>
__global__ void
unpackKernel(double *__restrict__ C, const float *__restrict__ P, int M,
             int Ncols, int ldc, int cColMajor, int pColMajor, double alpha,
             double beta, Widen widen, unsigned y0) {
  const int i = tileI(), j = tileJ(y0);
  if (i >= contiguousExtent(M, Ncols, cColMajor) ||
      j >= stridedExtent(M, Ncols, cColMajor))
    return;
  const int r = cColMajor ? i : j, c = cColMajor ? j : i;
  const double v =
      widen(P[pColMajor ? (size_t)c * M + r : (size_t)r * Ncols + c]);
  const size_t d = (size_t)j * ldc + i;
  C[d] = Accumulate ? fma(alpha, v, beta * C[d]) : alpha * v;
}

template <bool Accumulate, class Widen>
__global__ void unpackFlatKernel(double *__restrict__ C,
                                 const float *__restrict__ P, size_t n,
                                 double alpha, double beta, Widen widen) {
  const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const double v = widen(P[i]);
  C[i] = Accumulate ? fma(alpha, v, beta * C[i]) : alpha * v;
}

// A C packed in P's layout is one flat pass.
template <class Widen>
void unpack(double *C, const float *P, int M, int Ncols, int ldc, int cColMajor,
            int pColMajor, double alpha, double beta, Widen widen,
            cudaStream_t stream) {
  const int fast = contiguousExtent(M, Ncols, cColMajor);
  auto launch = [&](auto accumulate) {
    constexpr bool Accumulate = decltype(accumulate)::value;
    if (cColMajor == pColMajor && ldc == fast) {
      const size_t n = (size_t)M * Ncols;
      unpackFlatKernel<Accumulate><<<flatBlocks(n), kFlatBlock, 0, stream>>>(
          C, P, n, alpha, beta, widen);
      return;
    }
    forTiles(
        fast, stridedExtent(M, Ncols, cColMajor), [&](dim3 grid, unsigned y0) {
          unpackKernel<Accumulate><<<grid, dim3(kTile, kTile), 0, stream>>>(
              C, P, M, Ncols, ldc, cColMajor, pColMajor, alpha, beta, widen,
              y0);
        });
  };
  if (beta != 0.0)
    launch(std::true_type());
  else
    launch(std::false_type());
}

} // namespace gemm
} // namespace poseidon

#endif // POSEIDON_RUNTIME_GEMM_GEMM_RUNTIME_H
