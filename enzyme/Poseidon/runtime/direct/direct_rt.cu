//===- direct_rt.cu - direct reduced-precision GEMM runtime ---------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The direct reduced-precision GEMM, the library realization of the in-kernel
// direct tensor-core raise.
//
// This runtime performs no operand scaling: each FP64 operand is rounded to the
// target input format as the in-kernel direct raise rounds it (FP16 and BF16
// once, including FP16's subnormal floor and 65504 ceiling; TF32 to FP32 and
// then to nearest with ties away, as f2tf32.rna), the product runs as one
// cublasGemmEx with an FP32 accumulator in the library's order, and the result
// is widened back to FP64.
//
// The GEMM runs with alpha=1, beta=0 into scratch and the FP64 alpha/beta are
// applied in the widen kernel, so C = alpha*(A@B) + beta*C keeps FP64 scalar
// semantics.
//
// Compiled as HIP (POSEIDON_ROCM_PATH) the same entries run on hipBLAS, every
// mode but TF32, which has no hipBLAS compute type.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#ifndef __HIP__
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#endif

#include "../GemmRuntime.h"
#include "../ozaki/OzakiII.h"

namespace {

// Widest element the operand buffers ever hold is a float (the TF32 and FP32
// modes), so one byte capacity per buffer serves every mode.
struct PdrDevice : PoseidonDevice {
  size_t capA = 0, capB = 0, capC = 0; // bytes
  void *A = nullptr, *B = nullptr, *C = nullptr;
};

#ifndef __HIP__
// An FP32 element holding a TF32 value.
struct PdrTf32 {
  float v;
};
#endif

// FP64 -> the input format, with the primitives the in-kernel raise emits:
// FP16 and BF16 in one step (through FP32 would round twice), TF32 through
// FP32.
__device__ __forceinline__ void pdr_round(__half *d, double v) {
  *d = __double2half(v);
}
__device__ __forceinline__ void pdr_round(__nv_bfloat16 *d, double v) {
  *d = __double2bfloat16(v);
}
__device__ __forceinline__ void pdr_round(float *d, double v) { *d = (float)v; }
#ifndef __HIP__
// Rounded here rather than by the library, which may truncate: the TF32
// product then takes the operand as it is.
__device__ __forceinline__ void pdr_round(PdrTf32 *d, double v) {
#if __CUDA_ARCH__ >= 800
  unsigned bits;
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(bits) : "f"((float)v));
  d->v = __uint_as_float(bits);
#else
  d->v = (float)v; // no TF32 before sm_80: the library computes in FP32
#endif
}
#endif

// Narrow a layout-aware FP64 source into a contiguous buffer that keeps the
// source's own layout (packed to the tight leading dimension), so both the
// read and the write are coalesced and the transpose becomes the GEMM's op
// flag.
//   colMajor source: element (r,c) at c*ld + r  ->  dst[c*rows + r]
//   row-major source: element (r,c) at r*ld + c ->  dst[r*cols + c]
// Blocks in y are numbered from y0 (poseidonForGridY).
template <typename T>
__global__ void pdr_narrow(T *__restrict__ dst, const double *__restrict__ src,
                           int rows, int cols, int ld, int colMajor,
                           unsigned y0) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // fastest-varying axis
  int j = (blockIdx.y + y0) * blockDim.y + threadIdx.y;
  if (colMajor) {
    if (i >= rows || j >= cols)
      return;
    pdr_round(&dst[(size_t)j * rows + i], src[(size_t)j * ld + i]);
  } else {
    if (i >= cols || j >= rows)
      return;
    pdr_round(&dst[(size_t)j * cols + i], src[(size_t)j * ld + i]);
  }
}

// Packed fast path: when the leading dimension equals the packed extent the
// narrow is a flat elementwise pass; the general kernel above stays for
// strided sub-matrix views.
template <typename T>
__global__ void pdr_narrow_flat(T *__restrict__ dst,
                                const double *__restrict__ src, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n)
    pdr_round(&dst[i], src[i]);
}

// Widen the FP32 result back into the layout-aware FP64 destination, applying
// the FP64 alpha/beta the library never saw. The scratch is column-major
// M x Ncols (ld M) when C is column-major and column-major Ncols x M (ld Ncols)
// when C is row-major (the operands are swapped), so in both cases the scratch
// index is (slow index)*(fast extent) + (fast index) in C's axis order.
//
// ACC is a template parameter rather than a runtime `beta == 0` test: written
// as a ternary the accumulate arm's `C[d]` read is emitted as an unconditional
// load, so the beta = 0 case would pay an extra read of the FP64 output. The
// fused multiply-add is written out so that its rounding does not depend on
// the compiler's contraction.
template <bool ACC>
__global__ void pdr_widen(double *__restrict__ C, const float *__restrict__ src,
                          int rows, int cols, int ldc, int cColMajor,
                          double alpha, double beta, unsigned y0) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = (blockIdx.y + y0) * blockDim.y + threadIdx.y;
  size_t o, d;
  if (cColMajor) {
    if (i >= rows || j >= cols)
      return;
    o = (size_t)j * rows + i;
    d = (size_t)j * ldc + i;
  } else {
    if (i >= cols || j >= rows)
      return;
    o = (size_t)j * cols + i;
    d = (size_t)j * ldc + i;
  }
  const double c = (double)src[o];
  if (ACC)
    C[d] = fma(alpha, c, beta * C[d]);
  else
    C[d] = alpha * c;
}

// Packed fast path for the widen: when C's leading dimension equals its packed
// extent the scratch index and the destination index coincide.
template <bool ACC>
__global__ void pdr_widen_flat(double *__restrict__ C,
                               const float *__restrict__ src, size_t n,
                               double alpha, double beta) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const double c = (double)src[i];
  if (ACC)
    C[i] = fma(alpha, c, beta * C[i]);
  else
    C[i] = alpha * c;
}

struct PdrFormat {
  cudaDataType_t dt;
  cublasComputeType_t ct;
  size_t elemBytes;
};

PdrFormat pdr_format(int mode) {
  switch (mode) {
  case kDirectFp16:
    return {CUDA_R_16F, CUBLAS_COMPUTE_32F, sizeof(__half)};
  case kDirectBf16:
    return {CUDA_R_16BF, CUBLAS_COMPUTE_32F, sizeof(__nv_bfloat16)};
#ifndef __HIP__
  case kDirectTf32:
    return {CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32, sizeof(float)};
#endif
  case kDirectFp32:
    return {CUDA_R_32F, CUBLAS_COMPUTE_32F, sizeof(float)};
  }
#ifdef __HIP__
  const char *modes = "1, 2 or 4 (hipBLAS has no TF32)";
#else
  const char *modes = "1 to 4";
#endif
  fflush(stdout);
  fprintf(stderr,
          "Poseidon GEMM runtime: __poseidon_direct_dgemm with mode %d: this "
          "build takes a PoseidonDirectMode, %s\n",
          mode, modes);
  abort();
}

// Narrow a rows x cols operand into dst, in the element type of `mode`: flat
// when its leading dimension is its packed extent, strided otherwise.
void pdr_launchNarrow(int mode, void *dst, const double *src, int rows,
                      int cols, int ld, int colMajor, cudaStream_t stream) {
  auto launch = [&](auto *typed) {
    if (ld == (colMajor ? rows : cols)) {
      const size_t n = (size_t)rows * cols;
      pdr_narrow_flat<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(typed,
                                                                       src, n);
      return;
    }
    const unsigned fast = colMajor ? rows : cols, slow = colMajor ? cols : rows;
    poseidonForGridY((slow + 15) / 16, [&](unsigned y0, unsigned n) {
      pdr_narrow<<<dim3((fast + 15) / 16, n), dim3(16, 16), 0, stream>>>(
          typed, src, rows, cols, ld, colMajor, y0);
    });
  };
  if (mode == kDirectFp16)
    launch((__half *)dst);
  else if (mode == kDirectBf16)
    launch((__nv_bfloat16 *)dst);
#ifndef __HIP__
  else if (mode == kDirectTf32)
    launch((PdrTf32 *)dst);
#endif
  else
    launch((float *)dst);
}

// Narrow -> one cublasGemmEx -> widen.
void pdr_run(PdrDevice &g, cublasHandle_t h, double *C, const double *A,
             const double *B, int M, int Ncols, int K, int lda, int ldb,
             int ldc, int aColMajor, int bColMajor, int cColMajor, double alpha,
             double beta, cudaStream_t stream, int mode, const PdrFormat &f) {
  poseidonReserve(g, &g.A, &g.capA, (size_t)M * K * f.elemBytes);
  poseidonReserve(g, &g.B, &g.capB, (size_t)K * Ncols * f.elemBytes);
  poseidonReserve(g, &g.C, &g.capC, (size_t)M * Ncols * sizeof(float));
  float *wide = (float *)g.C;

  dim3 b(16, 16);
  pdr_launchNarrow(mode, g.A, A, M, K, lda, aColMajor, stream);
  pdr_launchNarrow(mode, g.B, B, K, Ncols, ldb, bColMajor, stream);

  // The packed A buffer is a column-major M x K matrix (ld = M) when the source
  // was column-major, and a column-major K x M matrix (ld = K) otherwise, so
  // the op flag that recovers the logical M x K operand is N or T respectively;
  // likewise for B. A row-major C is produced by computing its transpose,
  // (A@B)^T = B^T @ A^T, which swaps the operands and flips both op flags.
  const cublasOperation_t opA = aColMajor ? CUBLAS_OP_N : CUBLAS_OP_T;
  const cublasOperation_t opB = bColMajor ? CUBLAS_OP_N : CUBLAS_OP_T;
  const int packLdA = aColMajor ? M : K;
  const int packLdB = bColMajor ? K : Ncols;
  auto flip = [](cublasOperation_t o) {
    return o == CUBLAS_OP_N ? CUBLAS_OP_T : CUBLAS_OP_N;
  };

  const float one = 1.0f, zero = 0.0f;
  if (cColMajor) {
    POSEIDON_GEMM_CB(cublasGemmEx(h, opA, opB, M, Ncols, K, &one, g.A, f.dt,
                                  packLdA, g.B, f.dt, packLdB, &zero, wide,
                                  CUDA_R_32F, M, f.ct, CUBLAS_GEMM_DEFAULT));
  } else {
    POSEIDON_GEMM_CB(cublasGemmEx(h, flip(opB), flip(opA), Ncols, M, K, &one,
                                  g.B, f.dt, packLdB, g.A, f.dt, packLdA, &zero,
                                  wide, CUDA_R_32F, Ncols, f.ct,
                                  CUBLAS_GEMM_DEFAULT));
  }

  const bool acc = (beta != 0.0);
  if (ldc == (cColMajor ? M : Ncols)) {
    const size_t n = (size_t)M * Ncols;
    const unsigned wg = (unsigned)((n + 255) / 256);
    if (acc)
      pdr_widen_flat<true><<<wg, 256, 0, stream>>>(C, wide, n, alpha, beta);
    else
      pdr_widen_flat<false><<<wg, 256, 0, stream>>>(C, wide, n, alpha, beta);
  } else {
    const unsigned fast = cColMajor ? M : Ncols, slow = cColMajor ? Ncols : M;
    poseidonForGridY((slow + 15) / 16, [&](unsigned y0, unsigned n) {
      const dim3 wgrid((fast + 15) / 16, n);
      if (acc)
        pdr_widen<true><<<wgrid, b, 0, stream>>>(C, wide, M, Ncols, ldc,
                                                 cColMajor, alpha, beta, y0);
      else
        pdr_widen<false><<<wgrid, b, 0, stream>>>(C, wide, M, Ncols, ldc,
                                                  cColMajor, alpha, beta, y0);
    });
  }
}

} // namespace

// The operands are repacked during the narrow, so any shape and layout
// combination is one library call.
extern "C" void __poseidon_direct_dgemm(
    double *C, const double *A, const double *B, int M, int Ncols, int K,
    int lda, int ldb, int ldc, int aColMajor, int bColMajor, int cColMajor,
    double alpha, double beta, cudaStream_t stream, int mode) {
  const PdrFormat f = pdr_format(mode);
  if (!poseidonGemmExtents("__poseidon_direct_dgemm", M, Ncols, K)) {
    if (M > 0 && Ncols > 0)
      poseidonGemmEmptyK(C, M, Ncols, ldc, cColMajor, beta, stream);
    return;
  }
  {
    PoseidonGemmCall<PdrDevice> call(stream, /*scratch=*/true);
    pdr_run(call.dev, call.blas(), C, A, B, M, Ncols, K, lda, ldb, ldc,
            aColMajor, bColMajor, cColMajor, alpha, beta, stream, mode, f);
  }
  POSEIDON_GEMM_CK(cudaGetLastError());
}
