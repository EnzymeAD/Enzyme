//===- Direct.cu - FP64 and direct reduced-precision GEMM runtimes --------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The library's FP64 product: one cublasDgemm (hipblasDgemm) in the
// column-major form of the call.
//
// The direct reduced-precision GEMM, the library realization of the in-kernel
// direct tensor-core raise.
//
// This runtime performs no operand scaling: each FP64 operand is rounded to the
// target input format as the in-kernel direct raise rounds it (FP16 and BF16
// once, including FP16's subnormal floor and 65504 ceiling; TF32 to FP32 and
// then to nearest with ties away, as f2tf32.rna), the product runs as one
// cublasGemmEx with an FP32 accumulator in the library's order, and the result
// is widened back to FP64 with the FP64 alpha and beta applied.
//
// Each operand is packed in its own layout, so the transpose is the call's op
// flag. Compiled as HIP (POSEIDON_ROCM_PATH) the same entry runs on hipBLAS,
// every mode but TF32, which has no hipBLAS compute type.
//
//===----------------------------------------------------------------------===//

#ifndef __HIP__
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#endif

#include "GemmRuntime.h"
#include "SchemeConstants.h"

using namespace poseidon::gemm;

namespace {

// The packed operands and the FP32 product. Their widest element is a float
// (TF32 and FP32), so one byte capacity per buffer serves every mode.
struct DirectDevice : Device {
  ScratchBuffer A, B, P;
};

// FP64 to the operand format, with the primitives the in-kernel raise emits:
// FP16 and BF16 in one step (through FP32 would round twice), TF32 through
// FP32.
struct RoundToHalf {
  __device__ __half operator()(double v) const { return __double2half(v); }
};
struct RoundToBFloat16 {
  __device__ __nv_bfloat16 operator()(double v) const {
    return __double2bfloat16(v);
  }
};
struct RoundToFloat {
  __device__ float operator()(double v) const { return (float)v; }
};
#ifndef __HIP__
// Rounded here rather than by the library, which may truncate: the TF32
// product then takes the operand as it is.
struct RoundToTf32 {
  __device__ float operator()(double v) const {
#if __CUDA_ARCH__ >= 800
    unsigned bits;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(bits) : "f"((float)v));
    return __uint_as_float(bits);
#else
    return (float)v; // no TF32 before sm_80: the library computes in FP32
#endif
  }
};
#endif

struct WidenFloat {
  __device__ double operator()(float p) const { return (double)p; }
};

// How the library takes the operands of a mode.
struct LibraryFormat {
  cudaDataType_t operand;
  cublasComputeType_t compute;
  size_t operandBytes;
};

LibraryFormat formatOf(int mode) {
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
  fail("__poseidon_direct_dgemm with mode %d: this build takes a "
       "PoseidonDirectMode, %s",
       mode, modes);
}

// Packs a rows x cols operand, in its own layout, in the format of `mode`.
void packOperand(int mode, void *dst, const double *src, int rows, int cols,
                 int ld, int colMajor, cudaStream_t stream) {
  auto as = [&](auto *typed, auto round) {
    pack(typed, src, rows, cols, ld, colMajor, colMajor, round, stream);
  };
  switch (mode) {
  case kDirectFp16:
    return as(static_cast<__half *>(dst), RoundToHalf());
  case kDirectBf16:
    return as(static_cast<__nv_bfloat16 *>(dst), RoundToBFloat16());
#ifndef __HIP__
  case kDirectTf32:
    return as(static_cast<float *>(dst), RoundToTf32());
#endif
  default:
    return as(static_cast<float *>(dst), RoundToFloat());
  }
}

// Pack, one cublasGemmEx with alpha = 1 and beta = 0 into P, unpack. P is in
// C's layout, which the column-major form of the call gives it.
void multiply(Call<DirectDevice> &call, int mode, const LibraryFormat &format,
              double *C, const double *A, const double *B, int M, int Ncols,
              int K, int lda, int ldb, int ldc, int aColMajor, int bColMajor,
              int cColMajor, double alpha, double beta) {
  DirectDevice &dev = call.dev;
  dev.A.reserve(dev, (size_t)M * K * format.operandBytes);
  dev.B.reserve(dev, (size_t)K * Ncols * format.operandBytes);
  dev.P.reserve(dev, (size_t)M * Ncols * sizeof(float));
  packOperand(mode, dev.A.data, A, M, K, lda, aColMajor, call.stream);
  packOperand(mode, dev.B.data, B, K, Ncols, ldb, bColMajor, call.stream);

  const ColumnMajorGemm g = columnMajorGemm(
      M, Ncols, dev.A.data, contiguousExtent(M, K, aColMajor), aColMajor,
      dev.B.data, contiguousExtent(K, Ncols, bColMajor), bColMajor, cColMajor);
  const float one = 1.0f, zero = 0.0f;
  POSEIDON_GEMM_CB(cublasGemmEx(call.blas(), g.opA, g.opB, g.m, g.n, K, &one,
                                g.A, format.operand, g.lda, g.B, format.operand,
                                g.ldb, &zero, dev.P.as<float>(), CUDA_R_32F,
                                g.m, format.compute, CUBLAS_GEMM_DEFAULT));

  unpack(C, dev.P.as<float>(), M, Ncols, ldc, cColMajor,
         /*pColMajor=*/cColMajor, alpha, beta, WidenFloat(), call.stream);
}

} // namespace

extern "C" void __poseidon_direct_dgemm(
    double *C, const double *A, const double *B, int M, int Ncols, int K,
    int lda, int ldb, int ldc, int aColMajor, int bColMajor, int cColMajor,
    double alpha, double beta, cudaStream_t stream, int mode) {
  const LibraryFormat format = formatOf(mode);
  runEntry<DirectDevice>(
      "__poseidon_direct_dgemm", C, M, Ncols, K, ldc, cColMajor, beta, stream,
      /*scratch=*/true, [&](Call<DirectDevice> &call) {
        multiply(call, mode, format, C, A, B, M, Ncols, K, lda, ldb, ldc,
                 aColMajor, bColMajor, cColMajor, alpha, beta);
      });
}

extern "C" void __poseidon_native_dgemm(double *C, const double *A,
                                        const double *B, int M, int Ncols,
                                        int K, int lda, int ldb, int ldc,
                                        int aColMajor, int bColMajor,
                                        int cColMajor, double alpha,
                                        double beta, cudaStream_t stream) {
  runEntry<Device>(
      "__poseidon_native_dgemm", C, M, Ncols, K, ldc, cColMajor, beta, stream,
      /*scratch=*/false, [&](Call<Device> &call) {
        const ColumnMajorGemm g = columnMajorGemm(M, Ncols, A, lda, aColMajor,
                                                  B, ldb, bColMajor, cColMajor);
        POSEIDON_GEMM_CB(cublasDgemm(call.blas(), g.opA, g.opB, g.m, g.n, K,
                                     &alpha, static_cast<const double *>(g.A),
                                     g.lda, static_cast<const double *>(g.B),
                                     g.ldb, &beta, C, ldc));
      });
}
