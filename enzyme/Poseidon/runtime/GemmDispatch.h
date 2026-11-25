//===- GemmDispatch.h - the host GEMM dispatch entry points ---------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The entry points the pass replaces a matrix product's launch with. Each
// computes C = alpha * (A @ B) + beta * C for A (M x K), B (K x Ncols) and
// C (M x Ncols) in device memory, each operand column-major (X[c * ld + r])
// when its ColMajor flag is set and row-major (X[r * ld + c]) otherwise, on
// `stream` (null for the default stream). An empty extent leaves C as it is.
// An error stops the program.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_GEMM_DISPATCH_H
#define POSEIDON_GEMM_DISPATCH_H

#ifdef __HIP__
#include "hip_as_cuda.h"
#else
#include <cuda_runtime.h>
#endif

extern "C" {

// The FP64 product of the GEMM library (cuBLAS or hipBLAS).
void __poseidon_native_dgemm(double *C, const double *A, const double *B, int M,
                             int Ncols, int K, int lda, int ldb, int ldc,
                             int aColMajor, int bColMajor, int cColMajor,
                             double alpha, double beta, cudaStream_t stream);

// Ozaki scheme II with num_moduli moduli, kOzakiIIMinModuli to
// kOzakiIIMaxModuli (OzakiII.h); K below kOzakiIIMaxK.
void __poseidon_ozaki_dgemm(double *C, const double *A, const double *B, int M,
                            int Ncols, int K, int lda, int ldb, int ldc,
                            int aColMajor, int bColMajor, int cColMajor,
                            double alpha, double beta, cudaStream_t stream,
                            int num_moduli);

// Tensor-core error correction on cuMpSGEMM in a PoseidonTcecMode (OzakiII.h).
// Only a build with cuMpSGEMM defines it.
void __poseidon_tcec_dgemm(double *C, const double *A, const double *B, int M,
                           int Ncols, int K, int lda, int ldb, int ldc,
                           int aColMajor, int bColMajor, int cColMajor,
                           double alpha, double beta, cudaStream_t stream,
                           int mode);

// Each operand rounded once to the PoseidonDirectMode's format (OzakiII.h).
void __poseidon_direct_dgemm(double *C, const double *A, const double *B, int M,
                             int Ncols, int K, int lda, int ldb, int ldc,
                             int aColMajor, int bColMajor, int cColMajor,
                             double alpha, double beta, cudaStream_t stream,
                             int mode);

} // extern "C"

#endif // POSEIDON_GEMM_DISPATCH_H
