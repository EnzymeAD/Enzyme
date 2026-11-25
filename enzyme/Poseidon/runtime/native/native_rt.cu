//===- native_rt.cu - FP64 GEMM runtime -----------------------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../GemmRuntime.h"

// A row-major rows x cols matrix with row stride ld is the column-major
// cols x rows transpose with the same ld, so every layout combination is one
// cublasDgemm: for a row-major C, the column-major C^T = op(B) * op(A) with
// the ops flipped.
extern "C" void __poseidon_native_dgemm(double *C, const double *A,
                                        const double *B, int M, int Ncols,
                                        int K, int lda, int ldb, int ldc,
                                        int aColMajor, int bColMajor,
                                        int cColMajor, double alpha,
                                        double beta, cudaStream_t stream) {
  if (!poseidonGemmExtents("__poseidon_native_dgemm", M, Ncols, K)) {
    if (M > 0 && Ncols > 0)
      poseidonGemmEmptyK(C, M, Ncols, ldc, cColMajor, beta, stream);
    return;
  }
  PoseidonGemmCall<PoseidonDevice> call(stream, /*scratch=*/false);
  cublasHandle_t h = call.blas();
  if (cColMajor)
    POSEIDON_GEMM_CB(cublasDgemm(h, aColMajor ? CUBLAS_OP_N : CUBLAS_OP_T,
                                 bColMajor ? CUBLAS_OP_N : CUBLAS_OP_T, M,
                                 Ncols, K, &alpha, A, lda, B, ldb, &beta, C,
                                 ldc));
  else
    POSEIDON_GEMM_CB(cublasDgemm(h, bColMajor ? CUBLAS_OP_T : CUBLAS_OP_N,
                                 aColMajor ? CUBLAS_OP_T : CUBLAS_OP_N, Ncols,
                                 M, K, &alpha, B, ldb, A, lda, &beta, C, ldc));
}
