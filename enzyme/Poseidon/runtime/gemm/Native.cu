//===- Native.cu - FP64 GEMM runtime --------------------------------------===//
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
//===----------------------------------------------------------------------===//

#include "GemmRuntime.h"

using namespace poseidon::gemm;

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
