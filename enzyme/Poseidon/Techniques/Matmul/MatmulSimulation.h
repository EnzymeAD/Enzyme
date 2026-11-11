//=- MatmulSimulation.h - What the GEMM runtimes return, on the host ------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The products the host runtimes return for a library-call candidate, computed
// on the host for the accuracy model (MatmulAccuracy.cpp): every step the
// runtime's source specifies, in its order and with its rounding. Free of
// LLVM, so that a test can hold it against what a runtime returned.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_MATMUL_SIMULATION_H
#define POSEIDON_MATMUL_SIMULATION_H

namespace poseidon {

// The backend of __poseidon_tcec_dgemm (runtime/tcec/tcec_rt.cu): the
// runtime's own split into three cuBLAS GEMMs, or cuMpSGEMM's FP16TCEC when
// the runtime is built with POSEIDON_TCEC_USE_CUMPSGEMM.
enum class TcecBackend { BuiltIn, CuMpSGEMM };

// The backend of the runtime this build of the compiler links.
#ifdef POSEIDON_TCEC_USE_CUMPSGEMM
inline constexpr TcecBackend kLinkedTcecBackend = TcecBackend::CuMpSGEMM;
#else
inline constexpr TcecBackend kLinkedTcecBackend = TcecBackend::BuiltIn;
#endif

const char *tcecBackendName(TcecBackend backend);

// C = A * B as __poseidon_tcec_dgemm returns it in mode 1 (FP16TCEC) on
// `backend`, for A (M x K), B (K x N) and C (M x N) row-major. No source
// specifies how a tensor-core MMA adds its products; it is taken to round the
// exact sum of its 16 products and the accumulator toward zero to FP32.
void simulateTcecDgemm(TcecBackend backend, const double *A, const double *B,
                       unsigned M, unsigned N, unsigned K, double *C);

} // namespace poseidon

#endif // POSEIDON_MATMUL_SIMULATION_H
