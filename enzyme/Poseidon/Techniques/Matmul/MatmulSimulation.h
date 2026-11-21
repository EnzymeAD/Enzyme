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

// C = A * B as __poseidon_tcec_dgemm (runtime/tcec/tcec_rt.cu) returns it in
// mode 1, cuMpSGEMM's FP16TCEC, for A (M x K), B (K x N) and C (M x N)
// row-major. No source specifies how a tensor-core MMA adds its products; it
// is taken to round the exact sum of its 16 products and the accumulator
// toward zero to FP32.
void simulateTcecDgemm(const double *A, const double *B, unsigned M, unsigned N,
                       unsigned K, double *C);

// One operand of __poseidon_ozaki_dgemm(_ex) as runtime/ozaki/ozaki_rt.cu
// scales it: R vectors of K elements, element k of vector r at src[r * ld + k]
// when `contiguous` and at src[k * ld + r] otherwise. The layout decides the
// order in which the runtime sums a vector's squares.
struct OzakiIIOperand {
  const double *src;
  unsigned ld;
  unsigned R;
  bool contiguous;
};

// P (A.R x B.R, row-major): the product of each vector of A with each vector
// of B as __poseidon_ozaki_dgemm(_ex) returns it with nm moduli before its
// alpha/beta epilogue, which is GEMMul8's fast mode, bit for bit wherever the
// host's log2 and the device's __log2f put a scaling shift on the same side of
// an integer. A vector whose sum of squares is not finite is outside the fast
// mode's range, and its products are NaN.
void simulateOzakiIIDgemm(unsigned nm, const OzakiIIOperand &A,
                          const OzakiIIOperand &B, unsigned K, double *P);

} // namespace poseidon

#endif // POSEIDON_MATMUL_SIMULATION_H
