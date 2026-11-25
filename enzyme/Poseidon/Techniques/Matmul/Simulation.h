//===- Simulation.h - What the GEMM runtimes return, on the host ----------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The products the host runtimes return for a library-call candidate, computed
// on the host for the accuracy model (MatmulAccuracy.cpp): every step the
// runtime's source specifies, in its order and with its rounding. Free of
// LLVM, so that a test can hold it against what a runtime returned. It models
// the arithmetic of cuMpSGEMM and of GEMMul8 v3.1.1's fast mode as the
// runtimes run them; no code is taken from either.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_TECHNIQUES_MATMUL_SIMULATION_H
#define POSEIDON_TECHNIQUES_MATMUL_SIMULATION_H

namespace poseidon {

/// C = A * B as __poseidon_tcec_dgemm (runtime/gemm/Tcec.cu) returns it in
/// mode kTcecFp16Tcec, cuMpSGEMM's FP16TCEC, for A (M x K), B (K x N) and C (M
/// x N) row-major. No source specifies how a tensor-core MMA adds its products;
/// it is taken to round the exact sum of its 16 products and the accumulator
/// toward zero to FP32.
void simulateTcecDgemm(const double *A, const double *B, unsigned M, unsigned N,
                       unsigned K, double *C);

/// One operand of __poseidon_ozakiii_dgemm as runtime/gemm/OzakiII.cu
/// scales it: R vectors of K elements, element k of vector r at src[r * ld + k]
/// when `contiguous` and at src[k * ld + r] otherwise. The layout decides the
/// order in which the runtime sums a vector's squares.
struct OzakiIIOperand {
  /// The operand's first element.
  const double *src;
  /// The stride in elements between vectors when `contiguous`, between elements
  /// of a vector otherwise.
  unsigned ld;
  /// The number of vectors.
  unsigned R;
  /// Whether the K elements of a vector are adjacent.
  bool contiguous;
};

/// P (A.R x B.R, row-major): the product of each vector of A with each vector
/// of B as __poseidon_ozakiii_dgemm returns it with nm moduli before its
/// alpha/beta epilogue, which is GEMMul8's fast mode, bit for bit wherever the
/// host's log2 and the device's __log2f put a scaling shift on the same side of
/// an integer. A vector whose sum of squares is not finite is outside the fast
/// mode's range, and its products are NaN.
void simulateOzakiIIDgemm(unsigned nm, const OzakiIIOperand &A,
                          const OzakiIIOperand &B, unsigned K, double *P);

} // namespace poseidon

#endif // POSEIDON_TECHNIQUES_MATMUL_SIMULATION_H
