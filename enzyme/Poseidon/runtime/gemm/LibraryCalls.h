//===- LibraryCalls.h - the host library call entry points ----------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The entry points of LibraryCalls.inc. Each computes
// C = alpha * (A @ B) + beta * C for A (M x K), B (K x Ncols) and C (M x Ncols)
// in memory of the current device, each operand column-major (X[c * ld + r])
// when its ColMajor flag is set and row-major (X[r * ld + c]) otherwise, on
// `stream` (a cudaStream_t or hipStream_t), from any host thread. With K == 0
// it computes C = beta * C, with M == 0 or Ncols == 0 nothing. A negative
// extent, or any error, stops the program.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_RUNTIME_GEMM_LIBRARY_CALLS_H
#define POSEIDON_RUNTIME_GEMM_LIBRARY_CALLS_H

#define POSEIDON_GEMM_DECL(CType, IRType, Name) CType Name,
#define POSEIDON_GEMM_DECL_LAST(CType, IRType, Name) CType Name
#define POSEIDON_GEMM_ENTRY(Scheme, Symbol)                                    \
  void Symbol(                                                                 \
      POSEIDON_GEMM_PARAMS(POSEIDON_GEMM_DECL, POSEIDON_GEMM_DECL_LAST));
#define POSEIDON_GEMM_ENTRY_MODE(Scheme, Symbol, Mode)                         \
  void Symbol(                                                                 \
      POSEIDON_GEMM_PARAMS(POSEIDON_GEMM_DECL, POSEIDON_GEMM_DECL_LAST),       \
      int Mode);
extern "C" {
#include "LibraryCalls.inc"
}
#undef POSEIDON_GEMM_DECL
#undef POSEIDON_GEMM_DECL_LAST

#endif // POSEIDON_RUNTIME_GEMM_LIBRARY_CALLS_H
