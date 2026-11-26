/*===- LibraryCalls.inc - the host library call entry points --*- C++ -*-===*\
|*
|* Part of the Enzyme Project, under the Apache License v2.0 with LLVM
|* Exceptions. See https://llvm.org/LICENSE.txt for license information.
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
|*
\*===----------------------------------------------------------------------===*/
/*
 * The one list of the entry points the pass replaces a matrix product's
 * launch with, included by the pass to build their prototypes and calls and
 * by the runtimes and their callers to declare them (LibraryCalls.h). An
 * includer defines the macros it uses; the others expand to nothing.
 *
 * POSEIDON_GEMM_PARAMS(P, LAST) lists the parameters every entry takes, in
 * order, as P(C type, IR type, name) and the last one as LAST(...); the IR
 * type is Ptr, I32 or Double.
 *
 * POSEIDON_GEMM_ENTRY(Scheme, Symbol) is an entry with only those
 * parameters; POSEIDON_GEMM_ENTRY_MODE(Scheme, Symbol, Mode) one with a last
 * int parameter Mode. Scheme names its Scheme enumerator in the pass.
 *
 * The values of Mode (SchemeConstants.h):
 *   __poseidon_ozakiii_dgemm   num_moduli kOzakiIIMinModuli..kOzakiIIMaxModuli;
 *                            the pass emits kOzakiIIModuliCounts only, the
 *                            smaller counts exist for direct callers.
 *   __poseidon_tcec_dgemm    a PoseidonTcecMode; the pass emits
 *                            kTcecFp16Tcec only, the others exist for direct
 *                            callers.
 *   __poseidon_direct_dgemm  a PoseidonDirectMode, all of which the pass
 *                            emits; HIP has no kDirectTf32.
 * An entry stops the program on a value it does not implement.
\*===----------------------------------------------------------------------===*/

#ifndef POSEIDON_GEMM_PARAMS
#define POSEIDON_GEMM_PARAMS(P, LAST)                                          \
  P(double *, Ptr, C)                                                          \
  P(const double *, Ptr, A)                                                    \
  P(const double *, Ptr, B)                                                    \
  P(int, I32, M)                                                               \
  P(int, I32, Ncols)                                                           \
  P(int, I32, K)                                                               \
  P(int, I32, lda)                                                             \
  P(int, I32, ldb)                                                             \
  P(int, I32, ldc)                                                             \
  P(int, I32, aColMajor)                                                       \
  P(int, I32, bColMajor)                                                       \
  P(int, I32, cColMajor)                                                       \
  P(double, Double, alpha)                                                     \
  P(double, Double, beta)                                                      \
  LAST(gpuStream_t, Ptr, stream)
#endif

#ifndef POSEIDON_GEMM_ENTRY
#define POSEIDON_GEMM_ENTRY(Scheme, Symbol)
#endif
#ifndef POSEIDON_GEMM_ENTRY_MODE
#define POSEIDON_GEMM_ENTRY_MODE(Scheme, Symbol, Mode)
#endif

/* The library's FP64 product (cuBLAS or hipBLAS). */
POSEIDON_GEMM_ENTRY(Native, __poseidon_native_dgemm)
/* Ozaki scheme II with Mode moduli; K below kOzakiIIMaxK. */
POSEIDON_GEMM_ENTRY_MODE(OzakiII, __poseidon_ozakiii_dgemm, num_moduli)
/* Tensor-core error correction on cuMpSGEMM; only a build with cuMpSGEMM
   defines it. With M * Ncols below 2^22 cuMpSGEMM adds the K-slices of the
   product by FP32 atomics, so the result varies in its last bits from call
   to call. */
POSEIDON_GEMM_ENTRY_MODE(Tcec, __poseidon_tcec_dgemm, mode)
/* Each operand rounded to the format of Mode as the in-kernel raise rounds
   it. */
POSEIDON_GEMM_ENTRY_MODE(Direct, __poseidon_direct_dgemm, mode)

#undef POSEIDON_GEMM_ENTRY
#undef POSEIDON_GEMM_ENTRY_MODE
