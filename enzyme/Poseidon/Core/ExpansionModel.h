//===- ExpansionModel.h - Host emulation of FP32 expansions ---------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The accuracy model's emulation of the arithmetic Expansion.cpp emits, one
// routine per emitter, operation for operation, so that the model computes
// bit for bit what the emitted code computes.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_EXPANSIONMODEL_H
#define POSEIDON_EXPANSIONMODEL_H

namespace poseidon {

/// An n-component FP32 expansion, n = 2, 3, 4, most significant component
/// first; components past n are zero.
struct ExpansionLimbs {
  /// The components, most significant first.
  float x[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  /// The number of components in use.
  unsigned n = 0;
  /// The expansion of the `n` components `limbs` points to.
  static ExpansionLimbs of(const float *limbs, unsigned n);
};

/// The `n`-component expansion of `v`, as the emitted boundary conversion
/// computes it.
ExpansionLimbs expansionFromF64(double v, unsigned n);
/// The `n`-component expansion of `v`, exact.
ExpansionLimbs expansionFromF32(float v, unsigned n);
/// `a` rounded to double, as the emitted boundary conversion computes it.
double expansionToF64(const ExpansionLimbs &a);
/// `a` rounded to float, as the emitted boundary conversion computes it.
float expansionToF32(const ExpansionLimbs &a);

/// -`a`, exact.
ExpansionLimbs expansionNeg(const ExpansionLimbs &a);
/// `a` + `b` in a.n components, as the emitted routine computes it; `b` has as
/// many. Likewise for the subtraction, product, quotient and square root below.
ExpansionLimbs expansionAdd(const ExpansionLimbs &a, const ExpansionLimbs &b);
ExpansionLimbs expansionSub(const ExpansionLimbs &a, const ExpansionLimbs &b);
ExpansionLimbs expansionMul(const ExpansionLimbs &a, const ExpansionLimbs &b);
/// a * a where the materializer multiplies a value by itself: the square
/// routine for n >= 3, the general product for n = 2.
ExpansionLimbs expansionSqr(const ExpansionLimbs &a);
ExpansionLimbs expansionDiv(const ExpansionLimbs &a, const ExpansionLimbs &b);
ExpansionLimbs expansionSqrt(const ExpansionLimbs &a);
/// fma/fmuladd(a, b, c), and fma/fmuladd(a, a, c).
ExpansionLimbs expansionFma(const ExpansionLimbs &a, const ExpansionLimbs &b,
                            const ExpansionLimbs &c);
/// fma/fmuladd(a, a, c).
ExpansionLimbs expansionFmaSqr(const ExpansionLimbs &a,
                               const ExpansionLimbs &c);

} // namespace poseidon
#endif // POSEIDON_EXPANSIONMODEL_H
