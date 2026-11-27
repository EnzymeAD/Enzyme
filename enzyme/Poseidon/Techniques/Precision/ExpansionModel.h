//===- ExpansionModel.h - Host emulation of FP32 expansions ---------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
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

#ifndef POSEIDON_TECHNIQUES_PRECISION_EXPANSION_MODEL_H
#define POSEIDON_TECHNIQUES_PRECISION_EXPANSION_MODEL_H

#include "Core/NumberFormat.h"
#include "Model/Evaluators.h"

#include <memory>

namespace poseidon {

/// An n-component FP32 expansion, n = 2, 3, 4, most significant component
/// first; components past n are zero.
struct FloatExpansion {
  /// The components, most significant first.
  float x[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  /// The number of components in use.
  unsigned n = 0;
  /// The expansion of the `n` components `components` points to.
  static FloatExpansion of(const float *components, unsigned n);
};

/// The `n`-component expansion of `v`, as the emitted boundary conversion
/// computes it.
FloatExpansion expansionFromF64(double v, unsigned n);
/// The `n`-component expansion of `v`, exact.
FloatExpansion expansionFromF32(float v, unsigned n);
/// `a` rounded to double, as the emitted boundary conversion computes it.
double expansionToF64(const FloatExpansion &a);
/// `a` rounded to float, as the emitted boundary conversion computes it.
float expansionToF32(const FloatExpansion &a);

/// -`a`, exact.
FloatExpansion expansionNeg(const FloatExpansion &a);
/// `a` + `b` in a.n components, as the emitted routine computes it; `b` has as
/// many. Likewise for the subtraction, product, quotient and square root below.
FloatExpansion expansionAdd(const FloatExpansion &a, const FloatExpansion &b);
FloatExpansion expansionSub(const FloatExpansion &a, const FloatExpansion &b);
FloatExpansion expansionMul(const FloatExpansion &a, const FloatExpansion &b);
/// a * a where the materializer multiplies a value by itself: the square
/// routine for n >= 3, the general product for n = 2.
FloatExpansion expansionSqr(const FloatExpansion &a);
FloatExpansion expansionDiv(const FloatExpansion &a, const FloatExpansion &b);
FloatExpansion expansionSqrt(const FloatExpansion &a);
/// fma/fmuladd(a, b, c).
FloatExpansion expansionFma(const FloatExpansion &a, const FloatExpansion &b,
                            const FloatExpansion &c);
/// fma/fmuladd(a, a, c).
FloatExpansion expansionFmaSqr(const FloatExpansion &a,
                               const FloatExpansion &c);

/// The evaluator's arithmetic of the expansion format `format`, which must be
/// one: each operation computed as the emitted code computes it, an operand of
/// the same change read from the components kept for it, any other split from
/// its value as the emitted boundary conversion splits it.
std::unique_ptr<EmulatedArithmetic> expansionArithmetic(NumberFormat format);

} // namespace poseidon
#endif // POSEIDON_TECHNIQUES_PRECISION_EXPANSION_MODEL_H
