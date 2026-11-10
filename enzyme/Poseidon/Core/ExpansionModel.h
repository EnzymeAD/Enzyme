//=- ExpansionModel.h - Host emulation of FP32 expansions -----------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
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

// An n-component FP32 expansion, n = 2, 3, 4, most significant component
// first; components past n are zero.
struct ExpansionLimbs {
  float x[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  unsigned n = 0;
  static ExpansionLimbs of(const float *limbs, unsigned n);
};

// The boundary conversions (emitToDS / emitToExpansion and emitDSToFP /
// emitExpansionToFP), by the type of the value converted.
ExpansionLimbs expansionFromF64(double v, unsigned n);
ExpansionLimbs expansionFromF32(float v, unsigned n);
double expansionToF64(const ExpansionLimbs &a);
float expansionToF32(const ExpansionLimbs &a);

ExpansionLimbs expansionNeg(const ExpansionLimbs &a);
ExpansionLimbs expansionAdd(const ExpansionLimbs &a, const ExpansionLimbs &b);
ExpansionLimbs expansionSub(const ExpansionLimbs &a, const ExpansionLimbs &b);
ExpansionLimbs expansionMul(const ExpansionLimbs &a, const ExpansionLimbs &b);
// a * a where the materializer multiplies a value by itself: the square
// routine for n >= 3, the general product for n = 2.
ExpansionLimbs expansionSqr(const ExpansionLimbs &a);
ExpansionLimbs expansionDiv(const ExpansionLimbs &a, const ExpansionLimbs &b);
ExpansionLimbs expansionSqrt(const ExpansionLimbs &a);
// fma/fmuladd(a, b, c), and fma/fmuladd(a, a, c).
ExpansionLimbs expansionFma(const ExpansionLimbs &a, const ExpansionLimbs &b,
                            const ExpansionLimbs &c);
ExpansionLimbs expansionFmaSqr(const ExpansionLimbs &a,
                               const ExpansionLimbs &c);

} // namespace poseidon
#endif // POSEIDON_EXPANSIONMODEL_H
