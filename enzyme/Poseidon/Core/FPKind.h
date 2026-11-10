//=- FPKind.h - Number formats and rounding to them -----------------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The operand and accumulator formats of a matrix product, and rounding a
// double to them. The matrix accuracy model is computed from these and from
// MatmulAccuracy.cpp alone, and its identity is a digest of those files, so
// nothing else belongs here.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_FPKIND_H
#define POSEIDON_FPKIND_H

namespace poseidon {

enum class FPKind {
  Invalid = 0,
  F16 = 1,
  BF16 = 2,
  TF32 = 3,
  F32 = 4,
  F64 = 5,
  S8 = 6,  // 8-bit signed integer (INT8 tensor-core operand)
  S32 = 7, // 32-bit signed integer (INT8 tensor-core accumulator)
};
const char *fpKindName(FPKind k);

double roundToPrec(double x, FPKind k);
// Smallest positive (subnormal) magnitude representable in `k`, as a double.
// Returns 0 for F64/Invalid/integer kinds (no underflow constraint).
double minSubnormalForKind(FPKind k);

} // namespace poseidon
#endif // POSEIDON_FPKIND_H
