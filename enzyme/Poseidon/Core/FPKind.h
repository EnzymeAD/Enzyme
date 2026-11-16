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
// double to them. The matrix accuracy model is computed from these, from
// MatmulAccuracy.cpp and from Core/UniformReal.cpp alone, and its identity is a
// digest of those files, so nothing else belongs here.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_FPKIND_H
#define POSEIDON_FPKIND_H

#include <cstdint>

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
// roundToPrec(x, k) with the format of `k` looked up once, for a loop that
// rounds many values to it.
class Rounding {
public:
  explicit Rounding(FPKind k);
  double operator()(double x) const;

private:
  FPKind kind;
  // 0 for the kinds every double is already in (F64, Invalid).
  int precision = 0;
  int minExponent = 0;
  double subnormalUlp = 0.0;
  uint64_t maxFinite = 0; // binary64 encoding of the largest finite value
};
// Smallest positive (subnormal) magnitude representable in `k`, as a double.
// Returns 0 for F64/Invalid/integer kinds (no underflow constraint).
double minSubnormalForKind(FPKind k);

} // namespace poseidon
#endif // POSEIDON_FPKIND_H
