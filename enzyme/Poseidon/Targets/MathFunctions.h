//===- MathFunctions.h - the math functions the pass knows by name --------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// One row per libm function the pass knows, with a column for each question a
// part of the pass asks about it, so that every list of names is read off one
// table. include/poseidon/poseidon.h keeps its own literal lists, being a
// public header: the functions it links on NVPTX are the Device column, those
// on AMDGPU the Device column without ceil, copysign, fabs, floor, fma, fmax,
// fmin, powi, rint, round, sqrt and trunc.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_MATH_FUNCTIONS_H
#define POSEIDON_MATH_FUNCTIONS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <string>
#include <unordered_set>

namespace poseidon {

struct MathFunction {
  llvm::StringLiteral Name;
  // Priced from the cost model and retyped by a precision change: libmFuncs.
  bool Libm;
  // A call to it, or to its float form with an f suffix, is an operation
  // isOptimizable accepts.
  bool Optimizable;
  // Both GPU targets select its device math library function (mathBases).
  bool Device;
  // The C rendering of a rewrite writes its float form with an f suffix.
  bool FloatSuffixInC;
  // A rewrite writes Herbie's operator of this name as a math-library call on
  // either GPU target: deviceCallsMathLibrary.
  bool DeviceCall;
};

llvm::ArrayRef<MathFunction> mathFunctions();

// The libm calls Poseidon prices and can retype.
const std::unordered_set<std::string> &libmFuncs();

bool isOptimizableLibmCall(llvm::StringRef Name);

bool hasFloatSuffixInC(llvm::StringRef Name);

} // namespace poseidon

#endif // POSEIDON_MATH_FUNCTIONS_H
