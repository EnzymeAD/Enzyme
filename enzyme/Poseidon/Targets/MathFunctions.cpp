//=- MathFunctions.cpp - the math functions the pass knows by name ----------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MathFunctions.h"

#include "llvm/ADT/STLExtras.h"

#include <vector>

using namespace llvm;

namespace poseidon {

namespace {

// Name, Libm, Optimizable, Device, FloatSuffixInC, DeviceCall. libmFuncs is
// built from the Libm rows in this order.
constexpr MathFunction kMathFunctions[] = {
    {"sin", true, true, true, true, true},
    {"cos", true, true, true, true, true},
    {"tan", true, true, true, true, true},
    {"asin", true, true, true, true, true},
    {"acos", true, true, true, true, true},
    {"atan", true, true, true, true, true},
    {"atan2", true, true, true, true, true},
    {"sinh", true, true, true, true, true},
    {"cosh", true, true, true, true, true},
    {"tanh", true, true, true, true, true},
    {"asinh", true, true, true, true, true},
    {"acosh", true, true, true, true, true},
    {"atanh", true, true, true, true, true},
    {"exp", true, true, true, true, true},
    {"log", true, true, true, true, true},
    {"sqrt", true, true, true, true, false},
    {"cbrt", true, true, true, true, true},
    {"pow", true, true, true, true, true},
    {"powi", true, false, true, false, false},
    {"fabs", true, true, true, true, false},
    {"fma", true, true, true, true, false},
    {"hypot", true, true, true, true, false},
    {"expm1", true, true, true, true, true},
    {"log1p", true, true, true, true, true},
    {"ceil", true, true, true, true, false},
    {"floor", true, true, true, true, false},
    {"erf", true, true, true, true, false},
    {"exp2", true, true, true, true, true},
    {"lgamma", true, true, true, true, false},
    {"log10", true, true, true, true, true},
    {"log2", true, true, true, true, true},
    {"rint", true, true, true, true, false},
    {"round", true, true, true, true, false},
    {"tgamma", true, true, true, true, false},
    {"trunc", true, true, true, true, false},
    {"copysign", true, true, true, true, false},
    {"fdim", true, true, true, true, false},
    {"fmod", true, true, true, true, false},
    {"remainder", true, true, true, true, false},
    {"fmin", false, false, true, true, false},
    {"fmax", false, false, true, true, false},
    {"erfc", false, false, false, true, false},
    {"nearbyint", false, false, false, true, false},
};

} // namespace

ArrayRef<MathFunction> mathFunctions() { return kMathFunctions; }

const std::unordered_set<std::string> &libmFuncs() {
  // Inserted in row order, which decides the set's iteration order and so the
  // function getPTFuncs names first when the cost model lacks several.
  static const std::unordered_set<std::string> LibmFuncs = [] {
    std::vector<std::string> Names;
    for (const MathFunction &F : kMathFunctions)
      if (F.Libm)
        Names.push_back(F.Name.str());
    return std::unordered_set<std::string>(Names.begin(), Names.end());
  }();
  return LibmFuncs;
}

bool isOptimizableLibmCall(StringRef Name) {
  auto Optimizable = [](StringRef N) {
    return any_of(kMathFunctions, [&](const MathFunction &F) {
      return F.Optimizable && F.Name == N;
    });
  };
  return Optimizable(Name) ||
         (Name.ends_with("f") && Optimizable(Name.drop_back()));
}

bool hasFloatSuffixInC(StringRef Name) {
  return any_of(kMathFunctions, [&](const MathFunction &F) {
    return F.FloatSuffixInC && F.Name == Name;
  });
}

} // namespace poseidon
