//===- UniformReal.h - Uniform reals that do not depend on the library ----===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The uniform reals the accuracy models sample at. The engines are fully
// specified by the C++ standard; the mapping from their output to a real in
// [A, B) is not, so it is written out here.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_SUPPORT_UNIFORM_REAL_H
#define POSEIDON_SUPPORT_UNIFORM_REAL_H

#include <random>

namespace poseidon {

/// A real in [A, B) from the next two draws of `G`, as libstdc++ 14's
/// std::uniform_real_distribution<double> computes it.
double uniformReal(std::minstd_rand0 &G, double A, double B);
/// A real in [A, B) from the next draw of `G`, computed the same way.
double uniformReal(std::mt19937_64 &G, double A, double B);

} // namespace poseidon

#endif // POSEIDON_SUPPORT_UNIFORM_REAL_H
