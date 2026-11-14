//=- UniformReal.h - Uniform reals that do not depend on the library ------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The uniform reals the accuracy models sample at. The engines are fully
// specified by the C++ standard; the mapping from their output to a real in
// [A, B) is not, so it is written out here.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_CORE_UNIFORM_REAL_H
#define POSEIDON_CORE_UNIFORM_REAL_H

#include <random>

namespace poseidon {

// A real in [A, B) from the next draws of G: two of a minstd_rand0, one of an
// mt19937_64.
double uniformReal(std::minstd_rand0 &G, double A, double B);
double uniformReal(std::mt19937_64 &G, double A, double B);

} // namespace poseidon

#endif // POSEIDON_CORE_UNIFORM_REAL_H
