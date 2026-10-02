//===- OzakiII.h - Ozaki-II modulus and moduli-count tables ---------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared by the compiler's Ozaki-II pricing and by Runtimes/OzakiRT, which
// includes it by relative path and is compiled by the application's clang:
// keep it free of LLVM and of anything beyond C++14.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_OZAKI_II_H
#define POSEIDON_OZAKI_II_H

#include <cmath>
#include <cstdint>

constexpr unsigned kOzakiIIMaxModuli = 14;

// INT8 moduli in the order the runtime consumes them; nm moduli use the first
// nm entries. p[0] = 256 is even and needs the 128 -> -128 residue fold.
constexpr int32_t kOzakiIIModuli[kOzakiIIMaxModuli] = {
    256, 255, 253, 251, 247, 241, 239, 233, 229, 227, 223, 217, 211, 199};

// Each count needs a measured ozaki_dispatch_rel cost-model row
// (poseidon-calibrate covers 8..14).
constexpr unsigned kOzIIModuliCounts[] = {8, 9, 10, 11, 12, 13, 14};

// Bits captured per operand at modulus count nm for a reduction of length
// 2^log2K: beta = clamp((bitlen(P) - 3 - ceil(log2 K)) / 2, 1, 50), the same
// scaling ozaki_rt.cu applies (pozComputeConsts + poz_run_square).
inline long ozakiIICapturedBits(unsigned nm, unsigned log2K) {
  double l2 = 0.0;
  for (unsigned i = 0; i < nm; ++i)
    l2 += std::log2((double)kOzakiIIModuli[i]);
  long log2P = (long)std::floor(l2) + 1;
  long bits = (log2P - 3 - (long)log2K) / 2;
  if (bits > 50)
    bits = 50;
  if (bits < 1)
    bits = 1;
  return bits;
}

#endif // POSEIDON_OZAKI_II_H
