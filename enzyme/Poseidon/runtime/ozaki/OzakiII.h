//===- OzakiII.h - Ozaki-II modulus and moduli-count tables ---------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared by the compiler's Ozaki-II pricing and accuracy model and by
// runtime/ozaki/ozaki_rt.cu, which includes it by its bare name: keep it free
// of LLVM and of anything beyond C++14.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_OZAKI_II_H
#define POSEIDON_OZAKI_II_H

#include <cstdint>

constexpr unsigned kOzakiIIMaxModuli = 14;

// INT8 moduli in the order the runtime consumes them; nm moduli use the first
// nm entries. p[0] = 256 is even and needs the 128 -> -128 residue fold.
constexpr int32_t kOzakiIIModuli[kOzakiIIMaxModuli] = {
    256, 255, 253, 251, 247, 241, 239, 233, 229, 227, 223, 217, 211, 199};

// Each count needs a measured ozaki_dispatch_rel cost-model row
// (poseidon-calibrate covers 8..14).
constexpr unsigned kOzIIModuliCounts[] = {8, 9, 10, 11, 12, 13, 14};

// Smallest modulus count the runtime accepts: the published scaling constant
// below is defined from nm = 3 on (the reference implementation's nm = 2
// entry does not follow its own formula, so nm = 2 is not offered).
constexpr unsigned kOzakiIIMinModuli = 3;

// Up to this many moduli P is exact in one double and the CRT runs in plain
// FP64; above it the CRT constants are split into a double-double.
constexpr unsigned kOzakiIIPIsDouble = 6;

// Published fast-mode scaling constant, indexed by nm:
// log2P(nm) = round-down-to-float(log2(P - 1)/2 - 1/2), P = the product of the
// first nm moduli, so 2^(2*log2P) <= (P - 1)/2. A row a of A (column b of B)
// is scaled by 2^floor(log2P - log2||a||_2), which by Cauchy-Schwarz keeps
// every entry of the integer product inside the symmetric CRT window.
// Computed in 80-digit decimal arithmetic; each literal round-trips exactly.
constexpr float kOzakiIILog2P[kOzakiIIMaxModuli + 1] = {
    0.0f,        0.0f,        0.0f,        11.4886732f, 15.4744444f,
    19.4486275f, 23.4050732f, 27.3555069f, 31.2875996f, 35.2071991f,
    39.1204758f, 43.0209236f, 46.9016991f, 50.762249f,  54.5805626f};

#endif // POSEIDON_OZAKI_II_H
