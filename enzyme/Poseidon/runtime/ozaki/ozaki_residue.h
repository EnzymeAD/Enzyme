//===- ozaki_residue.h - Ozaki-II residues by byte-weight dot products ----===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The symmetric residue modulo p = kOzakiIIModuli[i] of a 64-bit integer (a
// scaled operand entry) or a 32-bit one (an INT32 product entry), as
// runtime/ozaki/ozaki_rt.cu computes it, in host and device code. The method
// is GEMMul8's (reduce_mant and mod_small_nowrap in src/oz2/mod/mod_core.hpp):
// the bytes of |a| are dotted with the byte weights 2^(8k) mod p, four per
// 32-bit word (one __dp4a each on CUDA devices), and the sum s is reduced with
// one 32-bit multiply-high by m = floor(2^32/p) + 1. That quotient is
// floor(s/p) whenever s * (p - 2^32 mod p) < 2^32; the largest sum the eight
// bytes of any 64-bit magnitude give meets that bound for every modulus, which
// a static_assert checks, so the residue is exact for every input.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_OZAKI_RESIDUE_H
#define POSEIDON_OZAKI_RESIDUE_H

#include "OzakiII.h"

#include <cstdint>

#if defined(__CUDACC__) || defined(__HIP__)
#define POZ_HOST_DEVICE __host__ __device__
#else
#define POZ_HOST_DEVICE
#endif

// Modulus i with its byte weights: byte k of wlo is 2^(8k) mod p, byte k of
// whi is 2^(32+8k) mod p (each below p <= 256), and m32 = floor(2^32/p) + 1.
struct PozResidueModulus {
  uint32_t p, wlo, whi, m32;
};

POZ_HOST_DEVICE constexpr PozResidueModulus pozResidueModulus(unsigned i) {
  const uint32_t p = (uint32_t)kOzakiIIModuli[i];
  uint32_t wlo = 0, whi = 0;
  for (unsigned k = 0; k < 4; ++k) {
    wlo |= (uint32_t)((1ull << (8 * k)) % p) << (8 * k);
    whi |= (uint32_t)((1ull << (32 + 8 * k)) % p) << (8 * k);
  }
  return PozResidueModulus{p, wlo, whi, (uint32_t)((1ull << 32) / p + 1)};
}

// Every modulus: the largest byte-weight sum of a 64-bit magnitude, s_max =
// 255 * (sum of the eight weights), satisfies s_max * (p - 2^32 mod p) < 2^32.
constexpr bool pozResiduesExact() {
  for (unsigned i = 0; i < kOzakiIIMaxModuli; ++i) {
    const PozResidueModulus m = pozResidueModulus(i);
    uint64_t sMax = 0;
    for (unsigned k = 0; k < 4; ++k)
      sMax += 255 * (((m.wlo >> (8 * k)) & 255) + ((m.whi >> (8 * k)) & 255));
    if (sMax * (m.p - (1ull << 32) % m.p) >= (1ull << 32))
      return false;
  }
  return true;
}
static_assert(pozResiduesExact(),
              "the multiply-high reduction is not exact on every byte sum");

// c + sum_k byte_k(a) * byte_k(w), bytes unsigned.
POZ_HOST_DEVICE inline uint32_t poz_byteDot(uint32_t a, uint32_t w,
                                            uint32_t c) {
#ifdef __CUDA_ARCH__
  return __dp4a(a, w, c);
#else
  for (unsigned k = 0; k < 32; k += 8)
    c += ((a >> k) & 255) * ((w >> k) & 255);
  return c;
#endif
}

// s mod p for a byte-weight sum s of m.
POZ_HOST_DEVICE inline uint32_t poz_modSum(uint32_t s, PozResidueModulus m) {
  return s - m.p * (uint32_t)(((uint64_t)s * m.m32) >> 32);
}

// The representative of r = |a| mod p for a of sign `neg`: [-(p-1)/2, (p-1)/2]
// for odd p, [-128, 127] for p = 256 (the int8 wraparound of the published
// scheme). The CRT constants assume these.
POZ_HOST_DEVICE inline int poz_symmetric(uint32_t r, bool neg, uint32_t p) {
  const uint32_t t = (neg && r != 0) ? p - r : r;
  return (int)t - (t > (p - 1) / 2 ? (int)p : 0);
}

// The residue of a scaled operand entry, an INT8 plane element.
POZ_HOST_DEVICE inline int8_t poz_modI8(int64_t a, PozResidueModulus m) {
  const uint64_t u = a < 0 ? 0 - (uint64_t)a : (uint64_t)a;
  const uint32_t s = poz_byteDot((uint32_t)u, m.wlo,
                                 poz_byteDot((uint32_t)(u >> 32), m.whi, 0));
  return (int8_t)poz_symmetric(poz_modSum(s, m), a < 0, m.p);
}

// The residue of an INT32 product entry.
POZ_HOST_DEVICE inline int poz_modI32(int32_t a, PozResidueModulus m) {
  const uint32_t u = a < 0 ? 0 - (uint32_t)a : (uint32_t)a;
  return poz_symmetric(poz_modSum(poz_byteDot(u, m.wlo, 0), m), a < 0, m.p);
}

#undef POZ_HOST_DEVICE

#endif // POSEIDON_OZAKI_RESIDUE_H
