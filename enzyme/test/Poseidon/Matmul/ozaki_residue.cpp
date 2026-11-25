// The residue arithmetic of the Ozaki-II runtime
// (runtime/ozaki/OzakiII.h) on the host, against plain remainders, for
// every modulus. The runtime's HIP build computes it with this code; its CUDA
// build differs only in taking each four-byte dot product from __dp4a.
//   weights: byte k of the weight words is 2^(8k) mod p, k = 0..7, by repeated
//            doubling;
//   sums:    the multiply-high reduction is s % p for every s < 2^24, which
//            holds every byte-weight sum (< 2^20);
//   int64:   poz_modI8 is the symmetric representative of a % p for every
//            |a| < 2^24, every a = +-(2^b + d) with |d| <= 2 and 0 <= b < 63,
//            the extremes, and 2^20 random 64-bit values;
//   int32:   poz_modI32 likewise for every |a| < 2^24, the extremes and 2^20
//            random 32-bit values.
// The representatives are [-(p-1)/2, (p-1)/2] for odd p and [-128, 127] for
// p = 256, which the CRT constants assume.
//
// RUN: %clang++ -std=c++17 -O2 -I%poseidon_runtime_inc %s -o %t
// RUN: %t | FileCheck %s

// CHECK: p=256 weights ok sums 0 differ int64 0 differ int32 0 differ
// CHECK-NEXT: p=255 weights ok sums 0 differ int64 0 differ int32 0 differ
// CHECK-NEXT: p=253 weights ok sums 0 differ int64 0 differ int32 0 differ
// CHECK-NEXT: p=251 weights ok sums 0 differ int64 0 differ int32 0 differ
// CHECK-NEXT: p=247 weights ok sums 0 differ int64 0 differ int32 0 differ
// CHECK-NEXT: p=241 weights ok sums 0 differ int64 0 differ int32 0 differ
// CHECK-NEXT: p=239 weights ok sums 0 differ int64 0 differ int32 0 differ
// CHECK-NEXT: p=233 weights ok sums 0 differ int64 0 differ int32 0 differ
// CHECK-NEXT: p=229 weights ok sums 0 differ int64 0 differ int32 0 differ
// CHECK-NEXT: p=227 weights ok sums 0 differ int64 0 differ int32 0 differ
// CHECK-NEXT: p=223 weights ok sums 0 differ int64 0 differ int32 0 differ
// CHECK-NEXT: p=217 weights ok sums 0 differ int64 0 differ int32 0 differ
// CHECK-NEXT: p=211 weights ok sums 0 differ int64 0 differ int32 0 differ
// CHECK-NEXT: p=199 weights ok sums 0 differ int64 0 differ int32 0 differ

#include "ozaki/OzakiII.h"

#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>

namespace {

// The symmetric representative of a modulo p from C++'s truncating remainder.
int symmetric(int64_t a, int p) {
  int r = (int)(a % p), h = p / 2;
  if (r > h)
    r -= p;
  else if (r < -h)
    r += p;
  if ((p & 1) == 0 && r == h)
    r -= p;
  return r;
}

bool weightsOk(const PozResidueModulus &m) {
  uint32_t w = 1;
  for (unsigned k = 0; k < 8; ++k) {
    const uint32_t word = k < 4 ? m.wlo : m.whi;
    if (((word >> (8 * (k % 4))) & 255) != w % m.p)
      return false;
    w = (uint32_t)(((uint64_t)(w % m.p) << 8) % m.p);
  }
  return m.m32 == (uint32_t)((1ull << 32) / m.p + 1);
}

} // namespace

int main() {
  std::mt19937_64 rng(0x6f7a616b69ull);
  for (unsigned i = 0; i < kOzakiIIMaxModuli; ++i) {
    const PozResidueModulus m = pozResidueModulus(i);
    const int p = (int)m.p;
    size_t sums = 0, i64 = 0, i32 = 0;
    for (uint32_t s = 0; s < (1u << 24); ++s)
      sums += poz_modSum(s, m) != s % m.p;
    auto check64 = [&](int64_t a) {
      i64 += poz_modI8(a, m) != symmetric(a, p);
    };
    auto check32 = [&](int32_t a) {
      i32 += poz_modI32(a, m) != symmetric(a, p);
    };
    for (int64_t a = -(1 << 24) + 1; a < (1 << 24); ++a) {
      check64(a);
      check32((int32_t)a);
    }
    for (int b = 0; b < 63; ++b)
      for (int64_t d = -2; d <= 2; ++d) {
        check64(((int64_t)1 << b) + d);
        check64(-(((int64_t)1 << b) + d));
      }
    check64(std::numeric_limits<int64_t>::min());
    check64(std::numeric_limits<int64_t>::max());
    check32(std::numeric_limits<int32_t>::min());
    check32(std::numeric_limits<int32_t>::max());
    for (int n = 0; n < (1 << 20); ++n) {
      const uint64_t x = rng();
      check64((int64_t)x);
      check32((int32_t)(uint32_t)x);
    }
    printf(
        "p=%d weights %s sums %zu differ int64 %zu differ int32 %zu differ\n",
        p, weightsOk(m) ? "ok" : "WRONG", sums, i64, i32);
  }
  return 0;
}
