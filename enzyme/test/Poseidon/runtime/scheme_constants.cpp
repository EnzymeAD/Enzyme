// RUN: %clang++ -std=c++17 -O2 -ffp-contract=off -I%poseidon_runtime_inc %s -o %t
// RUN: %t | FileCheck %s

// The scheme constants of the GEMM runtimes (runtime/gemm/SchemeConstants.h)
// on the host, which the HIP build computes with this code and the CUDA build
// with __dp4a for the dot products.
// Ozaki-II residues, for every modulus: byte k of the weight words is 2^(8k)
// mod p; the multiply-high reduction is s % p for every s < 2^24; the int64
// and int32 residues are the symmetric representative ([-(p-1)/2, (p-1)/2],
// [-128, 127] for p = 256) of a % p for every |a| < 2^24, +-(2^b + d),
// |d| <= 2, the extremes and 2^20 random values.
// TCEC scaling: tcecScale(x, n) is x * 2^n rounded once (ldexp) for every
// class of x and every n the runtime forms; a product 1.5 * 2^eA times
// 1.25 * 2^eB scaled, narrowed to FP32 and unscaled is 1.875 * 2^(eA + eB)
// rounded once; and wherever the FP32 formula (exp2f(10 - ilogbf(max)),
// 1.f / (sA * sB)) is finite and nonzero the exponents give it bit for bit.

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
// CHECK-NEXT: ldexp 4053014 cases, 0 mismatches
// CHECK-NEXT: pairs 4395312, representable 3294261, exact 3294261; not representable 1101051, rounded once 1101051
// CHECK-NEXT: before: scales 735, differing 0; unscalings 415071, differing 0
// CHECK-NEXT: unscaled: 0 0 0
// CHECK-NEXT: PASS

#include "gemm/SchemeConstants.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

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

bool weightsOk(const OzakiIIModulus &m) {
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

static void ozakiResidues() {
  std::mt19937_64 rng(0x6f7a616b69ull);
  for (unsigned i = 0; i < kOzakiIIMaxModuli; ++i) {
    const OzakiIIModulus m = ozakiIIModulus(i);
    const int p = (int)m.p;
    size_t sums = 0, i64 = 0, i32 = 0;
    for (uint32_t s = 0; s < (1u << 24); ++s)
      sums += ozakiIIReduceByteSum(s, m) != s % m.p;
    auto check64 = [&](int64_t a) {
      i64 += ozakiIIResidueI8(a, m) != symmetric(a, p);
    };
    auto check32 = [&](int32_t a) {
      i32 += ozakiIIResidueI32(a, m) != symmetric(a, p);
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
}

static uint64_t bitsOf(double d) {
  uint64_t b;
  memcpy(&b, &d, sizeof b);
  return b;
}

static bool sameBits(double a, double b) {
  return (std::isnan(a) && std::isnan(b)) || bitsOf(a) == bitsOf(b);
}

static uint64_t g_state = 0x9e3779b97f4a7c15ull;
static uint64_t next() {
  g_state = g_state * 6364136223846793005ull + 1442695040888963407ull;
  return g_state;
}

static int tcecScaling() {
  int failed = 0;

  // ldexp
  std::vector<double> xs = {0.0,
                            -0.0,
                            std::ldexp(1.0, -1074),
                            -std::ldexp(3.0, -1074),
                            std::ldexp(0x1.fffffffffffffp0, -1023),
                            std::ldexp(1.0, -1022),
                            1.0,
                            -1.5,
                            0x1.fffffffffffffp0,
                            std::ldexp(0x1.fffffffffffffp0, 1023),
                            -std::ldexp(1.0, 1023),
                            INFINITY,
                            -INFINITY,
                            NAN};
  for (int e = -1074; e <= 1023; e += 7)
    for (int k = 0; k < 2; ++k) {
      double m = 1.0 + (double)(next() >> 11) * 0x1.0p-53;
      xs.push_back((k ? -1.0 : 1.0) * std::ldexp(m, e));
    }
  long ldexpCases = 0, ldexpBad = 0;
  for (double x : xs)
    for (int n = -3300; n <= 3300; ++n) {
      ++ldexpCases;
      double got = tcecScale(x, n), want = std::ldexp(x, n);
      if (!sameBits(got, want) && ldexpBad++ < 5)
        printf("ldexp MISMATCH x=%a n=%d got=%a want=%a\n", x, n, got, want);
    }
  printf("ldexp %ld cases, %ld mismatches\n", ldexpCases, ldexpBad);
  failed |= ldexpBad != 0;

  // pairs
  long pairs = 0, representable = 0, exact = 0, rounded = 0, bad = 0;
  for (int eA = -1073; eA <= 1023; ++eA) {
    const double a = std::ldexp(1.5, eA);
    const int na = tcecScaleExponent(a);
    const double aScaled = tcecScale(a, na);
    const float aNarrow = (float)aScaled;
    if ((double)aNarrow != aScaled || aNarrow < 0x1.0p9f ||
        aNarrow > 0x1.0p11f) {
      printf("pairs A NOT SCALED EXACTLY eA=%d n=%d %a\n", eA, na, aScaled);
      failed = 1;
      continue;
    }
    for (int eB = -1072; eB <= 1023; ++eB) {
      const double b = std::ldexp(1.25, eB);
      const int nb = tcecScaleExponent(b);
      const float bNarrow = (float)tcecScale(b, nb);
      const double product = (double)aNarrow * (double)bNarrow;
      const double got = tcecScale(product, -(na + nb));
      const double want = std::ldexp(1.875, eA + eB);
      const int e = eA + eB;
      ++pairs;
      // 1.875 * 2^e has four significant bits, the last at 2^(e - 3).
      const bool isRep = e >= -1074 + 3 && e <= 1023;
      representable += isRep;
      if (sameBits(got, want)) {
        exact += isRep;
        rounded += !isRep;
      } else if (bad++ < 5) {
        printf("pairs MISMATCH eA=%d eB=%d got=%a want=%a\n", eA, eB, got,
               want);
      }
    }
  }
  printf("pairs %ld, representable %ld, exact %ld; not representable "
         "%ld, rounded once %ld\n",
         pairs, representable, exact, pairs - representable, rounded);
  failed |= bad != 0 || exact != representable;

  // before
  std::vector<float> maxima;
  for (int e = -149; e <= 127; ++e)
    for (float m : {1.0f, 1.5f, 1.9999999f}) {
      float v = std::ldexp(m, e);
      if (v != 0.f && std::ilogb(v) == e)
        maxima.push_back(v);
    }
  long scaleCases = 0, scaleBad = 0, unscaleCases = 0, unscaleBad = 0;
  for (float ma : maxima) {
    const float sa = exp2f((float)(10 - ilogbf(ma)));
    if (!std::isfinite(sa))
      continue;
    const int na = tcecScaleExponent((double)ma);
    ++scaleCases;
    bool ok = na == 10 - ilogbf(ma);
    for (int k = 0; k < 16 && ok; ++k) {
      double x = (double)ma * ((double)(next() >> 11) * 0x1.0p-53);
      ok = sameBits((double)(float)(x * (double)sa),
                    (double)(float)tcecScale(x, na));
    }
    scaleBad += !ok;
    for (float mb : maxima) {
      const float sb = exp2f((float)(10 - ilogbf(mb)));
      if (!std::isfinite(sb))
        continue;
      const float s2 = 1.f / (sa * sb);
      if (!(s2 > 0.f && std::isfinite(s2)))
        continue;
      const int nb = tcecScaleExponent((double)mb);
      ++unscaleCases;
      bool same = true;
      for (int k = 0; k < 4 && same; ++k) {
        double v = (double)(float)std::ldexp((double)(next() >> 11) * 0x1.0p-53,
                                             (int)(next() % 64) - 20);
        same = sameBits(v * (double)s2, tcecScale(v, -(na + nb)));
      }
      unscaleBad += !same;
    }
  }
  printf("before: scales %ld, differing %ld; unscalings %ld, differing %ld\n",
         scaleCases, scaleBad, unscaleCases, unscaleBad);
  failed |= scaleBad != 0 || unscaleBad != 0;

  // Operands with no finite nonzero maximum are left unscaled.
  printf("unscaled: %d %d %d\n", tcecScaleExponent(0.0),
         tcecScaleExponent(INFINITY), tcecScaleExponent(NAN));

  printf(failed ? "FAIL\n" : "PASS\n");
  return failed;
}

int main() {
  ozakiResidues();
  return tcecScaling();
}
