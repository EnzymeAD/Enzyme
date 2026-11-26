// RUN: rm -rf %t && mkdir %t
// RUN: %clang++ -std=c++17 -O2 -ffp-contract=off -I%poseidon_runtime_inc %s -o %t/tcec_scale
// RUN: %t/tcec_scale | FileCheck %s

// The TCEC runtime's power-of-two operand scaling (runtime/gemm/SchemeConstants.h),
// run on the host over the whole exponent range.
//
// ldexp: tcecScale(x, n) is x * 2^n rounded once, against the C library's
// ldexp, for finite, subnormal, zero, infinite and NaN x and every n the
// runtime can form.
//
// pairs: a single-entry product 1.5 * 2^eA times 1.25 * 2^eB for every eA, eB
// at which both operands are exact doubles. Each operand is scaled by its
// exponent and narrowed to FP32 exactly, the scaled product is exact, and
// unscaling it gives 1.875 * 2^(eA + eB) rounded once: exact wherever that is
// representable, zero or infinite where the true product is.
//
// before: wherever the FP32 formula (scales exp2f(10 - ilogbf(max)) and the
// unscaling factor 1.f / (sA * sB)) is finite and nonzero, the exponents give
// the same scales and the same unscaling bit for bit.

#include "gemm/SchemeConstants.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

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

int main() {
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

// CHECK: ldexp 4053014 cases, 0 mismatches
// CHECK-NEXT: pairs 4395312, representable 3294261, exact 3294261; not representable 1101051, rounded once 1101051
// CHECK-NEXT: before: scales 735, differing 0; unscalings 415071, differing 0
// CHECK-NEXT: unscaled: 0 0 0
// CHECK-NEXT: PASS
