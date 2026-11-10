// tcec_scale.h: the TCEC runtime's power-of-two operand scaling, carried as
// integer exponents. Compiles for the host as well, so that the whole exponent
// range can be checked without a device.
#ifndef POSEIDON_TCEC_SCALE_H
#define POSEIDON_TCEC_SCALE_H

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__) || defined(__HIP__)
#define PTC_HD __host__ __device__
#else
#define PTC_HD
#endif

// Exponent n of the scale 2^n that brings an operand whose largest magnitude is
// m to [2^10, 2^11]: inside FP16's range with headroom for the high limb of the
// split, and far above its subnormals. The exponent is taken from m rounded to
// FP32, the operand the split sees, wherever that is finite and nonzero, and
// from m itself beyond FP32's range. An operand that is zero, or holds an Inf
// or NaN, is left unscaled.
PTC_HD inline int ptc_scaleExponent(double m) {
  float f = (float)m;
  if (f > 0.f && f <= FLT_MAX)
    return 10 - ilogbf(f);
  if (m > 0. && m <= DBL_MAX)
    return 10 - ilogb(m);
  return 0;
}

// 2^n for n in [-1022, 1023].
PTC_HD inline double ptc_pow2(int n) {
  uint64_t bits = (uint64_t)(n + 1023) << 52;
  double d;
  memcpy(&d, &bits, sizeof d);
  return d;
}

// x * 2^n, rounded once, for any n. A factor beyond the normal range is applied
// in parts; toward the subnormals each part leaves the intermediate 2^53 above
// them, so only the last product can round.
PTC_HD inline double ptc_ldexp(double x, int n) {
  for (int i = 0; i < 2 && n > 1023; ++i) {
    x *= ptc_pow2(1023);
    n -= 1023;
  }
  for (int i = 0; i < 2 && n < -1022; ++i) {
    x *= ptc_pow2(-1022 + 53);
    n += 1022 - 53;
  }
  n = n > 1023 ? 1023 : n < -1022 ? -1022 : n;
  return x * ptc_pow2(n);
}

#endif // POSEIDON_TCEC_SCALE_H
