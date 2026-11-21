// Operands for the expansion_*_identity.cpp tests: n FP32 components in eight
// classes, each drawn from a fixed seed so that every run sees the same cases.
#ifndef EXPANSION_MATCH_GEN_H
#define EXPANSION_MATCH_GEN_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>

namespace expansion_match {

enum { kClasses = 8 };
inline const char *className(int c) {
  static const char *names[kClasses] = {"moderate",  "wide",  "near-overflow",
                                        "subnormal", "exact", "not-normalized",
                                        "specials",  "zeros"};
  return names[c];
}

struct Gen {
  std::mt19937_64 rng;
  explicit Gen(uint64_t seed) : rng(seed) {}
  int ui(int a, int b) { return std::uniform_int_distribution<int>(a, b)(rng); }
  // A float in [1, 2) with a random significand.
  float mant() {
    uint32_t u = 0x3f800000u | (uint32_t)(rng() & 0x7fffffu);
    float f;
    std::memcpy(&f, &u, 4);
    return f;
  }
  float sgn(float x) { return (rng() & 1) ? -x : x; }
  // A normalized expansion with leading exponent e0; a lower component is
  // zero one time in sixteen.
  void normalized(float *x, int n, int e0) {
    x[0] = sgn(std::ldexp(mant(), e0));
    for (int i = 1; i < n; ++i) {
      if (x[i - 1] == 0.0f || rng() % 16 == 0) {
        x[i] = 0.0f;
        continue;
      }
      x[i] = sgn(std::ldexp(mant(), std::ilogb(x[i - 1]) - 24 - ui(0, 3)));
    }
  }
  void make(int cls, float *x, int n) {
    switch (cls) {
    case 0: // moderate exponents
      normalized(x, n, ui(-20, 20));
      break;
    case 1: // the whole exponent range, components reaching subnormals
      normalized(x, n, ui(-125, 126));
      break;
    case 2: // near overflow
      normalized(x, n, ui(118, 127));
      break;
    case 3: // subnormal or near it
      normalized(x, n, ui(-149, -110));
      break;
    case 4: // small exact values, signed zero lower components
      x[0] = sgn((float)ui(0, 64) * 0.25f);
      for (int i = 1; i < n; ++i)
        x[i] = (rng() & 1) ? -0.0f : 0.0f;
      break;
    case 5: // overlapping components
      for (int i = 0; i < n; ++i)
        x[i] = sgn(std::ldexp(mant(), ui(-8, 8)));
      break;
    case 6: { // an infinity, a NaN or a signed zero in one component
      normalized(x, n, ui(-10, 10));
      static const float sp[] = {INFINITY, -INFINITY, NAN, -NAN, 0.0f, -0.0f};
      x[ui(0, n - 1)] = sp[ui(0, 5)];
      break;
    }
    case 7: // zeros with every sign pattern
      for (int i = 0; i < n; ++i)
        x[i] = (rng() & 1) ? -0.0f : 0.0f;
      break;
    }
  }
};

struct Case {
  float a[4], b[4], c[4];
  int cls;
};

// Cases for n components: `perClass` per class, one in eight of the first four
// classes a near-cancellation b = -a with a perturbed last component.
template <typename Fn> void forEachCase(int n, int perClass, Fn fn) {
  Gen g(0x5eed2026u + n);
  for (int cls = 0; cls < kClasses; ++cls)
    for (int t = 0; t < perClass; ++t) {
      Case k{};
      k.cls = cls;
      g.make(cls, k.a, n);
      g.make(cls, k.b, n);
      g.make(cls, k.c, n);
      if (t % 8 == 0 && cls <= 3) {
        for (int i = 0; i < n; ++i)
          k.b[i] = -k.a[i];
        int e = k.a[0] == 0.0f ? 0 : std::ilogb(k.a[0]);
        k.b[n - 1] = g.sgn(std::ldexp(
            g.mant(), std::max(-140, e - 24 * (n - 1) - g.ui(0, 4))));
      }
      fn(k);
    }
}

// Two floats are the same result when their bits are equal or both are NaN;
// the materializer does not keep NaN payloads or signs.
inline bool sameFloat(float x, float y) {
  uint32_t a, b;
  std::memcpy(&a, &x, 4);
  std::memcpy(&b, &y, 4);
  return a == b || (x != x && y != y);
}
inline bool sameDouble(double x, double y) {
  uint64_t a, b;
  std::memcpy(&a, &x, 8);
  std::memcpy(&b, &y, 8);
  return a == b || (x != x && y != y);
}

} // namespace expansion_match

#endif // EXPANSION_MATCH_GEN_H
