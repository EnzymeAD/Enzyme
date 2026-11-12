//=- ExpansionModel.cpp - Host emulation of FP32 expansions ---------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The routines of ExpansionRoutines.h, which Expansion.cpp emits, on host
// floats: std::fma exactly where the emitter emits llvm.fma.f32. The file is
// compiled without floating-point contraction (CMakeLists.txt), so that no
// build of the compiler fuses a product and a sum the emitted code keeps
// apart.
//
//===----------------------------------------------------------------------===//

#include "ExpansionModel.h"
#include "ExpansionRoutines.h"

#include <cfloat>
#include <cmath>

#if defined(__FAST_MATH__)
#error "ExpansionModel.cpp must be compiled without fast-math"
#endif
static_assert(FLT_EVAL_METHOD == 0,
              "ExpansionModel.cpp needs each float operation rounded to float");

namespace poseidon {

namespace {

struct F2 {
  float hi, lo;
};

// Host floats; the names of the emitted instructions are not used.
struct HostArith {
  using F32 = float;
  using F64 = double;
  using Bool = bool;
  using Pair = F2;

  float add(float a, float b, const char *) { return a + b; }
  double add(double a, double b, const char *) { return a + b; }
  float sub(float a, float b, const char *) { return a - b; }
  double sub(double a, double b, const char *) { return a - b; }
  float mul(float a, float b, const char *) { return a * b; }
  float div(float a, float b, const char *) { return a / b; }
  float neg(float a, const char *) { return -a; }
  // Up to LLVM 22, clang takes fma(x, x, y) for never -0, although a negative
  // sum that underflows is -0, and then rewrites a later b - (s - a) into
  // b + (a - s), which turns a -0 into +0. Read through a volatile, the result
  // is what IEEE arithmetic gives whatever compiler builds the pass.
  static float unanalyzed(float v) {
    volatile float r = v;
    return r;
  }
  float fma(float a, float b, float c, const char *) {
    return unanalyzed(std::fma(a, b, c));
  }
  float fmaCall(float a, float b, float c, const char *) {
    return unanalyzed(std::fma(a, b, c));
  }
  float sqrt(float a, const char *) { return std::sqrt(a); }
  float sqrtCall(float a, const char *) { return std::sqrt(a); }
  float constant(double v) { return (float)v; }
  bool equal(float a, float b, const char *) { return a == b; }
  bool both(bool a, bool b, const char *) { return a && b; }
  float select(bool c, float t, float f, const char *) { return c ? t : f; }
  float truncate(double v, const char *) { return (float)v; }
  double extend(float v, const char *) { return (double)v; }
};

using Routines = ExpansionRoutines<HostArith>;
Routines routines() { return Routines(HostArith()); }

F2 pair(const ExpansionLimbs &a) { return {a.x[0], a.x[1]}; }
ExpansionLimbs limbs2(F2 r) {
  ExpansionLimbs c;
  c.n = 2;
  c.x[0] = r.hi;
  c.x[1] = r.lo;
  return c;
}
ExpansionLimbs sized(unsigned n) {
  ExpansionLimbs c;
  c.n = n;
  return c;
}

} // namespace

ExpansionLimbs ExpansionLimbs::of(const float *limbs, unsigned n) {
  ExpansionLimbs r;
  r.n = n;
  for (unsigned i = 0; i < n; ++i)
    r.x[i] = limbs[i];
  return r;
}

ExpansionLimbs expansionFromF64(double v, unsigned n) {
  if (n == 2)
    return limbs2(routines().dsFromF64(v));
  ExpansionLimbs r = sized(n);
  routines().expansionFromF64(n, v, r.x);
  return r;
}

ExpansionLimbs expansionFromF32(float v, unsigned n) {
  if (n == 2)
    return limbs2(routines().dsFromF32(v));
  ExpansionLimbs r = sized(n);
  routines().expansionFromF32(n, v, r.x);
  return r;
}

double expansionToF64(const ExpansionLimbs &a) {
  if (a.n == 2)
    return routines().dsToF64(pair(a));
  return routines().expansionToF64(a.n, a.x);
}

float expansionToF32(const ExpansionLimbs &a) {
  if (a.n == 2)
    return routines().dsToF32(pair(a));
  return routines().expansionToF32(a.n, a.x);
}

ExpansionLimbs expansionNeg(const ExpansionLimbs &a) {
  if (a.n == 2)
    return limbs2(routines().dsNeg(pair(a)));
  ExpansionLimbs c = sized(a.n);
  routines().expansionNeg(a.n, a.x, c.x);
  return c;
}

ExpansionLimbs expansionAdd(const ExpansionLimbs &a, const ExpansionLimbs &b) {
  if (a.n == 2)
    return limbs2(routines().dsAdd(pair(a), pair(b)));
  ExpansionLimbs c = sized(a.n);
  routines().expansionAdd(a.n, a.x, b.x, c.x);
  return c;
}

ExpansionLimbs expansionSub(const ExpansionLimbs &a, const ExpansionLimbs &b) {
  if (a.n == 2)
    return limbs2(routines().dsSub(pair(a), pair(b)));
  ExpansionLimbs c = sized(a.n);
  routines().expansionSub(a.n, a.x, b.x, c.x);
  return c;
}

ExpansionLimbs expansionMul(const ExpansionLimbs &a, const ExpansionLimbs &b) {
  if (a.n == 2)
    return limbs2(routines().dsMul(pair(a), pair(b)));
  ExpansionLimbs c = sized(a.n);
  routines().expansionMul(a.n, a.x, b.x, c.x);
  return c;
}

ExpansionLimbs expansionSqr(const ExpansionLimbs &a) {
  if (a.n == 2)
    return limbs2(routines().dsMul(pair(a), pair(a)));
  ExpansionLimbs c = sized(a.n);
  routines().expansionSqr(a.n, a.x, c.x);
  return c;
}

ExpansionLimbs expansionDiv(const ExpansionLimbs &a, const ExpansionLimbs &b) {
  if (a.n == 2)
    return limbs2(routines().dsDiv(pair(a), pair(b)));
  ExpansionLimbs c = sized(a.n);
  routines().expansionDiv(a.n, a.x, b.x, c.x);
  return c;
}

ExpansionLimbs expansionSqrt(const ExpansionLimbs &a) {
  if (a.n == 2)
    return limbs2(routines().dsSqrt(pair(a)));
  ExpansionLimbs c = sized(a.n);
  routines().expansionSqrt(a.n, a.x, c.x);
  return c;
}

ExpansionLimbs expansionFma(const ExpansionLimbs &a, const ExpansionLimbs &b,
                            const ExpansionLimbs &c) {
  if (a.n == 2)
    return limbs2(routines().dsFma(pair(a), pair(b), pair(c)));
  return expansionAdd(expansionMul(a, b), c);
}

ExpansionLimbs expansionFmaSqr(const ExpansionLimbs &a,
                               const ExpansionLimbs &c) {
  if (a.n == 2)
    return limbs2(routines().dsFma(pair(a), pair(a), pair(c)));
  return expansionAdd(expansionSqr(a), c);
}

} // namespace poseidon
