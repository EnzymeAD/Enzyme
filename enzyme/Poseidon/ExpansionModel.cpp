//=- ExpansionModel.cpp - Host emulation of FP32 expansions ---------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Every routine is the twin of an emitter in Expansion.cpp: the same
// operations in the same order and association, std::fma exactly where the
// emitter emits llvm.fma.f32. The file is compiled without floating-point
// contraction (CMakeLists.txt), so that no build of the compiler fuses a
// product and a sum the emitted code keeps apart.
//
//===----------------------------------------------------------------------===//

#include "ExpansionModel.h"

#include <cfloat>
#include <cmath>
#include <limits>

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
// emitTwoSum
F2 expansionTwoSum(float a, float b) {
  float s = a + b, ap = s - b, bp = s - ap;
  return {s, (a - ap) + (b - bp)};
}
// ExpansionBuilder::fastTwoSum (QxW's form), the n >= 3 arm's.
F2 expansionFastTwoSum(float a, float b) {
  float s = a + b;
  return {s, (a - s) + b};
}
// emitTwoProdFMA
F2 expansionTwoProd(float a, float b) {
  float p = a * b;
  return {p, std::fma(a, b, -p)};
}

// Two components. The arm keeps MultiFloats.jl's FastTwoSum form.
F2 dsFastTwoSum(float a, float b) {
  float s = a + b;
  return {s, b - (s - a)};
}
F2 dsNeg(F2 x) { return {-x.hi, -x.lo}; }
// emitDSAdd
F2 dsAdd(F2 x, F2 y) {
  F2 ab = expansionTwoSum(x.hi, y.hi), cd = expansionTwoSum(x.lo, y.lo);
  F2 ac = dsFastTwoSum(ab.hi, cd.hi);
  return dsFastTwoSum(ac.hi, (ab.lo + cd.lo) + ac.lo);
}
F2 dsSub(F2 x, F2 y) { return dsAdd(x, dsNeg(y)); }
// emitDSMul
F2 dsMul(F2 x, F2 y) {
  F2 pe = expansionTwoProd(x.hi, y.hi);
  float c1 = x.hi * y.lo;
  float c2 = x.lo * y.hi;
  return dsFastTwoSum(pe.hi, pe.lo + (c1 + c2));
}
// emitDSFMA
F2 dsFma(F2 x, F2 y, F2 z) {
  float p = x.hi * y.hi;
  float e = std::fma(x.hi, y.hi, -p);
  e = std::fma(x.hi, y.lo, e);
  e = std::fma(x.lo, y.hi, e);
  F2 s = expansionTwoSum(p, z.hi);
  return dsFastTwoSum(s.hi, (e + z.lo) + s.lo);
}
// emitDSDiv
F2 dsDiv(F2 x, F2 y) {
  float zhi = x.hi / y.hi;
  F2 pe = expansionTwoProd(zhi, y.hi);
  float q = zhi * y.lo;
  float zlo = ((((x.hi - pe.hi) - pe.lo) + x.lo) - q) / y.hi;
  float s = zhi + zlo;
  return {s, (zhi - s) + zlo};
}
// emitDSSqrt
F2 dsSqrt(F2 x) {
  if (x.hi == 0.0f && x.lo == 0.0f)
    return x;
  float y = std::sqrt(x.hi);
  float r = std::fma(-y, y, x.hi) + x.lo;
  return dsFastTwoSum(y, r / (y + y));
}

// Three and four components.
// mX_real Normalize<Regular>, non-Quasi.
void expansionNorm3(float &x0, float &x1, float &x2) {
  F2 t = expansionFastTwoSum(x1, x2);
  x1 = t.hi;
  x2 = t.lo;
  t = expansionFastTwoSum(x0, x1);
  x0 = t.hi;
  x1 = t.lo;
  t = expansionFastTwoSum(x1, x2);
  x1 = t.hi;
  x2 = t.lo;
}
void expansionNorm4(float &x0, float &x1, float &x2, float &x3) {
  F2 t = expansionFastTwoSum(x2, x3);
  x2 = t.hi;
  x3 = t.lo;
  t = expansionFastTwoSum(x1, x2);
  x1 = t.hi;
  x2 = t.lo;
  t = expansionFastTwoSum(x0, x1);
  x0 = t.hi;
  x1 = t.lo;
  t = expansionFastTwoSum(x2, x3);
  x2 = t.hi;
  x3 = t.lo;
  t = expansionFastTwoSum(x1, x2);
  x1 = t.hi;
  x2 = t.lo;
  t = expansionFastTwoSum(x2, x3);
  x2 = t.hi;
  x3 = t.lo;
}

// QxW::add_QTW_QTW_QTW
void expansionAddRaw3(float a0, float a1, float a2, float b0, float b1,
                      float b2, float &c0, float &c1, float &c2) {
  F2 s = expansionTwoSum(a0, b0);
  c0 = s.hi;
  c1 = s.lo;
  F2 u = expansionTwoSum(a1, b1);
  float t0 = u.hi;
  c2 = u.lo;
  F2 v = expansionTwoSum(c1, t0);
  c1 = v.hi;
  t0 = v.lo;
  float t1 = a2 + b2;
  c2 = (c2 + t0) + t1;
}

// QxW::mul_QTW_QTW_QTW
void expansionMulRaw3(float a0, float a1, float a2, float b0, float b1,
                      float b2, float &c0, float &c1, float &c2) {
  F2 p = expansionTwoProd(a0, b0);
  c0 = p.hi;
  c1 = p.lo;
  F2 q = expansionTwoProd(a0, b1);
  c2 = q.hi;
  float t0 = q.lo;
  F2 r = expansionTwoProd(a1, b0);
  float t1 = r.hi, t2 = r.lo;
  F2 s = expansionTwoSum(c1, c2);
  c1 = s.hi;
  c2 = s.lo;
  F2 u = expansionTwoSum(c1, t1);
  c1 = u.hi;
  t1 = u.lo;
  t0 = std::fma(a0, b2, t0);
  c2 = std::fma(a1, b1, c2);
  t2 = std::fma(a2, b0, t2);
  c2 = ((t0 + c2) + t2) + t1;
}

// QxW::mul_PA_QTW_QTW
void expansionMulRaw23(float a0, float a1, float b0, float b1, float b2,
                       float &c0, float &c1, float &c2) {
  F2 p = expansionTwoProd(a0, b0);
  c0 = p.hi;
  c1 = p.lo;
  F2 q = expansionTwoProd(a0, b1);
  c2 = q.hi;
  float t0 = q.lo;
  F2 r = expansionTwoProd(a1, b0);
  float t1 = r.hi, t2 = r.lo;
  F2 s = expansionTwoSum(c1, c2);
  c1 = s.hi;
  c2 = s.lo;
  F2 u = expansionTwoSum(c1, t1);
  c1 = u.hi;
  t1 = u.lo;
  t0 = std::fma(a0, b2, t0);
  c2 = std::fma(a1, b1, c2);
  c2 = ((t0 + c2) + t2) + t1;
}

// QxW::div_PA_PA_PA
void expansionDivRaw22(float a0, float a1, float b0, float b1, float &c0,
                       float &c1) {
  float bh = b0 + b1;
  c0 = a0 / bh;
  c1 = std::fma(-b0, c0, a0) + a1;
  c1 = std::fma(-b1, c0, c1) / bh;
}

// QxW::div_QTW_QTW_QTW
void expansionDivRaw3(float a0, float a1, float a2, float b0, float b1,
                      float b2, float &c0, float &c1, float &c2) {
  float e40 = a1 + a2;
  float e41 = b1 + b2;
  expansionDivRaw22(a0, e40, b0, e41, c0, c1);
  float t0, t1, t2;
  expansionMulRaw23(c0, c1, b0, b1, b2, t0, t1, t2);
  float s0, s1, s2;
  expansionAddRaw3(a0, a1, a2, -t0, -t1, -t2, s0, s1, s2);
  float tn = (s0 + s1) + s2;
  float td = (b0 + b1) + b2;
  c2 = tn / td;
}

// QxW::sqrt_QTW_PA
void expansionSqrtRaw32(float a0, float a1, float a2, float &c0, float &c1) {
  c0 = std::sqrt(a0);
  float num = (std::fma(-c0, c0, a0) + a1) + a2;
  c1 = num / (c0 + c0);
}

// QxW::sqr_PA_QTW
void expansionSqrRaw23(float a0, float a1, float &c0, float &c1, float &c2) {
  F2 p = expansionTwoProd(a0, a0);
  c0 = p.hi;
  c1 = p.lo;
  float t0 = a0 + a0;
  F2 q = expansionTwoProd(t0, a1);
  c2 = q.hi;
  float t1 = q.lo;
  F2 s = expansionTwoSum(c1, c2);
  c1 = s.hi;
  c2 = s.lo;
  c2 = c2 + t1;
  c2 = std::fma(a1, a1, c2);
}

// QxW::sqr_QTW_QTW
void expansionSqrRaw3(float a0, float a1, float a2, float &c0, float &c1,
                      float &c2) {
  F2 p = expansionTwoProd(a0, a0);
  c0 = p.hi;
  c1 = p.lo;
  float t0 = a0 + a0;
  F2 q = expansionTwoProd(t0, a1);
  c2 = q.hi;
  float t1 = q.lo;
  F2 s = expansionTwoSum(c1, c2);
  c1 = s.hi;
  c2 = s.lo;
  c2 = c2 + t1;
  c2 = std::fma(t0, a2, c2);
  c2 = std::fma(a1, a1, c2);
}

// QxW::sqrt_QTW_QTW
void expansionSqrtRaw3(float a0, float a1, float a2, float &c0, float &c1,
                       float &c2) {
  expansionSqrtRaw32(a0, a1, a2, c0, c1);
  float t0, t1, t2;
  expansionSqrRaw23(c0, c1, t0, t1, t2);
  float s0, s1, s2;
  expansionAddRaw3(a0, a1, a2, -t0, -t1, -t2, s0, s1, s2);
  float tn = (s0 + s1) + s2;
  float td = (c0 + c1) * 2.0f;
  c2 = tn / td;
}

// QxW::add_QQW_QQW_QQW
void expansionAddRaw4(float a0, float a1, float a2, float a3, float b0,
                      float b1, float b2, float b3, float &c0, float &c1,
                      float &c2, float &c3) {
  F2 s = expansionTwoSum(a0, b0);
  c0 = s.hi;
  c1 = s.lo;
  F2 u = expansionTwoSum(a1, b1);
  float t0 = u.hi;
  c2 = u.lo;
  F2 v = expansionTwoSum(a2, b2);
  float t1 = v.hi;
  c3 = v.lo;
  F2 w = expansionTwoSum(c1, t0);
  c1 = w.hi;
  t0 = w.lo;
  F2 x = expansionTwoSum(c2, t0);
  c2 = x.hi;
  t0 = x.lo;
  F2 y = expansionTwoSum(c2, t1);
  c2 = y.hi;
  t1 = y.lo;
  float t2 = a3 + b3;
  c3 = ((c3 + t0) + t1) + t2;
}

// QxW::mul_QQW_QQW_QQW
void expansionMulRaw4(float a0, float a1, float a2, float a3, float b0,
                      float b1, float b2, float b3, float &c0, float &c1,
                      float &c2, float &c3) {
  F2 p = expansionTwoProd(a0, b0);
  c0 = p.hi;
  c1 = p.lo;
  F2 q = expansionTwoProd(a0, b1);
  c2 = q.hi;
  c3 = q.lo;
  F2 r = expansionTwoProd(a1, b0);
  float t0 = r.hi, t1 = r.lo;
  F2 s = expansionTwoSum(c1, c2);
  c1 = s.hi;
  c2 = s.lo;
  F2 u = expansionTwoSum(c1, t0);
  c1 = u.hi;
  t0 = u.lo;
  F2 v = expansionTwoProd(a0, b2);
  float t2 = v.hi, t3 = v.lo;
  F2 w = expansionTwoProd(a1, b1);
  float t4 = w.hi, t5 = w.lo;
  F2 x = expansionTwoProd(a2, b0);
  float t6 = x.hi, t7 = x.lo;
  F2 y = expansionTwoSum(c2, t0);
  c2 = y.hi;
  t0 = y.lo;
  F2 z = expansionTwoSum(c2, c3);
  c2 = z.hi;
  c3 = z.lo;
  F2 aa = expansionTwoSum(c2, t1);
  c2 = aa.hi;
  t1 = aa.lo;
  F2 bb = expansionTwoSum(c2, t2);
  c2 = bb.hi;
  t2 = bb.lo;
  F2 cc = expansionTwoSum(c2, t4);
  c2 = cc.hi;
  t4 = cc.lo;
  F2 dd = expansionTwoSum(c2, t6);
  c2 = dd.hi;
  t6 = dd.lo;
  c3 = std::fma(a0, b3, c3);
  c3 = std::fma(a1, b2, c3);
  c3 = std::fma(a2, b1, c3);
  c3 = std::fma(a3, b0, c3);
  c3 = c3 + t0;
  c3 = c3 + t1;
  c3 = c3 + t2;
  c3 = c3 + t3;
  c3 = c3 + t4;
  c3 = c3 + t5;
  c3 = c3 + t6;
  c3 = c3 + t7;
}

// QxW::mul_QTW_QQW_QQW
void expansionMulRaw34(float a0, float a1, float a2, float b0, float b1,
                       float b2, float b3, float &c0, float &c1, float &c2,
                       float &c3) {
  F2 p = expansionTwoProd(a0, b0);
  c0 = p.hi;
  c1 = p.lo;
  F2 q = expansionTwoProd(a0, b1);
  c2 = q.hi;
  c3 = q.lo;
  F2 r = expansionTwoProd(a1, b0);
  float t0 = r.hi, t1 = r.lo;
  F2 s = expansionTwoSum(c1, c2);
  c1 = s.hi;
  c2 = s.lo;
  F2 u = expansionTwoSum(c1, t0);
  c1 = u.hi;
  t0 = u.lo;
  F2 v = expansionTwoProd(a0, b2);
  float t2 = v.hi, t3 = v.lo;
  F2 w = expansionTwoProd(a1, b1);
  float t4 = w.hi, t5 = w.lo;
  F2 x = expansionTwoProd(a2, b0);
  float t6 = x.hi, t7 = x.lo;
  F2 y = expansionTwoSum(c2, t0);
  c2 = y.hi;
  t0 = y.lo;
  F2 z = expansionTwoSum(c2, c3);
  c2 = z.hi;
  c3 = z.lo;
  F2 aa = expansionTwoSum(c2, t1);
  c2 = aa.hi;
  t1 = aa.lo;
  F2 bb = expansionTwoSum(c2, t2);
  c2 = bb.hi;
  t2 = bb.lo;
  F2 cc = expansionTwoSum(c2, t4);
  c2 = cc.hi;
  t4 = cc.lo;
  F2 dd = expansionTwoSum(c2, t6);
  c2 = dd.hi;
  t6 = dd.lo;
  c3 = std::fma(a0, b3, c3);
  c3 = std::fma(a1, b2, c3);
  c3 = std::fma(a2, b1, c3);
  c3 = c3 + t0;
  c3 = c3 + t1;
  c3 = c3 + t2;
  c3 = c3 + t3;
  c3 = c3 + t4;
  c3 = c3 + t5;
  c3 = c3 + t6;
  c3 = c3 + t7;
}

// QxW::sqr_QTW_QQW
void expansionSqrRaw34(float a0, float a1, float a2, float &c0, float &c1,
                       float &c2, float &c3) {
  F2 p = expansionTwoProd(a0, a0);
  c0 = p.hi;
  c1 = p.lo;
  float t0 = a0 + a0;
  F2 q = expansionTwoProd(t0, a1);
  c2 = q.hi;
  c3 = q.lo;
  F2 s = expansionTwoSum(c1, c2);
  c1 = s.hi;
  c2 = s.lo;
  F2 u = expansionTwoSum(c2, c3);
  c2 = u.hi;
  c3 = u.lo;
  F2 v = expansionTwoProd(t0, a2);
  float t1 = v.hi, t2 = v.lo;
  F2 w = expansionTwoProd(a1, a1);
  float t3 = w.hi, t4 = w.lo;
  F2 x = expansionTwoSum(c2, t1);
  c2 = x.hi;
  t1 = x.lo;
  F2 y = expansionTwoSum(c2, t3);
  c2 = y.hi;
  t3 = y.lo;
  float t5 = a1 + a1;
  c3 = (((c3 + t1) + t2) + t3) + t4;
  c3 = std::fma(t5, a2, c3);
}

// QxW::sqr_QQW_QQW
void expansionSqrRaw4(float a0, float a1, float a2, float a3, float &c0,
                      float &c1, float &c2, float &c3) {
  F2 p = expansionTwoProd(a0, a0);
  c0 = p.hi;
  c1 = p.lo;
  float t0 = a0 + a0;
  F2 q = expansionTwoProd(t0, a1);
  c2 = q.hi;
  c3 = q.lo;
  F2 s = expansionTwoSum(c1, c2);
  c1 = s.hi;
  c2 = s.lo;
  F2 u = expansionTwoSum(c2, c3);
  c2 = u.hi;
  c3 = u.lo;
  F2 v = expansionTwoProd(t0, a2);
  float t1 = v.hi, t2 = v.lo;
  F2 w = expansionTwoProd(a1, a1);
  float t3 = w.hi, t4 = w.lo;
  F2 x = expansionTwoSum(c2, t1);
  c2 = x.hi;
  t1 = x.lo;
  F2 y = expansionTwoSum(c2, t3);
  c2 = y.hi;
  t3 = y.lo;
  float t5 = a1 + a1;
  c3 = (((c3 + t1) + t2) + t3) + t4;
  c3 = std::fma(t0, a3, c3);
  c3 = std::fma(t5, a2, c3);
}

// QxW::div_QQW_QQW_QQW
void expansionDivRaw4(float a0, float a1, float a2, float a3, float b0,
                      float b1, float b2, float b3, float &c0, float &c1,
                      float &c2, float &c3) {
  float e40 = a2 + a3;
  float e41 = b2 + b3;
  expansionDivRaw3(a0, a1, e40, b0, b1, e41, c0, c1, c2);
  float t0, t1, t2, t3;
  expansionMulRaw34(c0, c1, c2, b0, b1, b2, b3, t0, t1, t2, t3);
  float s0, s1, s2, s3;
  expansionAddRaw4(a0, a1, a2, a3, -t0, -t1, -t2, -t3, s0, s1, s2, s3);
  float tn = ((s0 + s1) + s2) + s3;
  float td = ((b0 + b1) + b2) + b3;
  c3 = tn / td;
}

// QxW::sqrt_QQW_QQW (its sqrt_QQW_QTW is sqrt_QTW_QTW(a0, a1, a2 + a3, ...))
void expansionSqrtRaw4(float a0, float a1, float a2, float a3, float &c0,
                       float &c1, float &c2, float &c3) {
  expansionSqrtRaw3(a0, a1, a2 + a3, c0, c1, c2);
  float t0, t1, t2, t3;
  expansionSqrRaw34(c0, c1, c2, t0, t1, t2, t3);
  float s0, s1, s2, s3;
  expansionAddRaw4(a0, a1, a2, a3, -t0, -t1, -t2, -t3, s0, s1, s2, s3);
  float tn = ((s0 + s1) + s2) + s3;
  float td = ((c0 + c1) + c2) * 2.0f;
  c3 = tn / td;
}

F2 pair(const ExpansionLimbs &a) { return {a.x[0], a.x[1]}; }
ExpansionLimbs limbs2(F2 r) {
  ExpansionLimbs c;
  c.n = 2;
  c.x[0] = r.hi;
  c.x[1] = r.lo;
  return c;
}
ExpansionLimbs normalized(ExpansionLimbs c) {
  if (c.n == 3)
    expansionNorm3(c.x[0], c.x[1], c.x[2]);
  else
    expansionNorm4(c.x[0], c.x[1], c.x[2], c.x[3]);
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

// emitF64ToDS, and emitToExpansion's greedy split.
ExpansionLimbs expansionFromF64(double v, unsigned n) {
  ExpansionLimbs r;
  r.n = n;
  double rem = v;
  for (unsigned i = 0; i < n; ++i) {
    r.x[i] = (float)rem;
    rem -= (double)r.x[i];
  }
  return r;
}

ExpansionLimbs expansionFromF32(float v, unsigned n) {
  ExpansionLimbs r;
  r.n = n;
  r.x[0] = v;
  return r;
}

// Least significant component first, as emitDSToFP and emitExpansionToFP sum.
double expansionToF64(const ExpansionLimbs &a) {
  double s = (double)a.x[a.n - 1];
  for (int i = (int)a.n - 2; i >= 0; --i)
    s = (double)a.x[i] + s;
  return s;
}

float expansionToF32(const ExpansionLimbs &a) {
  float s = a.x[a.n - 1];
  for (int i = (int)a.n - 2; i >= 0; --i)
    s = a.x[i] + s;
  return s;
}

ExpansionLimbs expansionNeg(const ExpansionLimbs &a) {
  ExpansionLimbs c = a;
  for (unsigned i = 0; i < a.n; ++i)
    c.x[i] = -a.x[i];
  return c;
}

ExpansionLimbs expansionAdd(const ExpansionLimbs &a, const ExpansionLimbs &b) {
  if (a.n == 2)
    return limbs2(dsAdd(pair(a), pair(b)));
  ExpansionLimbs c;
  c.n = a.n;
  if (a.n == 3)
    expansionAddRaw3(a.x[0], a.x[1], a.x[2], b.x[0], b.x[1], b.x[2], c.x[0],
                     c.x[1], c.x[2]);
  else
    expansionAddRaw4(a.x[0], a.x[1], a.x[2], a.x[3], b.x[0], b.x[1], b.x[2],
                     b.x[3], c.x[0], c.x[1], c.x[2], c.x[3]);
  return normalized(c);
}

ExpansionLimbs expansionSub(const ExpansionLimbs &a, const ExpansionLimbs &b) {
  if (a.n == 2)
    return limbs2(dsSub(pair(a), pair(b)));
  return expansionAdd(a, expansionNeg(b));
}

ExpansionLimbs expansionMul(const ExpansionLimbs &a, const ExpansionLimbs &b) {
  if (a.n == 2)
    return limbs2(dsMul(pair(a), pair(b)));
  ExpansionLimbs c;
  c.n = a.n;
  if (a.n == 3)
    expansionMulRaw3(a.x[0], a.x[1], a.x[2], b.x[0], b.x[1], b.x[2], c.x[0],
                     c.x[1], c.x[2]);
  else
    expansionMulRaw4(a.x[0], a.x[1], a.x[2], a.x[3], b.x[0], b.x[1], b.x[2],
                     b.x[3], c.x[0], c.x[1], c.x[2], c.x[3]);
  return normalized(c);
}

ExpansionLimbs expansionSqr(const ExpansionLimbs &a) {
  if (a.n == 2)
    return limbs2(dsMul(pair(a), pair(a)));
  ExpansionLimbs c;
  c.n = a.n;
  if (a.n == 3)
    expansionSqrRaw3(a.x[0], a.x[1], a.x[2], c.x[0], c.x[1], c.x[2]);
  else
    expansionSqrRaw4(a.x[0], a.x[1], a.x[2], a.x[3], c.x[0], c.x[1], c.x[2],
                     c.x[3]);
  return normalized(c);
}

ExpansionLimbs expansionDiv(const ExpansionLimbs &a, const ExpansionLimbs &b) {
  if (a.n == 2)
    return limbs2(dsDiv(pair(a), pair(b)));
  ExpansionLimbs c;
  c.n = a.n;
  if (a.n == 3)
    expansionDivRaw3(a.x[0], a.x[1], a.x[2], b.x[0], b.x[1], b.x[2], c.x[0],
                     c.x[1], c.x[2]);
  else
    expansionDivRaw4(a.x[0], a.x[1], a.x[2], a.x[3], b.x[0], b.x[1], b.x[2],
                     b.x[3], c.x[0], c.x[1], c.x[2], c.x[3]);
  return normalized(c);
}

ExpansionLimbs expansionSqrt(const ExpansionLimbs &a) {
  if (a.n == 2)
    return limbs2(dsSqrt(pair(a)));
  // emitExpansionSqrt's special cases (mX_real's sqrt operator).
  if (a.x[0] == 0.0f)
    return a;
  ExpansionLimbs c;
  c.n = a.n;
  if (a.x[0] == std::numeric_limits<float>::infinity()) {
    for (unsigned i = 0; i < a.n; ++i)
      c.x[i] = a.x[0];
    return c;
  }
  if (a.n == 3)
    expansionSqrtRaw3(a.x[0], a.x[1], a.x[2], c.x[0], c.x[1], c.x[2]);
  else
    expansionSqrtRaw4(a.x[0], a.x[1], a.x[2], a.x[3], c.x[0], c.x[1], c.x[2],
                      c.x[3]);
  return normalized(c);
}

ExpansionLimbs expansionFma(const ExpansionLimbs &a, const ExpansionLimbs &b,
                            const ExpansionLimbs &c) {
  if (a.n == 2)
    return limbs2(dsFma(pair(a), pair(b), pair(c)));
  return expansionAdd(expansionMul(a, b), c);
}

ExpansionLimbs expansionFmaSqr(const ExpansionLimbs &a,
                               const ExpansionLimbs &c) {
  if (a.n == 2)
    return limbs2(dsFma(pair(a), pair(a), pair(c)));
  return expansionAdd(expansionSqr(a), c);
}

} // namespace poseidon
