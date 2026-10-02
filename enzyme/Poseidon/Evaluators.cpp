//=- Evaluators.cpp - Expression evaluators for Poseidon ------------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the evaluator classes for floating-point expressions
// in the Poseidon optimization pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/APFloat.h"
#include "llvm/Support/raw_ostream.h"

#include <cstring>

#include <algorithm>
#include <cmath>
#include <vector>

#include "Evaluators.h"
#include "Flags.h"
#include "Precision.h"
#include "Sampling.h"
#include "Types.h"

using namespace llvm;

namespace poseidon {

namespace {
struct DS {
  float hi, lo;
  double toF64() const { return (double)hi + (double)lo; }
  static DS fromF64(double x) {
    float h = (float)x;
    return {h, (float)(x - (double)h)};
  }
};
DS dsTwoSum(float a, float b) {
  float s = a + b, ap = s - b, bp = s - ap;
  return {s, (a - ap) + (b - bp)};
}
DS dsFastTwoSum(float a, float b) {
  float s = a + b;
  return {s, b - (s - a)};
}
DS dsTwoProd(float a, float b) {
  float p = a * b;
  const float c = 4097.0f;
  float ta = c * a, hi_a = ta - (ta - a), lo_a = a - hi_a;
  float tb = c * b, hi_b = tb - (tb - b), lo_b = b - hi_b;
  float e = ((hi_a * hi_b - p) + hi_a * lo_b + lo_a * hi_b) + lo_a * lo_b;
  return {p, e};
}
DS dsNeg(DS x) { return {-x.hi, -x.lo}; }
DS dsAdd(DS x, DS y) {
  DS ab = dsTwoSum(x.hi, y.hi), cd = dsTwoSum(x.lo, y.lo);
  DS ac = dsFastTwoSum(ab.hi, cd.hi);
  return dsFastTwoSum(ac.hi, (ab.lo + cd.lo) + ac.lo);
}
DS dsSub(DS x, DS y) { return dsAdd(x, dsNeg(y)); }
DS dsMul(DS x, DS y) {
  DS pe = dsTwoProd(x.hi, y.hi);
  return dsFastTwoSum(pe.hi, pe.lo + (x.hi * y.lo + x.lo * y.hi));
}
DS dsDiv(DS x, DS y) {
  float zhi = x.hi / y.hi;
  DS pe = dsTwoProd(zhi, y.hi);
  float d = ((x.hi - pe.hi) - pe.lo + x.lo) - zhi * y.lo;
  return dsFastTwoSum(zhi, d / y.hi);
}
DS dsSqrt(DS x) {
  float zhi = sqrtf(x.hi);
  DS pe = dsTwoProd(zhi, zhi);
  float d = ((x.hi - pe.hi) - pe.lo) + x.lo;
  return dsFastTwoSum(zhi, d / (2.0f * zhi));
}
DS dsFma(DS a, DS b, DS c) { return dsAdd(dsMul(a, b), c); }

// Wider FP32 expansions (n = 3, 4) for the accuracy model. Each routine is the
// operation-for-operation twin of the corresponding emitter in
// Expansion.cpp (same QxW routine, same association of the trailing
// sums, std::fma exactly where the emitter emits llvm.fma.f32), so the model
// simulates the code the materializer emits.
struct F2 {
  float hi, lo;
};
F2 expansionTwoSum(float a, float b) {
  float s = a + b, ap = s - b, bp = s - ap;
  return {s, (a - ap) + (b - bp)};
}
F2 expansionFastTwoSum(float a, float b) {
  float s = a + b;
  return {s, b - (s - a)};
}
F2 expansionTwoProd(float a, float b) {
  float p = a * b;
  return {p, std::fma(a, b, -p)};
}

struct ExpansionN {
  float x[4];
  int n;
  double toF64() const {
    double s = (double)x[n - 1];
    for (int i = n - 2; i >= 0; --i)
      s = (double)x[i] + s;
    return s;
  }
  static ExpansionN fromF64(double v, int n) {
    ExpansionN r;
    r.n = n;
    double rem = v;
    for (int i = 0; i < n; ++i) {
      float li = (float)rem;
      r.x[i] = li;
      rem -= (double)li;
    }
    for (int i = n; i < 4; ++i)
      r.x[i] = 0.0f;
    return r;
  }
};

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

ExpansionN expansionNNeg(ExpansionN a) {
  for (int i = 0; i < a.n; ++i)
    a.x[i] = -a.x[i];
  return a;
}
ExpansionN expansionNAdd(ExpansionN a, ExpansionN b) {
  ExpansionN c;
  c.n = a.n;
  c.x[3] = 0.0f;
  if (a.n == 3) {
    expansionAddRaw3(a.x[0], a.x[1], a.x[2], b.x[0], b.x[1], b.x[2], c.x[0],
                     c.x[1], c.x[2]);
    expansionNorm3(c.x[0], c.x[1], c.x[2]);
  } else {
    expansionAddRaw4(a.x[0], a.x[1], a.x[2], a.x[3], b.x[0], b.x[1], b.x[2],
                     b.x[3], c.x[0], c.x[1], c.x[2], c.x[3]);
    expansionNorm4(c.x[0], c.x[1], c.x[2], c.x[3]);
  }
  return c;
}
ExpansionN expansionNSub(ExpansionN a, ExpansionN b) {
  return expansionNAdd(a, expansionNNeg(b));
}
ExpansionN expansionNMul(ExpansionN a, ExpansionN b) {
  ExpansionN c;
  c.n = a.n;
  c.x[3] = 0.0f;
  if (a.n == 3) {
    expansionMulRaw3(a.x[0], a.x[1], a.x[2], b.x[0], b.x[1], b.x[2], c.x[0],
                     c.x[1], c.x[2]);
    expansionNorm3(c.x[0], c.x[1], c.x[2]);
  } else {
    expansionMulRaw4(a.x[0], a.x[1], a.x[2], a.x[3], b.x[0], b.x[1], b.x[2],
                     b.x[3], c.x[0], c.x[1], c.x[2], c.x[3]);
    expansionNorm4(c.x[0], c.x[1], c.x[2], c.x[3]);
  }
  return c;
}
ExpansionN expansionNDiv(ExpansionN a, ExpansionN b) {
  ExpansionN c;
  c.n = a.n;
  c.x[3] = 0.0f;
  if (a.n == 3) {
    expansionDivRaw3(a.x[0], a.x[1], a.x[2], b.x[0], b.x[1], b.x[2], c.x[0],
                     c.x[1], c.x[2]);
    expansionNorm3(c.x[0], c.x[1], c.x[2]);
  } else {
    expansionDivRaw4(a.x[0], a.x[1], a.x[2], a.x[3], b.x[0], b.x[1], b.x[2],
                     b.x[3], c.x[0], c.x[1], c.x[2], c.x[3]);
    expansionNorm4(c.x[0], c.x[1], c.x[2], c.x[3]);
  }
  return c;
}
ExpansionN expansionNSqrt(ExpansionN a) {
  ExpansionN c;
  c.n = a.n;
  c.x[3] = 0.0f;
  if (a.n == 3) {
    expansionSqrtRaw3(a.x[0], a.x[1], a.x[2], c.x[0], c.x[1], c.x[2]);
    expansionNorm3(c.x[0], c.x[1], c.x[2]);
  } else {
    expansionSqrtRaw4(a.x[0], a.x[1], a.x[2], a.x[3], c.x[0], c.x[1], c.x[2],
                      c.x[3]);
    expansionNorm4(c.x[0], c.x[1], c.x[2], c.x[3]);
  }
  return c;
}
} // namespace

FPEvaluator::FPEvaluator(PTCandidate *pt) {
  exactExpansion = flags::AccuracyReferenceBits > 0;
  if (pt) {
    for (const auto &change : pt->changes) {
      for (auto node : change.nodes) {
        nodePrecisions[node] = change.newType;
      }
    }
  }
}

PrecisionChangeType FPEvaluator::getNodePrecision(const FPNode *node) const {
  auto it = nodePrecisions.find(node);
  if (it != nodePrecisions.end())
    return it->second;

  if (node->dtype == "f16")
    return PrecisionChangeType::FP16;
  if (node->dtype == "bf16")
    return PrecisionChangeType::BF16;
  if (node->dtype == "f32")
    return PrecisionChangeType::FP32;
  if (node->dtype == "f64")
    return PrecisionChangeType::FP64;
  // Herbie-dialect spellings, accepted so that a node reaching here straight
  // from Herbie text (rather than through parseHerbieExpr, which normalizes
  // them) degrades to the right precision instead of aborting the compile.
  if (node->dtype == "binary32")
    return PrecisionChangeType::FP32;
  if (node->dtype == "binary64")
    return PrecisionChangeType::FP64;

  llvm_unreachable(
      ("Operator " + node->op + " has unexpected dtype: " + node->dtype)
          .c_str());
}

void FPEvaluator::evaluateNode(const FPNode *node,
                               const MapVector<Value *, double> &inputValues) {
  if (cache.find(node) != cache.end())
    return;

  if (isa<FPConst>(node)) {
    double constVal = node->getLowerBound();
    cache.emplace(node, constVal);
    return;
  }

  if (isa<FPLLValue>(node) && inputValues.count(cast<FPLLValue>(node)->value)) {
    double inputValue = inputValues.lookup(cast<FPLLValue>(node)->value);
    cache.emplace(node, inputValue);
    return;
  }

  if (node->op == "if") {
    evaluateNode(node->operands[0].get(), inputValues);
    double cond = getResult(node->operands[0].get());

    if (cond == 1.0) {
      evaluateNode(node->operands[1].get(), inputValues);
      double then_val = getResult(node->operands[1].get());
      cache.emplace(node, then_val);
    } else {
      evaluateNode(node->operands[2].get(), inputValues);
      double else_val = getResult(node->operands[2].get());
      cache.emplace(node, else_val);
    }
    return;
  } else if (node->op == "and") {
    evaluateNode(node->operands[0].get(), inputValues);
    double op0 = getResult(node->operands[0].get());
    if (op0 != 1.0) {
      cache.emplace(node, 0.0);
      return;
    }
    evaluateNode(node->operands[1].get(), inputValues);
    double op1 = getResult(node->operands[1].get());
    if (op1 != 1.0) {
      cache.emplace(node, 0.0);
      return;
    }
    cache.emplace(node, 1.0);
    return;
  } else if (node->op == "or") {
    evaluateNode(node->operands[0].get(), inputValues);
    double op0 = getResult(node->operands[0].get());
    if (op0 == 1.0) {
      cache.emplace(node, 1.0);
      return;
    }
    evaluateNode(node->operands[1].get(), inputValues);
    double op1 = getResult(node->operands[1].get());
    if (op1 == 1.0) {
      cache.emplace(node, 1.0);
      return;
    }
    cache.emplace(node, 0.0);
    return;
  } else if (node->op == "not") {
    evaluateNode(node->operands[0].get(), inputValues);
    double op = getResult(node->operands[0].get());
    cache.emplace(node, (op == 1.0) ? 0.0 : 1.0);
    return;
  } else if (node->op == "TRUE") {
    cache.emplace(node, 1.0);
    return;
  } else if (node->op == "FALSE") {
    cache.emplace(node, 0.0);
    return;
  }

  PrecisionChangeType nodePrec = getNodePrecision(node);

  for (const auto &operand : node->operands) {
    evaluateNode(operand.get(), inputValues);
  }

  if (nodePrec == PrecisionChangeType::FP80 ||
      nodePrec == PrecisionChangeType::FP128)
    llvm_unreachable("FPEvaluator: FP80/FP128 evaluation not implemented");

  double res = 0.0;

  bool useReducedFloat = (nodePrec == PrecisionChangeType::FP32 ||
                          nodePrec == PrecisionChangeType::FP16 ||
                          nodePrec == PrecisionChangeType::BF16);
  const unsigned nExp = expansionComponents(nodePrec);
  bool useExpansion2 = (nExp == 2);
  bool useExpansion = (nExp >= 3);

  auto truncToPrec = [&](float val) -> float {
    FPKind k = FPKind::Invalid;
    if (nodePrec == PrecisionChangeType::BF16)
      k = FPKind::BF16;
    else if (nodePrec == PrecisionChangeType::FP16)
      k = FPKind::F16;
    else
      return val;
    return static_cast<float>(roundToPrec((double)val, k));
  };

  auto toF = [&](double v) -> float {
    return truncToPrec(static_cast<float>(v));
  };

  using DSUnaryFn = DS (*)(DS);
  using DSBinaryFn = DS (*)(DS, DS);
  using DSTernaryFn = DS (*)(DS, DS, DS);
  using ExpansionNUnaryFn = ExpansionN (*)(ExpansionN);
  using ExpansionNBinaryFn = ExpansionN (*)(ExpansionN, ExpansionN);

  // Prefer the operand's recorded limbs over re-splitting its double collapse,
  // which would pin the simulated chain to 53 bits; a wider operand is
  // truncated (correct for a normalized expansion), a narrower one zero-padded.
  auto operandExpansion = [&](unsigned i) -> ExpansionN {
    const FPNode *o = node->operands[i].get();
    auto it = exactExpansion ? expCache.find(o) : expCache.end();
    if (it != expCache.end() && !it->second.empty()) {
      ExpansionN m;
      m.n = (int)nExp;
      for (int k = 0; k < 4; ++k)
        m.x[k] = (k < (int)nExp && k < (int)it->second.size()) ? it->second[k]
                                                               : 0.0f;
      return m;
    }
    return ExpansionN::fromF64(getResult(o), (int)nExp);
  };
  auto recordExpansion = [&](const ExpansionN &m) -> double {
    if (exactExpansion) {
      SmallVector<float, 4> limbs;
      for (int k = 0; k < m.n; ++k)
        limbs.push_back(m.x[k]);
      expCache[node] = std::move(limbs);
    }
    return m.toF64();
  };

  auto evalUnary = [&](auto f64Fn, auto f32Fn, DSUnaryFn dsFn = nullptr,
                       ExpansionNUnaryFn expansionFn = nullptr) -> double {
    double op = getResult(node->operands[0].get());
    if (useReducedFloat)
      return (double)truncToPrec(f32Fn(toF(op)));
    if (useExpansion2 && dsFn)
      return dsFn(DS::fromF64(op)).toF64();
    if (useExpansion && expansionFn)
      return recordExpansion(expansionFn(operandExpansion(0)));
    return f64Fn(op);
  };
  auto evalBinary = [&](auto f64Fn, auto f32Fn, DSBinaryFn dsFn = nullptr,
                        ExpansionNBinaryFn expansionFn = nullptr) -> double {
    double a = getResult(node->operands[0].get());
    double b = getResult(node->operands[1].get());
    if (useReducedFloat)
      return (double)truncToPrec(f32Fn(toF(a), toF(b)));
    if (useExpansion2 && dsFn)
      return dsFn(DS::fromF64(a), DS::fromF64(b)).toF64();
    if (useExpansion && expansionFn)
      return recordExpansion(
          expansionFn(operandExpansion(0), operandExpansion(1)));
    return f64Fn(a, b);
  };
  auto evalTernary = [&](auto f64Fn, auto f32Fn, DSTernaryFn dsFn = nullptr,
                         ExpansionNBinaryFn expansionMulFn = nullptr,
                         ExpansionNBinaryFn expansionAddFn =
                             nullptr) -> double {
    double a = getResult(node->operands[0].get());
    double b = getResult(node->operands[1].get());
    double c = getResult(node->operands[2].get());
    if (useReducedFloat)
      return (double)truncToPrec(f32Fn(toF(a), toF(b), toF(c)));
    if (useExpansion2 && dsFn)
      return dsFn(DS::fromF64(a), DS::fromF64(b), DS::fromF64(c)).toF64();
    // The n >= 3 materializer has no fused composite: it emits
    // multiply-then-add, so the model must too.
    if (useExpansion && expansionMulFn && expansionAddFn)
      return recordExpansion(expansionAddFn(
          expansionMulFn(operandExpansion(0), operandExpansion(1)),
          operandExpansion(2)));
    return f64Fn(a, b, c);
  };
  auto evalCompare = [&](auto cmp) -> double {
    double a = getResult(node->operands[0].get());
    double b = getResult(node->operands[1].get());
    bool r = useReducedFloat ? cmp(toF(a), toF(b)) : cmp(a, b);
    return r ? 1.0 : 0.0;
  };

  if (node->op == "neg") {
    res = evalUnary([](double x) { return -x; }, [](float x) { return -x; },
                    dsNeg, expansionNNeg);
  } else if (node->op == "+") {
    res = evalBinary([](double a, double b) { return a + b; },
                     [](float a, float b) { return a + b; }, dsAdd,
                     expansionNAdd);
  } else if (node->op == "-") {
    res = evalBinary([](double a, double b) { return a - b; },
                     [](float a, float b) { return a - b; }, dsSub,
                     expansionNSub);
  } else if (node->op == "*") {
    res = evalBinary([](double a, double b) { return a * b; },
                     [](float a, float b) { return a * b; }, dsMul,
                     expansionNMul);
  } else if (node->op == "/") {
    res = evalBinary([](double a, double b) { return a / b; },
                     [](float a, float b) { return a / b; }, dsDiv,
                     expansionNDiv);
  } else if (node->op == "sin") {
    res = evalUnary(static_cast<double (*)(double)>(std::sin),
                    static_cast<float (*)(float)>(sinf));
  } else if (node->op == "cos") {
    res = evalUnary(static_cast<double (*)(double)>(std::cos),
                    static_cast<float (*)(float)>(cosf));
  } else if (node->op == "tan") {
    res = evalUnary(static_cast<double (*)(double)>(std::tan),
                    static_cast<float (*)(float)>(tanf));
  } else if (node->op == "exp") {
    res = evalUnary(static_cast<double (*)(double)>(std::exp),
                    static_cast<float (*)(float)>(expf));
  } else if (node->op == "expm1") {
    res = evalUnary(static_cast<double (*)(double)>(std::expm1),
                    static_cast<float (*)(float)>(expm1f));
  } else if (node->op == "log") {
    res = evalUnary(static_cast<double (*)(double)>(std::log),
                    static_cast<float (*)(float)>(logf));
  } else if (node->op == "log1p") {
    res = evalUnary(static_cast<double (*)(double)>(std::log1p),
                    static_cast<float (*)(float)>(log1pf));
  } else if (node->op == "sqrt") {
    res =
        evalUnary(static_cast<double (*)(double)>(std::sqrt),
                  static_cast<float (*)(float)>(sqrtf), dsSqrt, expansionNSqrt);
  } else if (node->op == "cbrt") {
    res = evalUnary(static_cast<double (*)(double)>(std::cbrt),
                    static_cast<float (*)(float)>(cbrtf));
  } else if (node->op == "asin") {
    res = evalUnary(static_cast<double (*)(double)>(std::asin),
                    static_cast<float (*)(float)>(asinf));
  } else if (node->op == "acos") {
    res = evalUnary(static_cast<double (*)(double)>(std::acos),
                    static_cast<float (*)(float)>(acosf));
  } else if (node->op == "atan") {
    res = evalUnary(static_cast<double (*)(double)>(std::atan),
                    static_cast<float (*)(float)>(atanf));
  } else if (node->op == "sinh") {
    res = evalUnary(static_cast<double (*)(double)>(std::sinh),
                    static_cast<float (*)(float)>(sinhf));
  } else if (node->op == "cosh") {
    res = evalUnary(static_cast<double (*)(double)>(std::cosh),
                    static_cast<float (*)(float)>(coshf));
  } else if (node->op == "tanh") {
    res = evalUnary(static_cast<double (*)(double)>(std::tanh),
                    static_cast<float (*)(float)>(tanhf));
  } else if (node->op == "asinh") {
    res = evalUnary(static_cast<double (*)(double)>(std::asinh),
                    static_cast<float (*)(float)>(asinhf));
  } else if (node->op == "acosh") {
    res = evalUnary(static_cast<double (*)(double)>(std::acosh),
                    static_cast<float (*)(float)>(acoshf));
  } else if (node->op == "atanh") {
    res = evalUnary(static_cast<double (*)(double)>(std::atanh),
                    static_cast<float (*)(float)>(atanhf));
  } else if (node->op == "ceil") {
    res = evalUnary(static_cast<double (*)(double)>(std::ceil),
                    static_cast<float (*)(float)>(ceilf));
  } else if (node->op == "floor") {
    res = evalUnary(static_cast<double (*)(double)>(std::floor),
                    static_cast<float (*)(float)>(floorf));
  } else if (node->op == "exp2") {
    res = evalUnary(static_cast<double (*)(double)>(std::exp2),
                    static_cast<float (*)(float)>(exp2f));
  } else if (node->op == "log10") {
    res = evalUnary(static_cast<double (*)(double)>(std::log10),
                    static_cast<float (*)(float)>(log10f));
  } else if (node->op == "log2") {
    res = evalUnary(static_cast<double (*)(double)>(std::log2),
                    static_cast<float (*)(float)>(log2f));
  } else if (node->op == "rint") {
    res = evalUnary(static_cast<double (*)(double)>(std::rint),
                    static_cast<float (*)(float)>(rintf));
  } else if (node->op == "round") {
    res = evalUnary(static_cast<double (*)(double)>(std::round),
                    static_cast<float (*)(float)>(roundf));
  } else if (node->op == "trunc") {
    res = evalUnary(static_cast<double (*)(double)>(std::trunc),
                    static_cast<float (*)(float)>(truncf));
  } else if (node->op == "pow") {
    res = evalBinary(static_cast<double (*)(double, double)>(std::pow),
                     static_cast<float (*)(float, float)>(powf));
  } else if (node->op == "fabs") {
    res = evalUnary(static_cast<double (*)(double)>(std::fabs),
                    static_cast<float (*)(float)>(fabsf));
  } else if (node->op == "hypot") {
    res = evalBinary(static_cast<double (*)(double, double)>(std::hypot),
                     static_cast<float (*)(float, float)>(hypotf));
  } else if (node->op == "atan2") {
    res = evalBinary(static_cast<double (*)(double, double)>(std::atan2),
                     static_cast<float (*)(float, float)>(atan2f));
  } else if (node->op == "copysign") {
    res = evalBinary(static_cast<double (*)(double, double)>(std::copysign),
                     static_cast<float (*)(float, float)>(copysignf));
  } else if (node->op == "fmax") {
    res = evalBinary(static_cast<double (*)(double, double)>(std::fmax),
                     static_cast<float (*)(float, float)>(fmaxf));
  } else if (node->op == "fmin") {
    res = evalBinary(static_cast<double (*)(double, double)>(std::fmin),
                     static_cast<float (*)(float, float)>(fminf));
  } else if (node->op == "fdim") {
    res = evalBinary(static_cast<double (*)(double, double)>(std::fdim),
                     static_cast<float (*)(float, float)>(fdimf));
  } else if (node->op == "fmod") {
    res = evalBinary(static_cast<double (*)(double, double)>(std::fmod),
                     static_cast<float (*)(float, float)>(fmodf));
  } else if (node->op == "remainder") {
    res = evalBinary(static_cast<double (*)(double, double)>(std::remainder),
                     static_cast<float (*)(float, float)>(remainderf));
  } else if (node->op == "fma") {
    res = evalTernary(static_cast<double (*)(double, double, double)>(std::fma),
                      static_cast<float (*)(float, float, float)>(fmaf), dsFma,
                      expansionNMul, expansionNAdd);
  } else if (node->op == "lgamma") {
    res = evalUnary(static_cast<double (*)(double)>(std::lgamma),
                    static_cast<float (*)(float)>(lgammaf));
  } else if (node->op == "tgamma") {
    res = evalUnary(static_cast<double (*)(double)>(std::tgamma),
                    static_cast<float (*)(float)>(tgammaf));
  } else if (node->op == "==") {
    res = evalCompare([](auto x, auto y) { return x == y; });
  } else if (node->op == "!=") {
    res = evalCompare([](auto x, auto y) { return x != y; });
  } else if (node->op == "<") {
    res = evalCompare([](auto x, auto y) { return x < y; });
  } else if (node->op == ">") {
    res = evalCompare([](auto x, auto y) { return x > y; });
  } else if (node->op == "<=") {
    res = evalCompare([](auto x, auto y) { return x <= y; });
  } else if (node->op == ">=") {
    res = evalCompare([](auto x, auto y) { return x >= y; });
  } else if (node->op == "PI") {
    res = M_PI;
  } else if (node->op == "E") {
    res = M_E;
  } else if (node->op == "INFINITY") {
    res = INFINITY;
  } else if (node->op == "NAN") {
    res = NAN;
  } else if (node->op == "binary64->binary32" ||
             node->op == "binary32->binary64") {
    // The value operation is the identity; the rounding is the node's own
    // precision (set by parseHerbieExpr to the destination format), which
    // evalUnary applies.
    res = evalUnary([](double x) { return x; }, [](float x) { return x; },
                    nullptr);
  } else {
    std::string msg = "FPEvaluator: Unexpected operator " + node->op;
    llvm_unreachable(msg.c_str());
  }

  cache.emplace(node, res);
}

double FPEvaluator::getResult(const FPNode *node) const {
  auto it = cache.find(node);
  assert(it != cache.end() && "Node not evaluated yet");
  return it->second;
}

MPFREvaluator::CachedValue::CachedValue(unsigned prec) : prec(prec) {
  mpfr_init2(value, prec);
  mpfr_set_zero(value, 1);
}

MPFREvaluator::CachedValue::CachedValue(CachedValue &&other) noexcept
    : prec(other.prec) {
  mpfr_init2(value, other.prec);
  mpfr_swap(value, other.value);
}

MPFREvaluator::CachedValue &
MPFREvaluator::CachedValue::operator=(CachedValue &&other) noexcept {
  if (this != &other) {
    mpfr_set_prec(value, other.prec);
    prec = other.prec;
    mpfr_swap(value, other.value);
  }
  return *this;
}

MPFREvaluator::CachedValue::~CachedValue() { mpfr_clear(value); }

MPFREvaluator::MPFREvaluator(unsigned prec, PTCandidate *pt) : prec(prec) {
  if (pt) {
    for (const auto &change : pt->changes) {
      for (auto node : change.nodes) {
        nodeToNewPrec[node] = getMPFRPrec(change.newType);
      }
    }
  }
}

unsigned MPFREvaluator::getNodePrecision(const FPNode *node,
                                         bool groundTruth) const {
  if (groundTruth)
    return prec;

  auto it = nodeToNewPrec.find(node);
  if (it != nodeToNewPrec.end()) {
    return it->second;
  }

  return node->getMPFRPrec();
}

void MPFREvaluator::evaluateNode(const FPNode *node,
                                 const MapVector<Value *, double> &inputValues,
                                 bool groundTruth) {
  if (cache.find(node) != cache.end())
    return;

  if (isa<FPConst>(node)) {
    double constVal = node->getLowerBound();
    CachedValue cv(53);
    mpfr_set_d(cv.value, constVal, MPFR_RNDN);
    cache.emplace(node, CachedValue(std::move(cv)));
    return;
  }

  if (isa<FPLLValue>(node) && inputValues.count(cast<FPLLValue>(node)->value)) {
    double inputValue = inputValues.lookup(cast<FPLLValue>(node)->value);
    CachedValue cv(53);
    mpfr_set_d(cv.value, inputValue, MPFR_RNDN);
    cache.emplace(node, std::move(cv));
    return;
  }

  if (node->op == "if") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &cond = getResult(node->operands[0].get());
    if (0 == mpfr_cmp_ui(cond, 1)) {
      evaluateNode(node->operands[1].get(), inputValues, groundTruth);
      mpfr_t &then_val = getResult(node->operands[1].get());
      cache.emplace(node, CachedValue(cache.at(node->operands[1].get()).prec));
      mpfr_set(cache.at(node).value, then_val, MPFR_RNDN);
    } else {
      evaluateNode(node->operands[2].get(), inputValues, groundTruth);
      mpfr_t &else_val = getResult(node->operands[2].get());
      cache.emplace(node, CachedValue(cache.at(node->operands[2].get()).prec));
      mpfr_set(cache.at(node).value, else_val, MPFR_RNDN);
    }
    return;
  }

  unsigned nodePrec = getNodePrecision(node, groundTruth);
  cache.emplace(node, CachedValue(nodePrec));
  mpfr_t &res = cache.at(node).value;

  if (node->op == "neg") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_neg(res, op, MPFR_RNDN);
  } else if (node->op == "+") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    evaluateNode(node->operands[1].get(), inputValues, groundTruth);
    mpfr_t &op0 = getResult(node->operands[0].get());
    mpfr_t &op1 = getResult(node->operands[1].get());
    mpfr_prec_round(op0, nodePrec, MPFR_RNDN);
    mpfr_prec_round(op1, nodePrec, MPFR_RNDN);
    mpfr_add(res, op0, op1, MPFR_RNDN);
  } else if (node->op == "-") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    evaluateNode(node->operands[1].get(), inputValues, groundTruth);
    mpfr_t &op0 = getResult(node->operands[0].get());
    mpfr_t &op1 = getResult(node->operands[1].get());
    mpfr_prec_round(op0, nodePrec, MPFR_RNDN);
    mpfr_prec_round(op1, nodePrec, MPFR_RNDN);
    mpfr_sub(res, op0, op1, MPFR_RNDN);
  } else if (node->op == "*") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    evaluateNode(node->operands[1].get(), inputValues, groundTruth);
    mpfr_t &op0 = getResult(node->operands[0].get());
    mpfr_t &op1 = getResult(node->operands[1].get());
    mpfr_prec_round(op0, nodePrec, MPFR_RNDN);
    mpfr_prec_round(op1, nodePrec, MPFR_RNDN);
    mpfr_mul(res, op0, op1, MPFR_RNDN);
  } else if (node->op == "/") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    evaluateNode(node->operands[1].get(), inputValues, groundTruth);
    mpfr_t &op0 = getResult(node->operands[0].get());
    mpfr_t &op1 = getResult(node->operands[1].get());
    mpfr_prec_round(op0, nodePrec, MPFR_RNDN);
    mpfr_prec_round(op1, nodePrec, MPFR_RNDN);
    mpfr_div(res, op0, op1, MPFR_RNDN);
  } else if (node->op == "sin") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_sin(res, op, MPFR_RNDN);
  } else if (node->op == "cos") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_cos(res, op, MPFR_RNDN);
  } else if (node->op == "tan") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_tan(res, op, MPFR_RNDN);
  } else if (node->op == "asin") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_asin(res, op, MPFR_RNDN);
  } else if (node->op == "acos") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_acos(res, op, MPFR_RNDN);
  } else if (node->op == "atan") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_atan(res, op, MPFR_RNDN);
  } else if (node->op == "atan2") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    evaluateNode(node->operands[1].get(), inputValues, groundTruth);
    mpfr_t &op0 = getResult(node->operands[0].get());
    mpfr_t &op1 = getResult(node->operands[1].get());
    mpfr_prec_round(op0, nodePrec, MPFR_RNDN);
    mpfr_prec_round(op1, nodePrec, MPFR_RNDN);
    mpfr_atan2(res, op0, op1, MPFR_RNDN);
  } else if (node->op == "exp") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_exp(res, op, MPFR_RNDN);
  } else if (node->op == "expm1") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_expm1(res, op, MPFR_RNDN);
  } else if (node->op == "log") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_log(res, op, MPFR_RNDN);
  } else if (node->op == "log1p") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_log1p(res, op, MPFR_RNDN);
  } else if (node->op == "sqrt") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_sqrt(res, op, MPFR_RNDN);
  } else if (node->op == "cbrt") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_cbrt(res, op, MPFR_RNDN);
  } else if (node->op == "pow") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    evaluateNode(node->operands[1].get(), inputValues, groundTruth);
    mpfr_t &op0 = getResult(node->operands[0].get());
    mpfr_t &op1 = getResult(node->operands[1].get());
    mpfr_prec_round(op0, nodePrec, MPFR_RNDN);
    mpfr_prec_round(op1, nodePrec, MPFR_RNDN);
    mpfr_pow(res, op0, op1, MPFR_RNDN);
  } else if (node->op == "fma") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    evaluateNode(node->operands[1].get(), inputValues, groundTruth);
    evaluateNode(node->operands[2].get(), inputValues, groundTruth);
    mpfr_t &op0 = getResult(node->operands[0].get());
    mpfr_t &op1 = getResult(node->operands[1].get());
    mpfr_t &op2 = getResult(node->operands[2].get());
    mpfr_prec_round(op0, nodePrec, MPFR_RNDN);
    mpfr_prec_round(op1, nodePrec, MPFR_RNDN);
    mpfr_prec_round(op2, nodePrec, MPFR_RNDN);
    mpfr_fma(res, op0, op1, op2, MPFR_RNDN);
  } else if (node->op == "fabs") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_abs(res, op, MPFR_RNDN);
  } else if (node->op == "hypot") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    evaluateNode(node->operands[1].get(), inputValues, groundTruth);
    mpfr_t &op0 = getResult(node->operands[0].get());
    mpfr_t &op1 = getResult(node->operands[1].get());
    mpfr_prec_round(op0, nodePrec, MPFR_RNDN);
    mpfr_prec_round(op1, nodePrec, MPFR_RNDN);
    mpfr_hypot(res, op0, op1, MPFR_RNDN);
  } else if (node->op == "asinh") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_asinh(res, op, MPFR_RNDN);
  } else if (node->op == "acosh") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_acosh(res, op, MPFR_RNDN);
  } else if (node->op == "atanh") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_atanh(res, op, MPFR_RNDN);
  } else if (node->op == "sinh") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_sinh(res, op, MPFR_RNDN);
  } else if (node->op == "cosh") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_cosh(res, op, MPFR_RNDN);
  } else if (node->op == "tanh") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_tanh(res, op, MPFR_RNDN);
  } else if (node->op == "ceil") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_ceil(res, op);
  } else if (node->op == "floor") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_floor(res, op);
  } else if (node->op == "erf") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_erf(res, op, MPFR_RNDN);
  } else if (node->op == "exp2") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_exp2(res, op, MPFR_RNDN);
  } else if (node->op == "log10") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_log10(res, op, MPFR_RNDN);
  } else if (node->op == "log2") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_log2(res, op, MPFR_RNDN);
  } else if (node->op == "rint") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_rint(res, op, MPFR_RNDN);
  } else if (node->op == "round") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_round(res, op);
  } else if (node->op == "trunc") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_trunc(res, op);
  } else if (node->op == "copysign") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    evaluateNode(node->operands[1].get(), inputValues, groundTruth);
    mpfr_t &op0 = getResult(node->operands[0].get());
    mpfr_t &op1 = getResult(node->operands[1].get());
    mpfr_prec_round(op0, nodePrec, MPFR_RNDN);
    mpfr_prec_round(op1, nodePrec, MPFR_RNDN);
    mpfr_copysign(res, op0, op1, MPFR_RNDN);
  } else if (node->op == "fdim") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    evaluateNode(node->operands[1].get(), inputValues, groundTruth);
    mpfr_t &op0 = getResult(node->operands[0].get());
    mpfr_t &op1 = getResult(node->operands[1].get());
    mpfr_prec_round(op0, nodePrec, MPFR_RNDN);
    mpfr_prec_round(op1, nodePrec, MPFR_RNDN);
    mpfr_dim(res, op0, op1, MPFR_RNDN);
  } else if (node->op == "fmod") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    evaluateNode(node->operands[1].get(), inputValues, groundTruth);
    mpfr_t &op0 = getResult(node->operands[0].get());
    mpfr_t &op1 = getResult(node->operands[1].get());
    mpfr_prec_round(op0, nodePrec, MPFR_RNDN);
    mpfr_prec_round(op1, nodePrec, MPFR_RNDN);
    mpfr_fmod(res, op0, op1, MPFR_RNDN);
  } else if (node->op == "remainder") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    evaluateNode(node->operands[1].get(), inputValues, groundTruth);
    mpfr_t &op0 = getResult(node->operands[0].get());
    mpfr_t &op1 = getResult(node->operands[1].get());
    mpfr_prec_round(op0, nodePrec, MPFR_RNDN);
    mpfr_prec_round(op1, nodePrec, MPFR_RNDN);
    mpfr_remainder(res, op0, op1, MPFR_RNDN);
  } else if (node->op == "fmax") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    evaluateNode(node->operands[1].get(), inputValues, groundTruth);
    mpfr_t &op0 = getResult(node->operands[0].get());
    mpfr_t &op1 = getResult(node->operands[1].get());
    mpfr_prec_round(op0, nodePrec, MPFR_RNDN);
    mpfr_prec_round(op1, nodePrec, MPFR_RNDN);
    mpfr_max(res, op0, op1, MPFR_RNDN);
  } else if (node->op == "fmin") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    evaluateNode(node->operands[1].get(), inputValues, groundTruth);
    mpfr_t &op0 = getResult(node->operands[0].get());
    mpfr_t &op1 = getResult(node->operands[1].get());
    mpfr_prec_round(op0, nodePrec, MPFR_RNDN);
    mpfr_prec_round(op1, nodePrec, MPFR_RNDN);
    mpfr_min(res, op0, op1, MPFR_RNDN);
  } else if (node->op == "==") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    evaluateNode(node->operands[1].get(), inputValues, groundTruth);
    mpfr_t &op0 = getResult(node->operands[0].get());
    mpfr_t &op1 = getResult(node->operands[1].get());
    if (0 == mpfr_cmp(op0, op1))
      mpfr_set_ui(res, 1, MPFR_RNDN);
    else
      mpfr_set_ui(res, 0, MPFR_RNDN);
  } else if (node->op == "!=") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    evaluateNode(node->operands[1].get(), inputValues, groundTruth);
    mpfr_t &op0 = getResult(node->operands[0].get());
    mpfr_t &op1 = getResult(node->operands[1].get());
    if (0 != mpfr_cmp(op0, op1))
      mpfr_set_ui(res, 1, MPFR_RNDN);
    else
      mpfr_set_ui(res, 0, MPFR_RNDN);
  } else if (node->op == "<") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    evaluateNode(node->operands[1].get(), inputValues, groundTruth);
    mpfr_t &op0 = getResult(node->operands[0].get());
    mpfr_t &op1 = getResult(node->operands[1].get());
    if (0 > mpfr_cmp(op0, op1))
      mpfr_set_ui(res, 1, MPFR_RNDN);
    else
      mpfr_set_ui(res, 0, MPFR_RNDN);
  } else if (node->op == ">") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    evaluateNode(node->operands[1].get(), inputValues, groundTruth);
    mpfr_t &op0 = getResult(node->operands[0].get());
    mpfr_t &op1 = getResult(node->operands[1].get());
    if (0 < mpfr_cmp(op0, op1))
      mpfr_set_ui(res, 1, MPFR_RNDN);
    else
      mpfr_set_ui(res, 0, MPFR_RNDN);
  } else if (node->op == "<=") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    evaluateNode(node->operands[1].get(), inputValues, groundTruth);
    mpfr_t &op0 = getResult(node->operands[0].get());
    mpfr_t &op1 = getResult(node->operands[1].get());
    if (0 >= mpfr_cmp(op0, op1))
      mpfr_set_ui(res, 1, MPFR_RNDN);
    else
      mpfr_set_ui(res, 0, MPFR_RNDN);
  } else if (node->op == ">=") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    evaluateNode(node->operands[1].get(), inputValues, groundTruth);
    mpfr_t &op0 = getResult(node->operands[0].get());
    mpfr_t &op1 = getResult(node->operands[1].get());
    if (0 <= mpfr_cmp(op0, op1))
      mpfr_set_ui(res, 1, MPFR_RNDN);
    else
      mpfr_set_ui(res, 0, MPFR_RNDN);
  } else if (node->op == "and") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    evaluateNode(node->operands[1].get(), inputValues, groundTruth);
    mpfr_t &op0 = getResult(node->operands[0].get());
    mpfr_t &op1 = getResult(node->operands[1].get());
    if (0 == mpfr_cmp_ui(op0, 1) && 0 == mpfr_cmp_ui(op1, 1))
      mpfr_set_ui(res, 1, MPFR_RNDN);
    else
      mpfr_set_ui(res, 0, MPFR_RNDN);
  } else if (node->op == "or") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    evaluateNode(node->operands[1].get(), inputValues, groundTruth);
    mpfr_t &op0 = getResult(node->operands[0].get());
    mpfr_t &op1 = getResult(node->operands[1].get());
    if (0 == mpfr_cmp_ui(op0, 1) || 0 == mpfr_cmp_ui(op1, 1))
      mpfr_set_ui(res, 1, MPFR_RNDN);
    else
      mpfr_set_ui(res, 0, MPFR_RNDN);
  } else if (node->op == "not") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_set_prec(res, nodePrec);
    if (0 == mpfr_cmp_ui(op, 1))
      mpfr_set_ui(res, 0, MPFR_RNDN);
    else
      mpfr_set_ui(res, 1, MPFR_RNDN);
  } else if (node->op == "TRUE") {
    mpfr_set_ui(res, 1, MPFR_RNDN);
  } else if (node->op == "FALSE") {
    mpfr_set_ui(res, 0, MPFR_RNDN);
  } else if (node->op == "PI") {
    mpfr_const_pi(res, MPFR_RNDN);
  } else if (node->op == "E") {
    mpfr_const_euler(res, MPFR_RNDN);
  } else if (node->op == "INFINITY") {
    mpfr_set_inf(res, 1);
  } else if (node->op == "NAN") {
    mpfr_set_nan(res);
  } else if (node->op == "binary64->binary32" ||
             node->op == "binary32->binary64") {
    // Identity on the value; mpfr_set into `res` at the node's precision
    // performs the destination rounding, and the ground-truth path deliberately
    // does not narrow.
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_set_prec(res, nodePrec);
    mpfr_set(res, getResult(node->operands[0].get()), MPFR_RNDN);
  } else {
    llvm::errs() << "MPFREvaluator: Unexpected operator '" << node->op << "'\n";
    llvm_unreachable("Unexpected operator encountered");
  }
}

mpfr_t &MPFREvaluator::getResult(FPNode *node) {
  assert(cache.count(node) > 0 && "MPFREvaluator: Unexpected unevaluated node");
  return cache.at(node).value;
}

const SmallVector<float, 4> *
FPEvaluator::getResultLimbs(const FPNode *node) const {
  auto it = expCache.find(node);
  return it == expCache.end() ? nullptr : &it->second;
}

void getFPValues(ArrayRef<FPNode *> outputs,
                 const MapVector<Value *, double> &inputValues,
                 SmallVectorImpl<double> &results, PTCandidate *pt) {
  assert(!outputs.empty());
  results.resize(outputs.size());

  FPEvaluator evaluator(pt);

  for (const auto *output : outputs) {
    evaluator.evaluateNode(output, inputValues);
  }

  for (size_t i = 0; i < outputs.size(); ++i) {
    results[i] = evaluator.getResult(outputs[i]);
  }
}

// The default path rounds both sides to a double before subtracting, so its
// finest resolvable error is one double ULP; here the reference stays in MPFR
// and an expansion candidate is lifted exactly as the sum of its limbs.
void getSampleErrorsWide(ArrayRef<FPNode *> outputs,
                         const MapVector<Value *, double> &inputValues,
                         SmallVectorImpl<double> &errors, unsigned refBits,
                         PTCandidate *pt,
                         SmallVectorImpl<char> *candNonFinite) {
  assert(!outputs.empty());
  assert(refBits > 0 && "getSampleErrorsWide: refBits must be positive");
  const size_t n = outputs.size();
  errors.assign(n, std::numeric_limits<double>::quiet_NaN());
  if (candNonFinite)
    candNonFinite->assign(n, 0);

  // Working precision for the reference and the subtraction. Generous: the
  // whole point is that the difference of two nearly equal values survives.
  const mpfr_prec_t work = (mpfr_prec_t)std::max(4u * refBits, 512u);

  std::vector<mpfr_t> gold(n);
  std::vector<bool> haveGold(n, false);
  for (size_t i = 0; i < n; ++i)
    mpfr_init2(gold[i], work);

  {
    std::vector<mpfr_exp_t> prevExp(n, 0);
    std::vector<char *> prevStr(n, nullptr);
    std::vector<int> prevSign(n, 0);
    std::vector<bool> converged(n, false);
    size_t numConverged = 0;
    unsigned curPrec = std::max(64u, refBits);

    while (true) {
      MPFREvaluator evaluator(curPrec, nullptr);
      for (const auto *output : outputs)
        evaluator.evaluateNode(output, inputValues, true);

      for (size_t i = 0; i < n; ++i) {
        if (converged[i])
          continue;
        mpfr_t &res = evaluator.getResult(outputs[i]);
        int sign = mpfr_sgn(res);
        mpfr_exp_t exp;
        char *str = mpfr_get_str(nullptr, &exp, 2, refBits, res, MPFR_RNDN);
        if (prevStr[i] && sign == prevSign[i] && exp == prevExp[i] &&
            strcmp(str, prevStr[i]) == 0) {
          converged[i] = true;
          ++numConverged;
          mpfr_set(gold[i], res, MPFR_RNDN);
          haveGold[i] = true;
          mpfr_free_str(str);
          mpfr_free_str(prevStr[i]);
          prevStr[i] = nullptr;
          continue;
        }
        if (prevStr[i])
          mpfr_free_str(prevStr[i]);
        prevStr[i] = str;
        prevExp[i] = exp;
        prevSign[i] = sign;
      }

      if (numConverged == n)
        break;

      curPrec *= 2;
      if (curPrec > flags::MaxMPFRPrec) {
        // Unconverged outputs keep haveGold == false and are reported as NaN,
        // which the reductions drop -- the same contract the double path has.
        for (size_t i = 0; i < n; ++i)
          if (prevStr[i])
            mpfr_free_str(prevStr[i]);
        break;
      }
    }
  }

  FPEvaluator evaluator(pt);
  for (const auto *output : outputs)
    evaluator.evaluateNode(output, inputValues);

  mpfr_t cand, diff, denom;
  mpfr_init2(cand, work);
  mpfr_init2(diff, work);
  mpfr_init2(denom, work);

  for (size_t i = 0; i < n; ++i) {
    if (!haveGold[i])
      continue;

    double collapsed = evaluator.getResult(outputs[i]);
    const SmallVector<float, 4> *limbs = evaluator.getResultLimbs(outputs[i]);
    if (limbs && !limbs->empty()) {
      // Exact: the limbs are non-overlapping and `work` >= 512 bits holds
      // their unevaluated sum with room to spare.
      mpfr_set_d(cand, (double)(*limbs)[0], MPFR_RNDN);
      for (size_t k = 1; k < limbs->size(); ++k)
        mpfr_add_d(cand, cand, (double)(*limbs)[k], MPFR_RNDN);
    } else {
      mpfr_set_d(cand, collapsed, MPFR_RNDN);
    }

    if (!mpfr_number_p(cand)) {
      // Same policy as sampleError: a spurious non-finite where the
      // reference is finite is catastrophic but must stay finite so the
      // reductions rank it worst instead of dropping it.
      if (candNonFinite)
        (*candNonFinite)[i] = 1;
      errors[i] = flags::NonfinitePenalty.getValue();
      continue;
    }

    mpfr_sub(diff, gold[i], cand, MPFR_RNDN);
    mpfr_abs(diff, diff, MPFR_RNDN);

    if (flags::RelativeError) {
      mpfr_abs(denom, gold[i], MPFR_RNDN);
      if (mpfr_zero_p(denom))
        mpfr_set_d(denom, std::numeric_limits<double>::min(), MPFR_RNDN);
      mpfr_div(diff, diff, denom, MPFR_RNDN);
    }
    errors[i] = mpfr_get_d(diff, MPFR_RNDN);
  }

  mpfr_clear(cand);
  mpfr_clear(diff);
  mpfr_clear(denom);
  for (size_t i = 0; i < n; ++i)
    mpfr_clear(gold[i]);
}

// Ground truth: evaluate at increasing MPFR precision until the first
// `groundTruthPrec` significand bits stop changing.
void getMPFRValues(ArrayRef<FPNode *> outputs,
                   const MapVector<Value *, double> &inputValues,
                   SmallVectorImpl<double> &results, bool groundTruth,
                   const unsigned groundTruthPrec, PTCandidate *pt) {
  assert(!outputs.empty());
  results.resize(outputs.size());

  if (!groundTruth) {
    MPFREvaluator evaluator(0, pt);

    for (const auto *output : outputs) {
      evaluator.evaluateNode(output, inputValues, false);
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
      results[i] = mpfr_get_d(evaluator.getResult(outputs[i]), MPFR_RNDN);
    }
    return;
  }

  unsigned curPrec = 64;
  std::vector<mpfr_exp_t> prevResExp(outputs.size(), 0);
  std::vector<char *> prevResStr(outputs.size(), nullptr);
  std::vector<int> prevResSign(outputs.size(), 0);
  std::vector<bool> converged(outputs.size(), false);
  size_t numConverged = 0;

  while (true) {
    MPFREvaluator evaluator(curPrec, nullptr);

    for (const auto *output : outputs) {
      evaluator.evaluateNode(output, inputValues, true);
    }

    for (size_t i = 0; i < outputs.size(); ++i) {
      if (converged[i])
        continue;

      mpfr_t &res = evaluator.getResult(outputs[i]);
      int resSign = mpfr_sgn(res);
      mpfr_exp_t resExp;
      char *resStr =
          mpfr_get_str(nullptr, &resExp, 2, groundTruthPrec, res, MPFR_RNDN);

      if (prevResStr[i] != nullptr && resSign == prevResSign[i] &&
          resExp == prevResExp[i] && strcmp(resStr, prevResStr[i]) == 0) {
        converged[i] = true;
        numConverged++;
        mpfr_free_str(resStr);
        mpfr_free_str(prevResStr[i]);
        prevResStr[i] = nullptr;
        continue;
      }

      if (prevResStr[i]) {
        mpfr_free_str(prevResStr[i]);
      }
      prevResStr[i] = resStr;
      prevResExp[i] = resExp;
      prevResSign[i] = resSign;
    }

    if (numConverged == outputs.size()) {
      for (size_t i = 0; i < outputs.size(); ++i) {
        results[i] = mpfr_get_d(evaluator.getResult(outputs[i]), MPFR_RNDN);
      }
      break;
    }

    curPrec *= 2;

    if (curPrec > flags::MaxMPFRPrec) {
      llvm::errs() << "getMPFRValues: MPFR precision limit reached for some "
                      "outputs, returning NaN\n";
      for (size_t i = 0; i < outputs.size(); ++i) {
        if (!converged[i]) {
          mpfr_free_str(prevResStr[i]);
          results[i] = std::numeric_limits<double>::quiet_NaN();
        } else {
          results[i] = mpfr_get_d(evaluator.getResult(outputs[i]), MPFR_RNDN);
        }
      }
      return;
    }
  }
}

} // namespace poseidon
