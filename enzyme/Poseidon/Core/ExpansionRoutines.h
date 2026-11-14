//=- ExpansionRoutines.h - FP32 expansion arithmetic, written once --------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The FP32 expansion routines, written once over the arithmetic they run on:
// Techniques/Expansion.cpp emits them through an IRBuilder, and
// Core/ExpansionModel.cpp evaluates them on host floats for the accuracy
// model, so that the model computes what the emitted code computes. Every
// operation is bound to a local, so that instructions are created in the order
// written, and is given the name of the instruction it emits.
//
// Arith supplies the types F32 (a component), F64, Bool and Pair (an aggregate
// {hi, lo} of F32) and the operations, each rounded once: add and sub (of F32
// or of F64), mul, div, neg, fma (a * b + c), sqrt, equal, both (logical and),
// select, truncate (F64 to F32), extend (F32 to F64) and constant (an F32
// constant). fmaCall and sqrtCall are fma and sqrt emitted as a call even when
// the operands are constants, which fma and sqrt, like the other operations,
// may fold; the two-component routines use them.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_CORE_EXPANSION_ROUTINES_H
#define POSEIDON_CORE_EXPANSION_ROUTINES_H

#include <limits>

namespace poseidon {

template <typename Arith> class ExpansionRoutines {
public:
  using F32 = typename Arith::F32;
  using F64 = typename Arith::F64;
  using Bool = typename Arith::Bool;
  using Pair = typename Arith::Pair;

  explicit ExpansionRoutines(Arith A) : A(A) {}

  // Two components. Each routine but division and the fused multiply-add
  // reproduces, operation for operation, a MultiFloats.jl routine at Val{2}
  // (MultiFloats.jl commit 28041a3, MIT License): twoSum two_sum
  // (src/MultiFloats.jl 313-321), dsFastTwoSum fast_two_sum (1047-1052),
  // twoProd two_prod (1070-1074), dsAdd mfadd (src/mfadd.jl 10-24), dsSub
  // x + (-y) (src/MultiFloats.jl 1102, 1020), dsMul mfmul (src/mfmul.jl
  // 10-23), dsSqrt sqrt_r, i.e. mfsqrt (src/MultiFloats.jl 1416-1421) under
  // sqrt_r's zero test (1448-1449). dsDiv reproduces mX_real's division of its
  // two-component type at Algorithm::Sloppy, QxW's div_PA_PA_PA followed by
  // Normalize<Regular>, i.e. QxW's FastTwoSum (RIKEN-RCCS/mX_real, which
  // contains the QxW generator, is BSD-2-Clause). dsFma is Poseidon's own
  // routine, with no library counterpart; it is checked against the accuracy
  // model only.

  Pair twoSum(F32 a, F32 b) {
    F32 s = A.add(a, b, "ts.s");
    F32 ap = A.sub(s, b, "ts.ap");
    F32 bp = A.sub(s, ap, "ts.bp");
    F32 da = A.sub(a, ap, "ts.da");
    F32 db = A.sub(b, bp, "ts.db");
    F32 e = A.add(da, db, "ts.e");
    return {s, e};
  }

  Pair dsFastTwoSum(F32 a, F32 b) {
    F32 s = A.add(a, b, "fts.s");
    F32 bp = A.sub(s, a, "fts.bp");
    F32 e = A.sub(b, bp, "fts.e");
    return {s, e};
  }

  Pair twoProd(F32 a, F32 b) {
    F32 p = A.mul(a, b, "tp.p");
    F32 np = A.neg(p, "tp.np");
    F32 e = A.fmaCall(a, b, np, "tp.e");
    return {p, e};
  }

  Pair dsAdd(Pair x, Pair y) {
    Pair ab = twoSum(x.hi, y.hi);
    Pair cd = twoSum(x.lo, y.lo);
    Pair ac = dsFastTwoSum(ab.hi, cd.hi);
    F32 bd = A.add(ab.lo, cd.lo, "dsa.bd");
    F32 b3 = A.add(bd, ac.lo, "dsa.b3");
    return dsFastTwoSum(ac.hi, b3);
  }

  Pair dsSub(Pair x, Pair y) {
    Pair negY = dsNeg(y);
    return dsAdd(x, negY);
  }

  Pair dsMul(Pair x, Pair y) {
    Pair pe = twoProd(x.hi, y.hi);
    F32 c1 = A.mul(x.hi, y.lo, "dsm.c1");
    F32 c2 = A.mul(x.lo, y.hi, "dsm.c2");
    F32 cross = A.add(c1, c2, "dsm.cr");
    F32 e2 = A.add(pe.lo, cross, "dsm.e2");
    return dsFastTwoSum(pe.hi, e2);
  }

  // x * y + z, for llvm.fma and llvm.fmuladd: 16 F32 ops against 30 for dsMul
  // + dsAdd. The product's renormalizing fastTwoSum is dead work when the pair
  // is consumed by an addition, the cross terms become hardware FMAs into the
  // residual, and the addition's second twoSum over the low limbs is dropped
  // because after the exact twoSum of the high parts every remaining term is
  // O(eps) relative to max(|x*y|, |z|). x.lo*y.lo is dropped as in dsMul.
  // Low-limb accumulation is ordered smallest-magnitude first.
  Pair dsFma(Pair x, Pair y, Pair z) {
    // p + e == x.hi*y.hi exactly.
    F32 p = A.mul(x.hi, y.hi, "dsf.p");
    F32 np = A.neg(p, "dsf.np");
    F32 e = A.fmaCall(x.hi, y.hi, np, "dsf.e");
    F32 e1 = A.fmaCall(x.hi, y.lo, e, "dsf.e1");
    F32 e2 = A.fmaCall(x.lo, y.hi, e1, "dsf.e2");
    // s.hi + s.lo == p + z.hi exactly (captures catastrophic cancellation).
    Pair s = twoSum(p, z.hi);
    F32 t0 = A.add(e2, z.lo, "dsf.t0");
    F32 t1 = A.add(t0, s.lo, "dsf.t1");
    return dsFastTwoSum(s.hi, t1);
  }

  // The seed is the correctly rounded square root on every target
  // (sqrt.rn.f32 on NVPTX), so that the device computes what the host model
  // and the reference routines compute. sqrt_r returns a zero operand itself;
  // the correction would divide 0 by 0.
  Pair dsSqrt(Pair x) {
    F32 y = A.sqrtCall(x.hi, "dssq.y");
    F32 ny = A.neg(y, "dssq.ny");
    F32 rem = A.fmaCall(ny, y, x.hi, "dssq.rem");
    F32 r = A.add(rem, x.lo, "dssq.r");
    F32 twoY = A.add(y, y, "dssq.2y");
    F32 q = A.div(r, twoY, "dssq.q");
    Pair root = dsFastTwoSum(y, q);
    F32 zero = A.constant(0.0);
    Bool hiZero = A.equal(x.hi, zero, "dssq.hz");
    Bool loZero = A.equal(x.lo, zero, "dssq.lz");
    Bool isZero = A.both(hiZero, loZero, "dssq.z");
    F32 hi = A.select(isZero, x.hi, root.hi, "dssq.hi");
    F32 lo = A.select(isZero, x.lo, root.lo, "dssq.lo");
    return {hi, lo};
  }

  Pair dsNeg(Pair x) {
    F32 hi = A.neg(x.hi, "dsn.hi");
    F32 lo = A.neg(x.lo, "dsn.lo");
    return {hi, lo};
  }

  Pair dsDiv(Pair x, Pair y) {
    F32 c0, c1;
    expansionDivRaw22(x.hi, x.lo, y.hi, y.lo, c0, c1);
    return expansionFastTwoSum(c0, c1);
  }

  // An FP64 value: its rounding to F32, and the rounding of the rest.
  Pair dsFromF64(F64 v) {
    F32 hi = A.truncate(v, "ds.hi");
    F64 hiBack = A.extend(hi, "ds.hib");
    F64 rest = A.sub(v, hiBack, "ds.los");
    F32 lo = A.truncate(rest, "ds.lo");
    return {hi, lo};
  }

  // An FP32 value: exact in the hi slot, lo = 0.
  Pair dsFromF32(F32 v) { return {v, A.constant(0.0)}; }

  F64 dsToF64(Pair x) {
    F64 hi = A.extend(x.hi, "ds.hi64");
    F64 lo = A.extend(x.lo, "ds.lo64");
    return A.add(hi, lo, "ds.f64");
  }

  // To FP32: the sum of the halves, as accurate as FP32 allows.
  F32 dsToF32(Pair x) { return A.add(x.hi, x.lo, "ds.f32"); }

  // Three and four components. Each component sequence reproduces, operation
  // for operation, a routine of the QxW generator as instantiated by mX_real
  // at Algorithm::Sloppy (the add/mul/sqr/div/sqrt _QTW_ and _QQW_ variants,
  // each followed by Normalize<Regular>, and the special cases of mX_real's
  // sqrt operator; RIKEN-RCCS/mX_real, which contains the generator, is
  // BSD-2-Clause). twoSum forms its error from s - b where QxW's starts from
  // s - a, which gives the same sum, error and sign of zero. Subtraction is
  // addition of the negated operand, and a product of a value with itself is
  // the square routine (mX_real squares when both operands are one object).
  // Each takes the n components of its operands a and b and writes the n
  // components of the result to c.

  void expansionAdd(unsigned N, const F32 *a, const F32 *b, F32 *c) {
    if (N == 3) {
      expansionAddRaw3(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
      expansionNormalize3(c[0], c[1], c[2]);
      return;
    }
    expansionAddRaw4(a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], c[0], c[1],
                     c[2], c[3]);
    expansionNormalize4(c[0], c[1], c[2], c[3]);
  }

  void expansionSub(unsigned N, const F32 *a, const F32 *b, F32 *c) {
    F32 negB[4];
    expansionNeg(N, b, negB);
    expansionAdd(N, a, negB, c);
  }

  void expansionMul(unsigned N, const F32 *a, const F32 *b, F32 *c) {
    if (N == 3) {
      expansionMulRaw3(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
      expansionNormalize3(c[0], c[1], c[2]);
      return;
    }
    expansionMulRaw4(a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], c[0], c[1],
                     c[2], c[3]);
    expansionNormalize4(c[0], c[1], c[2], c[3]);
  }

  void expansionSqr(unsigned N, const F32 *a, F32 *c) {
    if (N == 3) {
      expansionSqrRaw3(a[0], a[1], a[2], c[0], c[1], c[2]);
      expansionNormalize3(c[0], c[1], c[2]);
      return;
    }
    expansionSqrRaw4(a[0], a[1], a[2], a[3], c[0], c[1], c[2], c[3]);
    expansionNormalize4(c[0], c[1], c[2], c[3]);
  }

  void expansionDiv(unsigned N, const F32 *a, const F32 *b, F32 *c) {
    if (N == 3) {
      expansionDivRaw3(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
      expansionNormalize3(c[0], c[1], c[2]);
      return;
    }
    expansionDivRaw4(a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], c[0], c[1],
                     c[2], c[3]);
    expansionNormalize4(c[0], c[1], c[2], c[3]);
  }

  // mX_real's sqrt operator returns the operand when its leading component is
  // zero and +Inf in every component when it is +Inf; the QxW routine gives
  // NaN for both. A negative or NaN leading component gives NaN either way.
  void expansionSqrt(unsigned N, const F32 *a, F32 *c) {
    if (N == 3) {
      expansionSqrtRaw3(a[0], a[1], a[2], c[0], c[1], c[2]);
      expansionNormalize3(c[0], c[1], c[2]);
    } else {
      expansionSqrtRaw4(a[0], a[1], a[2], a[3], c[0], c[1], c[2], c[3]);
      expansionNormalize4(c[0], c[1], c[2], c[3]);
    }
    F32 zero = A.constant(0.0);
    F32 inf = A.constant(std::numeric_limits<double>::infinity());
    Bool isZero = A.equal(a[0], zero, "mfx.sqz");
    Bool isInf = A.equal(a[0], inf, "mfx.sqi");
    for (unsigned i = 0; i < N; ++i) {
      F32 r = A.select(isInf, inf, c[i], "mfx.sqrti");
      c[i] = A.select(isZero, a[i], r, "mfx.sqrtz");
    }
  }

  void expansionNeg(unsigned N, const F32 *a, F32 *c) {
    for (unsigned i = 0; i < N; ++i)
      c[i] = A.neg(a[i], "mfx.neg");
  }

  // Greedy split into n components, which mX_real has no counterpart for (it
  // converts a double to its leading float); an F64 source fills at most three
  // limbs, the extra precision is created by the arithmetic.
  void expansionFromF64(unsigned N, F64 v, F32 *c) {
    F64 rem = v;
    for (unsigned i = 0; i < N; ++i) {
      c[i] = A.truncate(rem, "mfx.split");
      if (i + 1 < N) {
        F64 back = A.extend(c[i], "mfx.splitb");
        rem = A.sub(rem, back, "mfx.splitr");
      }
    }
  }

  void expansionFromF32(unsigned N, F32 v, F32 *c) {
    c[0] = v;
    for (unsigned i = 1; i < N; ++i)
      c[i] = A.constant(0.0);
  }

  // Sum the limbs least significant first; mX_real converts back from the
  // leading limb alone.
  F64 expansionToF64(unsigned N, const F32 *x) {
    F64 s = A.extend(x[N - 1], "mfx.e");
    for (int i = (int)N - 2; i >= 0; --i) {
      F64 e = A.extend(x[i], "mfx.e");
      s = A.add(e, s, "mfx.tof64");
    }
    return s;
  }

  F32 expansionToF32(unsigned N, const F32 *x) {
    F32 s = x[N - 1];
    for (int i = (int)N - 2; i >= 0; --i)
      s = A.add(x[i], s, "mfx.tof32");
    return s;
  }

private:
  Arith A;

  // The operations of the routines below, with the names they give the
  // emitted instructions.
  F32 add(F32 a, F32 b) { return A.add(a, b, "mfx.a"); }
  F32 mul(F32 a, F32 b) { return A.mul(a, b, "mfx.m"); }
  F32 div(F32 a, F32 b) { return A.div(a, b, "mfx.d"); }
  F32 neg(F32 a) { return A.neg(a, "mfx.n"); }
  F32 fma(F32 a, F32 b, F32 c) { return A.fma(a, b, c, "mfx.f"); }
  // Correctly-rounded F32 sqrt (sqrt.rn.f32 on NVPTX), NOT the MUFU
  // approximation the lossy F32 tier uses: the expansion's Newton
  // corrections only remove the seed's error to second/third order, so a
  // 2-ulp seed would cap a triple-float at ~1e-14 instead of ~1e-21.
  F32 sqrt(F32 a) { return A.sqrt(a, "mfx.q"); }

  // QxW's FastTwoSum, y = (a - x) + b. Not dsFastTwoSum's b - (s - a),
  // MultiFloats.jl's form: the two differ in the sign of a zero error.
  Pair expansionFastTwoSum(F32 a, F32 b) {
    F32 s = add(a, b);
    F32 d = A.sub(a, s, "mfx.s");
    F32 e = add(d, b);
    return {s, e};
  }

  // mX_real Normalize<NormalizeOption::Regular> for a 3-limb expansion at a
  // non-Quasi algorithm: three in-place fastTwoSum sweeps.
  void expansionNormalize3(F32 &x0, F32 &x1, F32 &x2) {
    Pair t = expansionFastTwoSum(x1, x2);
    x1 = t.hi;
    x2 = t.lo;
    t = expansionFastTwoSum(x0, x1);
    x0 = t.hi;
    x1 = t.lo;
    t = expansionFastTwoSum(x1, x2);
    x1 = t.hi;
    x2 = t.lo;
  }

  // mX_real Normalize<Regular> for a 4-limb expansion at a non-Quasi
  // algorithm.
  void expansionNormalize4(F32 &x0, F32 &x1, F32 &x2, F32 &x3) {
    Pair t = expansionFastTwoSum(x2, x3);
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

  // QxW::add_QTW_QTW_QTW  (3 + 3 -> 3, unnormalized)
  void expansionAddRaw3(F32 a0, F32 a1, F32 a2, F32 b0, F32 b1, F32 b2, F32 &c0,
                        F32 &c1, F32 &c2) {
    Pair s = twoSum(a0, b0);
    c0 = s.hi;
    c1 = s.lo;
    Pair u = twoSum(a1, b1);
    F32 t0 = u.hi;
    c2 = u.lo;
    Pair v = twoSum(c1, t0);
    c1 = v.hi;
    t0 = v.lo;
    F32 t1 = add(a2, b2);
    c2 = add(c2, t0);
    c2 = add(c2, t1);
  }

  // QxW::mul_QTW_QTW_QTW  (3 * 3 -> 3, unnormalized)
  void expansionMulRaw3(F32 a0, F32 a1, F32 a2, F32 b0, F32 b1, F32 b2, F32 &c0,
                        F32 &c1, F32 &c2) {
    Pair p = twoProd(a0, b0);
    c0 = p.hi;
    c1 = p.lo;
    Pair q = twoProd(a0, b1);
    c2 = q.hi;
    F32 t0 = q.lo;
    Pair r = twoProd(a1, b0);
    F32 t1 = r.hi;
    F32 t2 = r.lo;
    Pair s = twoSum(c1, c2);
    c1 = s.hi;
    c2 = s.lo;
    Pair u = twoSum(c1, t1);
    c1 = u.hi;
    t1 = u.lo;
    t0 = fma(a0, b2, t0);
    c2 = fma(a1, b1, c2);
    t2 = fma(a2, b0, t2);
    c2 = add(t0, c2);
    c2 = add(c2, t2);
    c2 = add(c2, t1);
  }

  // QxW::mul_PA_QTW_QTW  (2 * 3 -> 3, unnormalized)
  void expansionMulRaw23(F32 a0, F32 a1, F32 b0, F32 b1, F32 b2, F32 &c0,
                         F32 &c1, F32 &c2) {
    Pair p = twoProd(a0, b0);
    c0 = p.hi;
    c1 = p.lo;
    Pair q = twoProd(a0, b1);
    c2 = q.hi;
    F32 t0 = q.lo;
    Pair r = twoProd(a1, b0);
    F32 t1 = r.hi;
    F32 t2 = r.lo;
    Pair s = twoSum(c1, c2);
    c1 = s.hi;
    c2 = s.lo;
    Pair u = twoSum(c1, t1);
    c1 = u.hi;
    t1 = u.lo;
    t0 = fma(a0, b2, t0);
    c2 = fma(a1, b1, c2);
    c2 = add(t0, c2);
    c2 = add(c2, t2);
    c2 = add(c2, t1);
  }

  // QxW::div_PA_PA_PA  (2 / 2 -> 2, unnormalized)
  void expansionDivRaw22(F32 a0, F32 a1, F32 b0, F32 b1, F32 &c0, F32 &c1) {
    F32 bh = add(b0, b1);
    c0 = div(a0, bh);
    F32 nb0 = neg(b0);
    c1 = fma(nb0, c0, a0);
    c1 = add(c1, a1);
    F32 nb1 = neg(b1);
    c1 = fma(nb1, c0, c1);
    c1 = div(c1, bh);
  }

  // QxW::div_QTW_QTW_QTW  (3 / 3 -> 3, unnormalized)
  void expansionDivRaw3(F32 a0, F32 a1, F32 a2, F32 b0, F32 b1, F32 b2, F32 &c0,
                        F32 &c1, F32 &c2) {
    F32 e40 = add(a1, a2);
    F32 e41 = add(b1, b2);
    expansionDivRaw22(a0, e40, b0, e41, c0, c1);
    F32 t0, t1, t2;
    expansionMulRaw23(c0, c1, b0, b1, b2, t0, t1, t2);
    F32 n0 = neg(t0);
    F32 n1 = neg(t1);
    F32 n2 = neg(t2);
    F32 s0, s1, s2;
    expansionAddRaw3(a0, a1, a2, n0, n1, n2, s0, s1, s2);
    F32 tn = add(s0, s1);
    tn = add(tn, s2);
    F32 td = add(b0, b1);
    td = add(td, b2);
    c2 = div(tn, td);
  }

  // QxW::sqrt_QTW_PA  (sqrt of a 3-limb value to 2 limbs, unnormalized)
  void expansionSqrtRaw32(F32 a0, F32 a1, F32 a2, F32 &c0, F32 &c1) {
    c0 = sqrt(a0);
    F32 nc0 = neg(c0);
    F32 num = fma(nc0, c0, a0);
    num = add(num, a1);
    num = add(num, a2);
    F32 twoC0 = add(c0, c0);
    c1 = div(num, twoC0);
  }

  // QxW::sqr_PA_QTW  (square of a 2-limb value to 3 limbs, unnormalized)
  void expansionSqrRaw23(F32 a0, F32 a1, F32 &c0, F32 &c1, F32 &c2) {
    Pair p = twoProd(a0, a0);
    c0 = p.hi;
    c1 = p.lo;
    F32 t0 = add(a0, a0);
    Pair q = twoProd(t0, a1);
    c2 = q.hi;
    F32 t1 = q.lo;
    Pair s = twoSum(c1, c2);
    c1 = s.hi;
    c2 = s.lo;
    c2 = add(c2, t1);
    c2 = fma(a1, a1, c2);
  }

  // QxW::sqr_QTW_QTW  (3 -> 3, unnormalized)
  void expansionSqrRaw3(F32 a0, F32 a1, F32 a2, F32 &c0, F32 &c1, F32 &c2) {
    Pair p = twoProd(a0, a0);
    c0 = p.hi;
    c1 = p.lo;
    F32 t0 = add(a0, a0);
    Pair q = twoProd(t0, a1);
    c2 = q.hi;
    F32 t1 = q.lo;
    Pair s = twoSum(c1, c2);
    c1 = s.hi;
    c2 = s.lo;
    c2 = add(c2, t1);
    c2 = fma(t0, a2, c2);
    c2 = fma(a1, a1, c2);
  }

  // QxW::sqrt_QTW_QTW  (3 -> 3, unnormalized)
  void expansionSqrtRaw3(F32 a0, F32 a1, F32 a2, F32 &c0, F32 &c1, F32 &c2) {
    expansionSqrtRaw32(a0, a1, a2, c0, c1);
    F32 t0, t1, t2;
    expansionSqrRaw23(c0, c1, t0, t1, t2);
    F32 n0 = neg(t0);
    F32 n1 = neg(t1);
    F32 n2 = neg(t2);
    F32 s0, s1, s2;
    expansionAddRaw3(a0, a1, a2, n0, n1, n2, s0, s1, s2);
    F32 tn = add(s0, s1);
    tn = add(tn, s2);
    F32 td = add(c0, c1);
    td = mul(td, A.constant(2.0));
    c2 = div(tn, td);
  }

  // QxW::add_QQW_QQW_QQW  (4 + 4 -> 4, unnormalized)
  void expansionAddRaw4(F32 a0, F32 a1, F32 a2, F32 a3, F32 b0, F32 b1, F32 b2,
                        F32 b3, F32 &c0, F32 &c1, F32 &c2, F32 &c3) {
    Pair s = twoSum(a0, b0);
    c0 = s.hi;
    c1 = s.lo;
    Pair u = twoSum(a1, b1);
    F32 t0 = u.hi;
    c2 = u.lo;
    Pair v = twoSum(a2, b2);
    F32 t1 = v.hi;
    c3 = v.lo;
    Pair w = twoSum(c1, t0);
    c1 = w.hi;
    t0 = w.lo;
    Pair x = twoSum(c2, t0);
    c2 = x.hi;
    t0 = x.lo;
    Pair y = twoSum(c2, t1);
    c2 = y.hi;
    t1 = y.lo;
    F32 t2 = add(a3, b3);
    c3 = add(c3, t0);
    c3 = add(c3, t1);
    c3 = add(c3, t2);
  }

  // QxW::mul_QQW_QQW_QQW  (4 * 4 -> 4, unnormalized)
  void expansionMulRaw4(F32 a0, F32 a1, F32 a2, F32 a3, F32 b0, F32 b1, F32 b2,
                        F32 b3, F32 &c0, F32 &c1, F32 &c2, F32 &c3) {
    Pair p = twoProd(a0, b0);
    c0 = p.hi;
    c1 = p.lo;
    Pair q = twoProd(a0, b1);
    c2 = q.hi;
    c3 = q.lo;
    Pair r = twoProd(a1, b0);
    F32 t0 = r.hi;
    F32 t1 = r.lo;
    Pair s = twoSum(c1, c2);
    c1 = s.hi;
    c2 = s.lo;
    Pair u = twoSum(c1, t0);
    c1 = u.hi;
    t0 = u.lo;
    Pair v = twoProd(a0, b2);
    F32 t2 = v.hi;
    F32 t3 = v.lo;
    Pair w = twoProd(a1, b1);
    F32 t4 = w.hi;
    F32 t5 = w.lo;
    Pair x = twoProd(a2, b0);
    F32 t6 = x.hi;
    F32 t7 = x.lo;
    Pair y = twoSum(c2, t0);
    c2 = y.hi;
    t0 = y.lo;
    Pair z = twoSum(c2, c3);
    c2 = z.hi;
    c3 = z.lo;
    Pair aa = twoSum(c2, t1);
    c2 = aa.hi;
    t1 = aa.lo;
    Pair bb = twoSum(c2, t2);
    c2 = bb.hi;
    t2 = bb.lo;
    Pair cc = twoSum(c2, t4);
    c2 = cc.hi;
    t4 = cc.lo;
    Pair dd = twoSum(c2, t6);
    c2 = dd.hi;
    t6 = dd.lo;
    c3 = fma(a0, b3, c3);
    c3 = fma(a1, b2, c3);
    c3 = fma(a2, b1, c3);
    c3 = fma(a3, b0, c3);
    c3 = add(c3, t0);
    c3 = add(c3, t1);
    c3 = add(c3, t2);
    c3 = add(c3, t3);
    c3 = add(c3, t4);
    c3 = add(c3, t5);
    c3 = add(c3, t6);
    c3 = add(c3, t7);
  }

  // QxW::mul_QTW_QQW_QQW  (3 * 4 -> 4, unnormalized)
  void expansionMulRaw34(F32 a0, F32 a1, F32 a2, F32 b0, F32 b1, F32 b2, F32 b3,
                         F32 &c0, F32 &c1, F32 &c2, F32 &c3) {
    Pair p = twoProd(a0, b0);
    c0 = p.hi;
    c1 = p.lo;
    Pair q = twoProd(a0, b1);
    c2 = q.hi;
    c3 = q.lo;
    Pair r = twoProd(a1, b0);
    F32 t0 = r.hi;
    F32 t1 = r.lo;
    Pair s = twoSum(c1, c2);
    c1 = s.hi;
    c2 = s.lo;
    Pair u = twoSum(c1, t0);
    c1 = u.hi;
    t0 = u.lo;
    Pair v = twoProd(a0, b2);
    F32 t2 = v.hi;
    F32 t3 = v.lo;
    Pair w = twoProd(a1, b1);
    F32 t4 = w.hi;
    F32 t5 = w.lo;
    Pair x = twoProd(a2, b0);
    F32 t6 = x.hi;
    F32 t7 = x.lo;
    Pair y = twoSum(c2, t0);
    c2 = y.hi;
    t0 = y.lo;
    Pair z = twoSum(c2, c3);
    c2 = z.hi;
    c3 = z.lo;
    Pair aa = twoSum(c2, t1);
    c2 = aa.hi;
    t1 = aa.lo;
    Pair bb = twoSum(c2, t2);
    c2 = bb.hi;
    t2 = bb.lo;
    Pair cc = twoSum(c2, t4);
    c2 = cc.hi;
    t4 = cc.lo;
    Pair dd = twoSum(c2, t6);
    c2 = dd.hi;
    t6 = dd.lo;
    c3 = fma(a0, b3, c3);
    c3 = fma(a1, b2, c3);
    c3 = fma(a2, b1, c3);
    c3 = add(c3, t0);
    c3 = add(c3, t1);
    c3 = add(c3, t2);
    c3 = add(c3, t3);
    c3 = add(c3, t4);
    c3 = add(c3, t5);
    c3 = add(c3, t6);
    c3 = add(c3, t7);
  }

  // QxW::sqr_QTW_QQW  (square of a 3-limb value to 4 limbs, unnormalized)
  void expansionSqrRaw34(F32 a0, F32 a1, F32 a2, F32 &c0, F32 &c1, F32 &c2,
                         F32 &c3) {
    Pair p = twoProd(a0, a0);
    c0 = p.hi;
    c1 = p.lo;
    F32 t0 = add(a0, a0);
    Pair q = twoProd(t0, a1);
    c2 = q.hi;
    c3 = q.lo;
    Pair s = twoSum(c1, c2);
    c1 = s.hi;
    c2 = s.lo;
    Pair u = twoSum(c2, c3);
    c2 = u.hi;
    c3 = u.lo;
    Pair v = twoProd(t0, a2);
    F32 t1 = v.hi;
    F32 t2 = v.lo;
    Pair w = twoProd(a1, a1);
    F32 t3 = w.hi;
    F32 t4 = w.lo;
    Pair x = twoSum(c2, t1);
    c2 = x.hi;
    t1 = x.lo;
    Pair y = twoSum(c2, t3);
    c2 = y.hi;
    t3 = y.lo;
    F32 t5 = add(a1, a1);
    c3 = add(c3, t1);
    c3 = add(c3, t2);
    c3 = add(c3, t3);
    c3 = add(c3, t4);
    c3 = fma(t5, a2, c3);
  }

  // QxW::sqr_QQW_QQW  (4 -> 4, unnormalized)
  void expansionSqrRaw4(F32 a0, F32 a1, F32 a2, F32 a3, F32 &c0, F32 &c1,
                        F32 &c2, F32 &c3) {
    Pair p = twoProd(a0, a0);
    c0 = p.hi;
    c1 = p.lo;
    F32 t0 = add(a0, a0);
    Pair q = twoProd(t0, a1);
    c2 = q.hi;
    c3 = q.lo;
    Pair s = twoSum(c1, c2);
    c1 = s.hi;
    c2 = s.lo;
    Pair u = twoSum(c2, c3);
    c2 = u.hi;
    c3 = u.lo;
    Pair v = twoProd(t0, a2);
    F32 t1 = v.hi;
    F32 t2 = v.lo;
    Pair w = twoProd(a1, a1);
    F32 t3 = w.hi;
    F32 t4 = w.lo;
    Pair x = twoSum(c2, t1);
    c2 = x.hi;
    t1 = x.lo;
    Pair y = twoSum(c2, t3);
    c2 = y.hi;
    t3 = y.lo;
    F32 t5 = add(a1, a1);
    c3 = add(c3, t1);
    c3 = add(c3, t2);
    c3 = add(c3, t3);
    c3 = add(c3, t4);
    c3 = fma(t0, a3, c3);
    c3 = fma(t5, a2, c3);
  }

  // QxW::div_QQW_QQW_QQW  (4 / 4 -> 4, unnormalized)
  void expansionDivRaw4(F32 a0, F32 a1, F32 a2, F32 a3, F32 b0, F32 b1, F32 b2,
                        F32 b3, F32 &c0, F32 &c1, F32 &c2, F32 &c3) {
    F32 e40 = add(a2, a3);
    F32 e41 = add(b2, b3);
    expansionDivRaw3(a0, a1, e40, b0, b1, e41, c0, c1, c2);
    F32 t0, t1, t2, t3;
    expansionMulRaw34(c0, c1, c2, b0, b1, b2, b3, t0, t1, t2, t3);
    F32 n0 = neg(t0);
    F32 n1 = neg(t1);
    F32 n2 = neg(t2);
    F32 n3 = neg(t3);
    F32 s0, s1, s2, s3;
    expansionAddRaw4(a0, a1, a2, a3, n0, n1, n2, n3, s0, s1, s2, s3);
    F32 tn = add(s0, s1);
    tn = add(tn, s2);
    tn = add(tn, s3);
    F32 td = add(b0, b1);
    td = add(td, b2);
    td = add(td, b3);
    c3 = div(tn, td);
  }

  // QxW::sqrt_QQW_QQW  (4 -> 4, unnormalized). QxW::sqrt_QQW_QTW is
  // sqrt_QTW_QTW(a0, a1, a2 + a3, ...).
  void expansionSqrtRaw4(F32 a0, F32 a1, F32 a2, F32 a3, F32 &c0, F32 &c1,
                         F32 &c2, F32 &c3) {
    F32 a23 = add(a2, a3);
    expansionSqrtRaw3(a0, a1, a23, c0, c1, c2);
    F32 t0, t1, t2, t3;
    expansionSqrRaw34(c0, c1, c2, t0, t1, t2, t3);
    F32 n0 = neg(t0);
    F32 n1 = neg(t1);
    F32 n2 = neg(t2);
    F32 n3 = neg(t3);
    F32 s0, s1, s2, s3;
    expansionAddRaw4(a0, a1, a2, a3, n0, n1, n2, n3, s0, s1, s2, s3);
    F32 tn = add(s0, s1);
    tn = add(tn, s2);
    tn = add(tn, s3);
    F32 td = add(c0, c1);
    td = add(td, c2);
    td = mul(td, A.constant(2.0));
    c3 = div(tn, td);
  }
};

} // namespace poseidon

#endif // POSEIDON_CORE_EXPANSION_ROUTINES_H
