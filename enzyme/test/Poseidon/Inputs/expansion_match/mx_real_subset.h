// Reference routines for expansion_reference_identity.cpp: mX_real's
// three- and four-component FP32 types at Algorithm::Sloppy (tf_Real_sloppy,
// qf_Real_sloppy), the routines the materializer's n = 3, 4 arm reproduces,
// and the division of its two-component type (df_Real_sloppy), which the
// two-component division reproduces.
//
// Provenance: RIKEN-RCCS/mX_real, https://github.com/RIKEN-RCCS/mX_real,
// commit 72a35d697b247fb44400a282a88e22ea3211a327 (2025-01-04).
// - The QxW routines in namespace QxW are copied verbatim from
//   Ozaki-QW/qxw.hpp, each at the line range given above it: the routines the
//   three- and four-component types' +, -, *, / and sqrt and the
//   two-component type's / call, and every routine those call.
//   INLINE, NOEXCEPT and fp_const<T>::two(), which qxw.hpp takes from
//   mX_real.hpp and Ozaki-QW/fp_const.hpp, are defined before them.
// - The operators in namespace mxref condense mX_real's dispatch for these
//   types with mX_real.hpp's settings MX_REAL_USE_INF_NAN_EXCEPTION 0 and
//   MX_REAL_OPTIMIZE_MUL_BY_SQR 1 (mX_real.hpp lines 32-33). Starting lines:
//   Normalize<Regular> at a non-Quasi algorithm, mX_real.hpp 431, 469 and
//   519, whose quickSum is QxW::FastTwoSum (mX_real.hpp 331, 339); the
//   two-component quotient, operator/ (dX_real.hpp 1877), operator_div (1811;
//   its special-value branch is compiled out by the first setting) and
//   operator_div_body (1682: div_PA_PA_PA and Normalize); unary minus,
//   reversed_sign (tX_real.hpp and qX_real.hpp 660); a - b as a + (-b)
//   (tX_real.hpp and qX_real.hpp 1015); the add, mul, div and sqrt bodies
//   (tX_real.hpp 734, 1117, 1610, 2107; qX_real.hpp 679, 1044, 1546, 2095),
//   where the product of an object with itself is the square routine; and
//   the sqrt operator's special cases (tX_real.hpp and qX_real.hpp 2165),
//   whose is_zero, is_negative and isinf test the leading component at a
//   non-Quasi algorithm (quick_Normalized, mX_real.hpp 417; tX_real.hpp 531,
//   539, 547; qX_real.hpp 527, 543, 547).
//
// mX_real is distributed under the following license.
//
// BSD 2-Clause License
//
// Copyright (c) 2023-24, RIKEN
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Compile without floating-point contraction.

// clang-format off
#ifndef EXPANSION_MATCH_MX_REAL_SUBSET_H
#define EXPANSION_MATCH_MX_REAL_SUBSET_H

#include <cmath>
#include <limits>
#include <type_traits>

#define INLINE inline
#define NOEXCEPT noexcept

namespace QxW {

template <typename T> struct fp_const {
  static INLINE constexpr T two() NOEXCEPT { return T(2); }
};

// qxw.hpp lines 25-59
  template < typename T > INLINE void constexpr
  TwoSum ( T const a, T const b, T &x, T &y ) NOEXCEPT
  {
#if defined(__llvm__) || defined(__clang__)
#if __clang_major__ >= 11
#pragma clang fp reassociate(off)
#endif
#if __clang_major__ <= 10
#pragma clang optimize off
#endif
#endif

    //
    // basic cost is 6 flops and 6 cycles per a signle call
    //
#if defined(__CUDA_ARCH__)
    if ( std::is_same<T,double>::value ) {
      x = __dadd_rn(a, b);
    } else if ( std::is_same<T,float>::value ) {
      x = __fadd_rn(a, b);
    } else {
      x = a + b;
    }
#else
    x = a + b;
#endif
    auto z = x - a;
    y = (a - (x - z)) + (b - z);

#if defined(__llvm__) || defined(__clang__)
#if __clang_major__ <= 10
#pragma clang optimize on
#endif
#endif
  }

// qxw.hpp lines 61-94
  template < typename T > INLINE void constexpr
  FastTwoSum ( T const a, T const b, T &x, T &y ) NOEXCEPT
  {
#if defined(__llvm__) || defined(__clang__)
#if __clang_major__ >= 11
#pragma clang fp reassociate(off)
#endif
#if __clang_major__ <= 10
#pragma clang optimize off
#endif
#endif

    //
    // basic cost is 3 flops and 3 cycles per a signle call
    //
#if defined(__CUDA_ARCH__)
    if ( std::is_same<T,double>::value ) {
      x = __dadd_rn(a, b);
    } else if ( std::is_same<T,float>::value ) {
      x = __fadd_rn(a, b);
    } else {
      x = a + b;
    }
#else
    x = a + b;
#endif
    y = (a - x) + b;

#if defined(__llvm__) || defined(__clang__)
#if __clang_major__ <= 10
#pragma clang optimize on
#endif
#endif
  }

// qxw.hpp lines 96-108
  template < typename T > INLINE void constexpr
  TwoProductFMA ( T const a, T const b, T &x, T &y ) NOEXCEPT
  {

    //
    // basic cost is 3 flops but 2 cycles per a signle call
    // (fma with negative arg is expected to be replaced by a sigle instruction
    // at the compilation
    //
    x = a * b;
    y = std::fma(a, b, -x);

  }

// qxw.hpp lines 611-621
  // add: 3-3-3
  template < typename T > INLINE void constexpr
  add_QTW_QTW_QTW( T const a0, T const a1, T const a2, T const b0, T const b1, T const b2, T &c0, T &c1, T &c2 ) NOEXCEPT
  {
    T t0, t1;
    TwoSum( a0, b0, c0, c1 );
    TwoSum( a1, b1, t0, c2 );
    TwoSum( c1, t0, c1, t0 );
    t1 = a2 + b2;
    c2 = c2 + t0 + t1;
  }

// qxw.hpp lines 863-876
  // add: 4-4-4
  template < typename T > INLINE void constexpr
  add_QQW_QQW_QQW( T const a0, T const a1, T const a2, T const a3, T const b0, T const b1, T const b2, T const b3, T &c0, T &c1, T &c2, T &c3 ) NOEXCEPT
  {
    T t0, t1, t2;
    TwoSum( a0, b0, c0, c1 );
    TwoSum( a1, b1, t0, c2 );
    TwoSum( a2, b2, t1, c3 );
    TwoSum( c1, t0, c1, t0 );
    TwoSum( c2, t0, c2, t0 );
    TwoSum( c2, t1, c2, t1 );
    t2 = a3 + b3;
    c3 = c3 + t0 + t1 + t2;
  }

// qxw.hpp lines 1172-1177
  // sub: 3-3-3
  template < typename T > INLINE void constexpr
  sub_QTW_QTW_QTW( T const a0, T const a1, T const a2, T const b0, T const b1, T const b2, T &c0, T &c1, T &c2 ) NOEXCEPT
  {
    add_QTW_QTW_QTW( a0, a1, a2, -b0, -b1, -b2, c0, c1, c2 );
  }

// qxw.hpp lines 1319-1324
  // sub: 4-4-4
  template < typename T > INLINE void constexpr
  sub_QQW_QQW_QQW( T const a0, T const a1, T const a2, T const a3, T const b0, T const b1, T const b2, T const b3, T &c0, T &c1, T &c2, T &c3 ) NOEXCEPT
  {
    add_QQW_QQW_QQW( a0, a1, a2, a3, -b0, -b1, -b2, -b3, c0, c1, c2, c3 );
  }

// qxw.hpp lines 1601-1614
  // mul: 2-3-3
  template < typename T > INLINE void constexpr
  mul_PA_QTW_QTW( T const a0, T const a1, T const b0, T const b1, T const b2, T &c0, T &c1, T &c2 ) NOEXCEPT
  {
    T t0, t1, t2;
    TwoProductFMA( a0, b0, c0, c1 );
    TwoProductFMA( a0, b1, c2, t0 );
    TwoProductFMA( a1, b0, t1, t2 );
    TwoSum( c1, c2, c1, c2 );
    TwoSum( c1, t1, c1, t1 );
    t0 = std::fma ( a0, b2, t0 );
    c2 = std::fma ( a1, b1, c2 );
    c2 = t0 + c2 + t2 + t1;
  }

// qxw.hpp lines 1820-1834
  // mul: 3-3-3
  template < typename T > INLINE void constexpr
  mul_QTW_QTW_QTW( T const a0, T const a1, T const a2, T const b0, T const b1, T const b2, T &c0, T &c1, T &c2 ) NOEXCEPT
  {
    T t0, t1, t2;
    TwoProductFMA( a0, b0, c0, c1 );
    TwoProductFMA( a0, b1, c2, t0 );
    TwoProductFMA( a1, b0, t1, t2 );
    TwoSum( c1, c2, c1, c2 );
    TwoSum( c1, t1, c1, t1 );
    t0 = std::fma ( a0, b2, t0 );
    c2 = std::fma ( a1, b1, c2 );
    t2 = std::fma ( a2, b0, t2 );
    c2 = t0 + c2 + t2 + t1;
  }

// qxw.hpp lines 1899-1922
  // mul: 3-4-4
  template < typename T > INLINE void constexpr
  mul_QTW_QQW_QQW( T const a0, T const a1, T const a2, T const b0, T const b1, T const b2, T const b3, T &c0, T &c1, T &c2, T &c3 ) NOEXCEPT
  {
    T t0, t1, t2, t3, t4, t5, t6, t7;
    TwoProductFMA( a0, b0, c0, c1 );
    TwoProductFMA( a0, b1, c2, c3 );
    TwoProductFMA( a1, b0, t0, t1 );
    TwoSum( c1, c2, c1, c2 );
    TwoSum( c1, t0, c1, t0 );
    TwoProductFMA( a0, b2, t2, t3 );
    TwoProductFMA( a1, b1, t4, t5 );
    TwoProductFMA( a2, b0, t6, t7 );
    TwoSum( c2, t0, c2, t0 );
    TwoSum( c2, c3, c2, c3 );
    TwoSum( c2, t1, c2, t1 );
    TwoSum( c2, t2, c2, t2 );
    TwoSum( c2, t4, c2, t4 );
    TwoSum( c2, t6, c2, t6 );
    c3 = std::fma ( a0, b3, c3 );
    c3 = std::fma ( a1, b2, c3 );
    c3 = std::fma ( a2, b1, c3 );
    c3 = c3 + t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7;
  }

// qxw.hpp lines 2134-2158
  // mul: 4-4-4
  template < typename T > INLINE void constexpr
  mul_QQW_QQW_QQW( T const a0, T const a1, T const a2, T const a3, T const b0, T const b1, T const b2, T const b3, T &c0, T &c1, T &c2, T &c3 ) NOEXCEPT
  {
    T t0, t1, t2, t3, t4, t5, t6, t7;
    TwoProductFMA( a0, b0, c0, c1 );
    TwoProductFMA( a0, b1, c2, c3 );
    TwoProductFMA( a1, b0, t0, t1 );
    TwoSum( c1, c2, c1, c2 );
    TwoSum( c1, t0, c1, t0 );
    TwoProductFMA( a0, b2, t2, t3 );
    TwoProductFMA( a1, b1, t4, t5 );
    TwoProductFMA( a2, b0, t6, t7 );
    TwoSum( c2, t0, c2, t0 );
    TwoSum( c2, c3, c2, c3 );
    TwoSum( c2, t1, c2, t1 );
    TwoSum( c2, t2, c2, t2 );
    TwoSum( c2, t4, c2, t4 );
    TwoSum( c2, t6, c2, t6 );
    c3 = std::fma ( a0, b3, c3 );
    c3 = std::fma ( a1, b2, c3 );
    c3 = std::fma ( a2, b1, c3 );
    c3 = std::fma ( a3, b0, c3 );
    c3 = c3 + t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7;
  }

// qxw.hpp lines 2435-2443
  // div: 2-2-2
  template < typename T > INLINE void constexpr
  div_PA_PA_PA( T const a0, T const a1, T const b0, T const b1, T &c0, T &c1 ) NOEXCEPT
  {
    T bh = b0 + b1;
    c0 = a0 / bh;
    c1 = std::fma(-b0, c0, a0 ) + a1;
    c1 = std::fma(-b1, c0, c1 ) / bh;
  }

// qxw.hpp lines 2703-2719
  // div: 3-3-3
  template < typename T > INLINE void constexpr
  div_QTW_QTW_QTW( T const a0, T const a1, T const a2, T const b0, T const b1, T const b2, T &c0, T &c1, T &c2 ) NOEXCEPT
  {
    T t0, t1, t2;
    T e40;
    e40 = a1 + a2;
    T e41;
    e41 = b1 + b2;
    div_PA_PA_PA( a0, e40, b0, e41, c0, c1 );
    mul_PA_QTW_QTW( c0, c1, b0, b1, b2, t0, t1, t2 );
    sub_QTW_QTW_QTW( a0, a1, a2, t0, t1, t2, t0, t1, t2 );
    T tn, td;
    tn = t0 + t1 + t2;
    td = b0 + b1 + b2;
    c2 = tn / td;
  }

// qxw.hpp lines 2975-2991
  // div: 4-4-4
  template < typename T > INLINE void constexpr
  div_QQW_QQW_QQW( T const a0, T const a1, T const a2, T const a3, T const b0, T const b1, T const b2, T const b3, T &c0, T &c1, T &c2, T &c3 ) NOEXCEPT
  {
    T t0, t1, t2, t3;
    T e40;
    e40 = a2 + a3;
    T e41;
    e41 = b2 + b3;
    div_QTW_QTW_QTW( a0, a1, e40, b0, b1, e41, c0, c1, c2 );
    mul_QTW_QQW_QQW( c0, c1, c2, b0, b1, b2, b3, t0, t1, t2, t3 );
    sub_QQW_QQW_QQW( a0, a1, a2, a3, t0, t1, t2, t3, t0, t1, t2, t3 );
    T tn, td;
    tn = t0 + t1 + t2 + t3;
    td = b0 + b1 + b2 + b3;
    c3 = tn / td;
  }

// qxw.hpp lines 3043-3054
  // sqr: 2-3
  template < typename T > INLINE void constexpr
  sqr_PA_QTW( T const a0, T const a1, T &c0, T &c1, T &c2 ) NOEXCEPT
  {
    T t0, t1;
    TwoProductFMA( a0, a0, c0, c1 );
    t0 = a0 + a0;
    TwoProductFMA( t0, a1, c2, t1 );
    TwoSum( c1, c2, c1, c2 );
    c2 = c2 + t1;
    c2 = std::fma ( a1, a1, c2 );
  }

// qxw.hpp lines 3091-3103
  // sqr: 3-3
  template < typename T > INLINE void constexpr
  sqr_QTW_QTW( T const a0, T const a1, T const a2, T &c0, T &c1, T &c2 ) NOEXCEPT
  {
    T t0, t1;
    TwoProductFMA( a0, a0, c0, c1 );
    t0 = a0 + a0;
    TwoProductFMA( t0, a1, c2, t1 );
    TwoSum( c1, c2, c1, c2 );
    c2 = c2 + t1;
    c2 = std::fma ( t0, a2, c2 );
    c2 = std::fma ( a1, a1, c2 );
  }

// qxw.hpp lines 3105-3122
  // sqr: 3-4
  template < typename T > INLINE void constexpr
  sqr_QTW_QQW( T const a0, T const a1, T const a2, T &c0, T &c1, T &c2, T &c3 ) NOEXCEPT
  {
    T t0, t1, t2, t3, t4, t5;
    TwoProductFMA( a0, a0, c0, c1 );
    t0 = a0 + a0;
    TwoProductFMA( t0, a1, c2, c3 );
    TwoSum( c1, c2, c1, c2 );
    TwoSum( c2, c3, c2, c3 );
    TwoProductFMA( t0, a2, t1, t2 );
    TwoProductFMA( a1, a1, t3, t4 );
    TwoSum( c2, t1, c2, t1 );
    TwoSum( c2, t3, c2, t3 );
    t5 = a1 + a1;
    c3 = c3 + t1 + t2 + t3 + t4;
    c3 = std::fma ( t5, a2, c3 );
  }

// qxw.hpp lines 3159-3177
  // sqr: 4-4
  template < typename T > INLINE void constexpr
  sqr_QQW_QQW( T const a0, T const a1, T const a2, T const a3, T &c0, T &c1, T &c2, T &c3 ) NOEXCEPT
  {
    T t0, t1, t2, t3, t4, t5;
    TwoProductFMA( a0, a0, c0, c1 );
    t0 = a0 + a0;
    TwoProductFMA( t0, a1, c2, c3 );
    TwoSum( c1, c2, c1, c2 );
    TwoSum( c2, c3, c2, c3 );
    TwoProductFMA( t0, a2, t1, t2 );
    TwoProductFMA( a1, a1, t3, t4 );
    TwoSum( c2, t1, c2, t1 );
    TwoSum( c2, t3, c2, t3 );
    t5 = a1 + a1;
    c3 = c3 + t1 + t2 + t3 + t4;
    c3 = std::fma ( t0, a3, c3 );
    c3 = std::fma ( t5, a2, c3 );
  }

// qxw.hpp lines 7534-7540
  // sqrt: 3-2
  template < typename T > INLINE void constexpr
  sqrt_QTW_PA ( T const a0, T const a1, T const a2, T &c0, T &c1 ) NOEXCEPT
  {
    c0 = std::sqrt( a0 );
    c1 = ( std::fma( -c0, c0, a0 ) + a1 + a2 ) / (c0 + c0);
  }

// qxw.hpp lines 7542-7555
  // sqrt: 3-3
  template < typename T > INLINE void constexpr
  sqrt_QTW_QTW ( T const a0, T const a1, T const a2, T &c0, T &c1, T &c2 ) NOEXCEPT
  {
    T t0, t1, t2;
    sqrt_QTW_PA ( a0, a1, a2, c0, c1 );
    sqr_PA_QTW ( c0, c1, t0, t1, t2 );
    sub_QTW_QTW_QTW ( a0, a1, a2, t0, t1, t2, t0, t1, t2 );
    T tn, td;
    tn = t0 + t1 + t2;
    td = c0 + c1;
    td = td * fp_const<T>::two();
    c2 = tn / td;
  }

// qxw.hpp lines 7587-7592
  // sqrt: 4-3
  template < typename T > INLINE void constexpr
  sqrt_QQW_QTW ( T const a0, T const a1, T const a2, T const a3, T &c0, T &c1, T &c2 ) NOEXCEPT
  {
    sqrt_QTW_QTW ( a0, a1, a2+a3, c0, c1, c2 );
  }

// qxw.hpp lines 7594-7607
  // sqrt: 4-4
  template < typename T > INLINE void constexpr
  sqrt_QQW_QQW ( T const a0, T const a1, T const a2, T const a3, T &c0, T &c1, T &c2, T &c3 ) NOEXCEPT
  {
    T t0, t1, t2, t3;
    sqrt_QQW_QTW ( a0, a1, a2, a3, c0, c1, c2 );
    sqr_QTW_QQW ( c0, c1, c2, t0, t1, t2, t3 );
    sub_QQW_QQW_QQW ( a0, a1, a2, a3, t0, t1, t2, t3, t0, t1, t2, t3 );
    T tn, td;
    tn = t0 + t1 + t2 + t3;
    td = c0 + c1 + c2;
    td = td * fp_const<T>::two();
    c3 = tn / td;
  }
} // namespace QxW

namespace mxref {

// Normalize<NormalizeOption::Regular>, non-Quasi.
inline void normalize(float *c, int n) {
  if (n == 2) {
    QxW::FastTwoSum(c[0], c[1], c[0], c[1]);
  } else if (n == 3) {
    QxW::FastTwoSum(c[1], c[2], c[1], c[2]);
    QxW::FastTwoSum(c[0], c[1], c[0], c[1]);
    QxW::FastTwoSum(c[1], c[2], c[1], c[2]);
  } else {
    QxW::FastTwoSum(c[2], c[3], c[2], c[3]);
    QxW::FastTwoSum(c[1], c[2], c[1], c[2]);
    QxW::FastTwoSum(c[0], c[1], c[0], c[1]);
    QxW::FastTwoSum(c[2], c[3], c[2], c[3]);
    QxW::FastTwoSum(c[1], c[2], c[1], c[2]);
    QxW::FastTwoSum(c[2], c[3], c[2], c[3]);
  }
}

inline void neg(const float *a, float *c, int n) {
  for (int i = 0; i < n; ++i)
    c[i] = -a[i];
}

inline void add(const float *a, const float *b, float *c, int n) {
  if (n == 3)
    QxW::add_QTW_QTW_QTW(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
  else
    QxW::add_QQW_QQW_QQW(a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], c[0],
                         c[1], c[2], c[3]);
  normalize(c, n);
}

inline void sub(const float *a, const float *b, float *c, int n) {
  float nb[4];
  neg(b, nb, n);
  add(a, nb, c, n);
}

// a * b; `a == b` (one object) takes the square routine.
inline void mul(const float *a, const float *b, float *c, int n) {
  if (n == 3) {
    if (a == b)
      QxW::sqr_QTW_QTW(a[0], a[1], a[2], c[0], c[1], c[2]);
    else
      QxW::mul_QTW_QTW_QTW(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1],
                           c[2]);
  } else {
    if (a == b)
      QxW::sqr_QQW_QQW(a[0], a[1], a[2], a[3], c[0], c[1], c[2], c[3]);
    else
      QxW::mul_QQW_QQW_QQW(a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3],
                           c[0], c[1], c[2], c[3]);
  }
  normalize(c, n);
}

inline void div(const float *a, const float *b, float *c, int n) {
  if (n == 2)
    QxW::div_PA_PA_PA(a[0], a[1], b[0], b[1], c[0], c[1]);
  else if (n == 3)
    QxW::div_QTW_QTW_QTW(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
  else
    QxW::div_QQW_QQW_QQW(a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], c[0],
                         c[1], c[2], c[3]);
  normalize(c, n);
}

inline void sqrt(const float *a, float *c, int n) {
  if (a[0] == 0.0f) { // is_zero: return the operand
    for (int i = 0; i < n; ++i)
      c[i] = a[i];
    return;
  }
  if (a[0] < 0.0f) { // is_negative: nan()
    for (int i = 0; i < n; ++i)
      c[i] = std::numeric_limits<float>::quiet_NaN();
    return;
  }
  if (std::isinf(a[0])) { // isinf: inf()
    for (int i = 0; i < n; ++i)
      c[i] = std::numeric_limits<float>::infinity();
    return;
  }
  if (n == 3)
    QxW::sqrt_QTW_QTW(a[0], a[1], a[2], c[0], c[1], c[2]);
  else
    QxW::sqrt_QQW_QQW(a[0], a[1], a[2], a[3], c[0], c[1], c[2], c[3]);
  normalize(c, n);
}

} // namespace mxref

#endif // EXPANSION_MATCH_MX_REAL_SUBSET_H
// clang-format on
