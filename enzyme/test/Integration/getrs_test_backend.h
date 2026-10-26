// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Portable test backend for the Fortran BLAS/LAPACK entry points used by getrs.
// Test kernels intentionally use scalar loops; production derivatives call
// BLAS.
#ifndef ENZYME_TEST_GETRS_BACKEND_H
#define ENZYME_TEST_GETRS_BACKEND_H
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#ifdef GETRS_TEST_BACKEND_IMPLEMENTATION

template <class T, class Int>
void test_getrs(const char *trans, const Int *n, const Int *nrhs, const T *a,
                const Int *lda, const Int *p, T *b, const Int *ldb, Int *info) {
  bool normal = *trans == 'N' || *trans == 'n';
  *info = 0;
  if (!normal && *trans != 'T' && *trans != 't' && *trans != 'C' &&
      *trans != 'c')
    *info = -1;
  else if (*n < 0)
    *info = -2;
  else if (*nrhs < 0)
    *info = -3;
  else if (*lda < std::max<Int>(1, *n))
    *info = -5;
  else if (*ldb < std::max<Int>(1, *n))
    *info = -8;
  if (*info || !*n || !*nrhs)
    return;
  for (Int col = 0; col < *nrhs; ++col) {
    T *x = b + col * *ldb;
    if (normal) {
      for (Int i = 0; i < *n; ++i)
        std::swap(x[i], x[p[i] - 1]);
      for (Int i = 0; i < *n; ++i)
        for (Int k = 0; k < i; ++k)
          x[i] -= a[i + k * *lda] * x[k];
      for (Int i = *n; i-- > 0;) {
        for (Int k = i + 1; k < *n; ++k)
          x[i] -= a[i + k * *lda] * x[k];
        x[i] /= a[i + i * *lda];
      }
    } else {
      for (Int i = 0; i < *n; ++i) {
        for (Int k = 0; k < i; ++k)
          x[i] -= a[k + i * *lda] * x[k];
        x[i] /= a[i + i * *lda];
      }
      for (Int i = *n; i-- > 0;)
        for (Int k = i + 1; k < *n; ++k)
          x[i] -= a[k + i * *lda] * x[k];
      for (Int i = *n; i-- > 0;)
        std::swap(x[i], x[p[i] - 1]);
    }
  }
}

template <class T, class Int>
void test_trmm(const char *side, const char *uplo, const char *trans,
               const char *diag, const Int *m, const Int *n, const T *alpha,
               const T *a, const Int *lda, T *b, const Int *ldb) {
  assert(*side == 'L');
  std::vector<T> in(*m * *n);
  for (Int j = 0; j < *n; ++j)
    for (Int i = 0; i < *m; ++i)
      in[i + j * *m] = b[i + j * *ldb];
  for (Int j = 0; j < *n; ++j)
    for (Int i = 0; i < *m; ++i) {
      T sum = 0;
      for (Int k = 0; k < *m; ++k) {
        Int r = *trans == 'N' ? i : k, c = *trans == 'N' ? k : i;
        bool inside = *uplo == 'L' ? r >= c : r <= c;
        T v = !inside ? T(0) : r == c && *diag == 'U' ? T(1) : a[r + c * *lda];
        sum += v * in[k + j * *m];
      }
      b[i + j * *ldb] = *alpha * sum;
    }
}

template <class T, class Int>
void test_gemm(const char *ta, const char *tb, const Int *m, const Int *n,
               const Int *k, const T *alpha, const T *a, const Int *lda,
               const T *b, const Int *ldb, const T *beta, T *c,
               const Int *ldc) {
  for (Int j = 0; j < *n; ++j)
    for (Int i = 0; i < *m; ++i) {
      T sum = 0;
      for (Int q = 0; q < *k; ++q) {
        T av = *ta == 'N' ? a[i + q * *lda] : a[q + i * *lda];
        T bv = *tb == 'N' ? b[q + j * *ldb] : b[j + q * *ldb];
        sum += av * bv;
      }
      c[i + j * *ldc] =
          *alpha * sum + (*beta == T(0) ? T(0) : *beta * c[i + j * *ldc]);
    }
}

template <class T, class Int>
void test_laswp(const Int *n, T *a, const Int *lda, const Int *k1,
                const Int *k2, const Int *p, const Int *inc) {
  assert(*inc == 1 || *inc == -1);
  for (Int q = 0; q <= *k2 - *k1; ++q) {
    Int i = *inc > 0 ? *k1 - 1 + q : *k2 - 1 - q;
    for (Int j = 0; j < *n; ++j)
      std::swap(a[i + j * *lda], a[p[i] - 1 + j * *lda]);
  }
}

#define GETRS_TEST_ABI(Name, T, Int, Suffix)                                   \
  extern "C" __attribute__((noinline)) void Name##getrs##Suffix(               \
      const char *t, const Int *n, const Int *r, const T *a, const Int *ld,    \
      const Int *p, T *b, const Int *ldb, Int *info, std::size_t) {            \
    test_getrs(t, n, r, a, ld, p, b, ldb, info);                               \
  }                                                                            \
  extern "C" __attribute__((noinline)) void Name##trmm##Suffix(                \
      const char *s, const char *u, const char *t, const char *d,              \
      const Int *m, const Int *n, const T *alpha, const T *a, const Int *lda,  \
      T *b, const Int *ldb, std::size_t, std::size_t, std::size_t,             \
      std::size_t) {                                                           \
    test_trmm(s, u, t, d, m, n, alpha, a, lda, b, ldb);                        \
  }                                                                            \
  extern "C" __attribute__((noinline)) void Name##gemm##Suffix(                \
      const char *ta, const char *tb, const Int *m, const Int *n,              \
      const Int *k, const T *alpha, const T *a, const Int *lda, const T *b,    \
      const Int *ldb, const T *beta, T *c, const Int *ldc, std::size_t,        \
      std::size_t) {                                                           \
    test_gemm(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);           \
  }                                                                            \
  extern "C" __attribute__((noinline)) void Name##laswp##Suffix(               \
      const Int *n, T *a, const Int *lda, const Int *k1, const Int *k2,        \
      const Int *p, const Int *inc) {                                          \
    test_laswp(n, a, lda, k1, k2, p, inc);                                     \
  }
GETRS_TEST_ABI(s, float, int, _)
GETRS_TEST_ABI(d, double, int, _)
GETRS_TEST_ABI(s, float, std::int64_t, _64_)
GETRS_TEST_ABI(d, double, std::int64_t, _64_)
#undef GETRS_TEST_ABI
#else
#define GETRS_TEST_DECL(Name, T, Int, Suffix)                                  \
  extern "C" void Name##getrs##Suffix(const char *, const Int *, const Int *,  \
                                      const T *, const Int *, const Int *,     \
                                      T *, const Int *, Int *, std::size_t);   \
  extern "C" void Name##trmm##Suffix(                                          \
      const char *, const char *, const char *, const char *, const Int *,     \
      const Int *, const T *, const T *, const Int *, T *, const Int *,        \
      std::size_t, std::size_t, std::size_t, std::size_t);                     \
  extern "C" void Name##gemm##Suffix(                                          \
      const char *, const char *, const Int *, const Int *, const Int *,       \
      const T *, const T *, const Int *, const T *, const Int *, const T *,    \
      T *, const Int *, std::size_t, std::size_t);                             \
  extern "C" void Name##laswp##Suffix(const Int *, T *, const Int *,           \
                                      const Int *, const Int *, const Int *,   \
                                      const Int *);
GETRS_TEST_DECL(s, float, int, _)
GETRS_TEST_DECL(d, double, int, _)
GETRS_TEST_DECL(s, float, std::int64_t, _64_)
GETRS_TEST_DECL(d, double, std::int64_t, _64_)
#undef GETRS_TEST_DECL
#endif

#ifdef GETRS_TEST_ILP64
#define LAPACK_sgetrs(...) sgetrs_64_(__VA_ARGS__, 1)
#define LAPACK_dgetrs(...) dgetrs_64_(__VA_ARGS__, 1)
#else
#define LAPACK_sgetrs(...) sgetrs_(__VA_ARGS__, 1)
#define LAPACK_dgetrs(...) dgetrs_(__VA_ARGS__, 1)
#endif
#endif
