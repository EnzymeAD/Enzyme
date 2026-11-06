// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// clang-format off
// RUN: %clang++ -std=c++2b -O1 -DGETRS_TEST_BACKEND_IMPLEMENTATION -x c++ -c %S/getrs_test_backend.h -o %t.backend.o
// RUN: if [ %llvmver -lt 17 ]; then flags="-Xclang -no-opaque-pointers"; else flags=""; fi; %clang++ -std=c++2b -O1 -fno-exceptions $flags -DDIFFERENTIATE_ENZYME -S -emit-llvm %s -o %t.input.ll && %opt %loadEnzyme %enzyme -S %t.input.ll -o %t.ad.ll && %clang++ -O2 %t.ad.ll %t.backend.o -o %t.exe && %t.exe
// RUN: if [ %llvmver -lt 17 ]; then flags="-Xclang -no-opaque-pointers"; else flags=""; fi; %clang++ -std=c++2b -O1 -fno-exceptions $flags -DDIFFERENTIATE_ENZYME -DGETRS_TEST_ILP64 -S -emit-llvm %s -o %t.input.ll && %opt %loadEnzyme %enzyme -S %t.input.ll -o %t.ad.ll && %clang++ -O2 %t.ad.ll %t.backend.o -o %t.exe && %t.exe
// RUN: if [ %llvmver -lt 17 ]; then flags="-Xclang -no-opaque-pointers"; else flags=""; fi; %clang++ -std=c++2b -O2 -fno-exceptions $flags -DDIFFERENTIATE_ENZYME -S -emit-llvm %s -o %t.input.ll && %opt %loadEnzyme %enzyme -S %t.input.ll -o %t.ad.ll && %clang++ -O2 %t.ad.ll %t.backend.o -o %t.exe && %t.exe
// clang-format on

// The backend is compiled separately: its implementation is not available to
// AD. This regression must fail without the native GETRS rule.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <type_traits>
#include <vector>
#ifdef GETRS_TEST_ILP64
using BlasInt = std::int64_t;
#else
using BlasInt = int;
#endif
#ifdef COMPARE_LAPACK
#include <lapacke.h>
#endif
#ifdef DIFFERENTIATE_ENZYME
#ifndef COMPARE_LAPACK
#include "getrs_test_backend.h"
#endif
int enzyme_const;
int enzyme_width;
template <typename... Args> void __enzyme_fwddiff(void *, Args...);
template <typename... Args> void __enzyme_autodiff(void *, Args...);

template <class T>
__attribute__((noinline)) void
getrs_primitive(char trans, int n, int nrhs, T *factors, int lda,
                BlasInt *pivots, T *rhs, int ldb, bool clobber) {
  BlasInt rows = n, columns = nrhs, leadingA = lda, leadingB = ldb, info = 0;
  if constexpr (std::is_same_v<T, double>)
    LAPACK_dgetrs(&trans, &rows, &columns, factors, &leadingA, pivots, rhs,
                  &leadingB, &info);
  else
    LAPACK_sgetrs(&trans, &rows, &columns, factors, &leadingA, pivots, rhs,
                  &leadingB, &info);
  if (clobber) {
    for (int j = 0; j < n; ++j) {
      pivots[j] = j + 1;
      for (int i = 0; i < n; ++i)
        factors[i + lda * j] = T(0);
    }
  }
}
#endif

template <class T> struct Matrix {
  int rows, cols, ld;
  std::vector<T> data;
  Matrix(int r, int c) : rows(r), cols(c), ld(r + 2), data(ld * c, T(999)) {
    for (int j = 0; j < cols; ++j)
      for (int i = 0; i < rows; ++i)
        (*this)(i, j) = T(0);
  }
  T &operator()(int i, int j) { return data[i + ld * j]; }
  T operator()(int i, int j) const { return data[i + ld * j]; }
  bool paddingIntact() const {
    for (int j = 0; j < cols; ++j)
      for (int i = rows; i < ld; ++i)
        if (data[i + ld * j] != T(999))
          return false;
    return true;
  }
};

template <class T> Matrix<T> transpose(const Matrix<T> &a) {
  Matrix<T> out(a.cols, a.rows);
  for (int j = 0; j < a.cols; ++j)
    for (int i = 0; i < a.rows; ++i)
      out(j, i) = a(i, j);
  return out;
}

template <class T> Matrix<T> multiply(const Matrix<T> &a, const Matrix<T> &b) {
  if (a.cols != b.rows)
    std::abort();
  Matrix<T> out(a.rows, b.cols);
  for (int j = 0; j < b.cols; ++j)
    for (int k = 0; k < a.cols; ++k)
      for (int i = 0; i < a.rows; ++i)
        out(i, j) += a(i, k) * b(k, j);
  return out;
}

template <class T>
void permute(Matrix<T> &a, const std::vector<BlasInt> &p, bool inverse) {
  for (int step = 0; step < a.rows; ++step) {
    int i = inverse ? a.rows - 1 - step : step;
    for (int j = 0; j < a.cols; ++j)
      std::swap(a(i, j), a(p[i] - 1, j));
  }
}

template <class T> Matrix<T> lower(const Matrix<T> &f, bool tangent = false) {
  Matrix<T> out(f.rows, f.cols);
  for (int j = 0; j < f.cols; ++j) {
    out(j, j) = tangent ? T(0) : T(1);
    for (int i = j + 1; i < f.rows; ++i)
      out(i, j) = f(i, j);
  }
  return out;
}

template <class T> Matrix<T> upper(const Matrix<T> &f) {
  Matrix<T> out(f.rows, f.cols);
  for (int j = 0; j < f.cols; ++j)
    for (int i = 0; i <= j; ++i)
      out(i, j) = f(i, j);
  return out;
}

// Scalar substitution is independent of the proposed generated BLAS derivative.
template <class T>
Matrix<T> solve(const Matrix<T> &f, const std::vector<BlasInt> &p, Matrix<T> b,
                bool trans) {
  int n = f.rows;
  if (!trans) {
    permute(b, p, false);
    for (int j = 0; j < b.cols; ++j) {
      for (int i = 0; i < n; ++i)
        for (int k = 0; k < i; ++k)
          b(i, j) -= f(i, k) * b(k, j);
      for (int i = n - 1; i >= 0; --i) {
        for (int k = i + 1; k < n; ++k)
          b(i, j) -= f(i, k) * b(k, j);
        b(i, j) /= f(i, i);
      }
    }
  } else {
    for (int j = 0; j < b.cols; ++j) {
      for (int i = 0; i < n; ++i) {
        for (int k = 0; k < i; ++k)
          b(i, j) -= f(k, i) * b(k, j);
        b(i, j) /= f(i, i);
      }
      for (int i = n - 1; i >= 0; --i)
        for (int k = i + 1; k < n; ++k)
          b(i, j) -= f(k, i) * b(k, j);
    }
    permute(b, p, true);
  }
  return b;
}

template <class T>
Matrix<T> add(const Matrix<T> &a, const Matrix<T> &b, T scale) {
  auto out = a;
  for (int j = 0; j < a.cols; ++j)
    for (int i = 0; i < a.rows; ++i)
      out(i, j) += scale * b(i, j);
  return out;
}

template <class T> T dot(const Matrix<T> &a, const Matrix<T> &b) {
  T out = 0;
  for (int j = 0; j < a.cols; ++j)
    for (int i = 0; i < a.rows; ++i)
      out += a(i, j) * b(i, j);
  return out;
}

template <class T> T maxDiff(const Matrix<T> &a, const Matrix<T> &b) {
  T out = 0;
  for (int j = 0; j < a.cols; ++j)
    for (int i = 0; i < a.rows; ++i)
      out = std::max(out, std::abs(a(i, j) - b(i, j)));
  return out;
}

template <class T>
bool check(int n, int nrhs, char operation, bool pivoted,
           bool clobber = false) {
  bool trans = operation != 'N' && operation != 'n';
  Matrix<T> f(n, n), df(n, n), b(n, nrhs), db(n, nrhs), g(n, nrhs);
  std::vector<BlasInt> p(n);
  for (int i = 0; i < n; ++i)
    p[i] = (pivoted ? std::min(i + 2, n - 1) : i) + 1;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      f(i, j) = i == j ? T(3) + T(i) / T(8) : T((i + 2 * j) % 7 - 3) / T(16);
      df(i, j) = T((2 * i + j) % 5 - 2) / T(32);
    }
  for (int j = 0; j < nrhs; ++j)
    for (int i = 0; i < n; ++i) {
      b(i, j) = T(i + j + 1) / T(8);
      db(i, j) = T((i + 2 * j) % 5 - 2) / T(16);
      g(i, j) = T((2 * i + j) % 7 - 3) / T(8);
    }

  const auto l = lower(f), u = upper(f);
  auto a = multiply(l, u);
  permute(a, p, true);
  auto da = add(multiply(lower(df, true), u), multiply(l, upper(df)), T(1));
  permute(da, p, true);
  auto x = solve(f, p, b, trans);
#ifdef COMPARE_LAPACK
  auto lapackX = b;
  int info;
  if constexpr (std::is_same_v<T, double>)
    info = LAPACKE_dgetrs(LAPACK_COL_MAJOR, operation, n, nrhs, f.data.data(),
                          f.ld, p.data(), lapackX.data.data(), lapackX.ld);
  else
    info = LAPACKE_sgetrs(LAPACK_COL_MAJOR, operation, n, nrhs, f.data.data(),
                          f.ld, p.data(), lapackX.data.data(), lapackX.ld);
  if (info != 0 ||
      maxDiff(x, lapackX) > T(128) * std::numeric_limits<T>::epsilon() ||
      !lapackX.paddingIntact()) {
    std::cerr << "LAPACK_REFERENCE_MISMATCH n=" << n << " rhs=" << nrhs
              << " transpose=" << trans << " info=" << info << '\n';
    return false;
  }
#endif
  auto dx = solve(f, p, add(db, multiply(trans ? transpose(da) : da, x), T(-1)),
                  trans);

  auto bbar = solve(f, p, g, !trans);
  auto abar =
      trans ? multiply(x, transpose(bbar)) : multiply(bbar, transpose(x));
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i)
      abar(i, j) = -abar(i, j);
  permute(abar, p, false);
  auto lbar = multiply(abar, transpose(u));
  auto ubar = multiply(transpose(l), abar);
  Matrix<T> fbar(n, n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i)
      fbar(i, j) = i > j ? lbar(i, j) : ubar(i, j);

  const T step = std::cbrt(std::numeric_limits<T>::epsilon());
  auto plus = solve(add(f, df, step), p, add(b, db, step), trans);
  auto minus = solve(add(f, df, -step), p, add(b, db, -step), trans);
  auto fd = add(plus, minus, T(-1));
  for (int j = 0; j < nrhs; ++j)
    for (int i = 0; i < n; ++i)
      fd(i, j) /= T(2) * step;
  const T eps = std::numeric_limits<T>::epsilon();
  const T fdTol = T(64) * step * step;
  const T residual = maxDiff(multiply(trans ? transpose(a) : a, x), b);
  const T fdError = maxDiff(fd, dx);
  const T adjError = std::abs(dot(g, dx) - dot(fbar, df) - dot(bbar, db));
#ifdef DIFFERENTIATE_ENZYME
  auto nativeX = b, nativeDX = db, nativeFactors = f, nativeDF = df;
  auto nativePivots = p;
  __enzyme_fwddiff((void *)getrs_primitive<T>, enzyme_const, operation,
                   enzyme_const, n, enzyme_const, nrhs,
                   nativeFactors.data.data(), nativeDF.data.data(),
                   enzyme_const, f.ld, enzyme_const, nativePivots.data(),
                   nativeX.data.data(), nativeDX.data.data(), enzyme_const,
                   b.ld, enzyme_const, clobber);
  auto reverseX = b, nativeBbar = g, reverseFactors = f;
  auto reversePivots = p;
  Matrix<T> nativeFbar(n, n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i)
      nativeFbar(i, j) = T(7) / T(64);
  __enzyme_autodiff((void *)getrs_primitive<T>, enzyme_const, operation,
                    enzyme_const, n, enzyme_const, nrhs,
                    reverseFactors.data.data(), nativeFbar.data.data(),
                    enzyme_const, f.ld, enzyme_const, reversePivots.data(),
                    reverseX.data.data(), nativeBbar.data.data(), enzyme_const,
                    b.ld, enzyme_const, clobber);
  auto expectedFbar = fbar;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i)
      expectedFbar(i, j) += clobber ? T(0) : T(7) / T(64);
  auto constX = b, constDX = db, constReverseX = b, constBbar = g;
  __enzyme_fwddiff((void *)getrs_primitive<T>, enzyme_const, operation,
                   enzyme_const, n, enzyme_const, nrhs, enzyme_const,
                   f.data.data(), enzyme_const, f.ld, enzyme_const, p.data(),
                   constX.data.data(), constDX.data.data(), enzyme_const, b.ld,
                   enzyme_const, false);
  __enzyme_autodiff((void *)getrs_primitive<T>, enzyme_const, operation,
                    enzyme_const, n, enzyme_const, nrhs, enzyme_const,
                    f.data.data(), enzyme_const, f.ld, enzyme_const, p.data(),
                    constReverseX.data.data(), constBbar.data.data(),
                    enzyme_const, b.ld, enzyme_const, false);
  auto constExpectedDX = solve(f, p, db, trans);
  auto vectorX = b, vectorD0 = db, vectorD1 = db;
  auto vectorF = f, vectorDF0 = df, vectorDF1 = df;
  auto vectorP = p;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i)
      vectorDF1(i, j) *= T(-0.5);
  for (int j = 0; j < nrhs; ++j)
    for (int i = 0; i < n; ++i)
      vectorD1(i, j) *= T(-0.5);
  __enzyme_fwddiff(
      (void *)getrs_primitive<T>, enzyme_width, 2, enzyme_const, operation,
      enzyme_const, n, enzyme_const, nrhs, vectorF.data.data(),
      vectorDF0.data.data(), vectorDF1.data.data(), enzyme_const, f.ld,
      enzyme_const, vectorP.data(), vectorX.data.data(), vectorD0.data.data(),
      vectorD1.data.data(), enzyme_const, b.ld, enzyme_const, clobber);
  auto vectorReverseX = b, vectorG0 = g, vectorG1 = g, vectorReverseF = f;
  auto vectorReverseP = p;
  Matrix<T> vectorFbar0(n, n), vectorFbar1(n, n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      vectorFbar0(i, j) = T(7) / T(64);
      vectorFbar1(i, j) = T(5) / T(64);
    }
  for (int j = 0; j < nrhs; ++j)
    for (int i = 0; i < n; ++i)
      vectorG1(i, j) *= T(-0.5);
  __enzyme_autodiff((void *)getrs_primitive<T>, enzyme_width, 2, enzyme_const,
                    operation, enzyme_const, n, enzyme_const, nrhs,
                    vectorReverseF.data.data(), vectorFbar0.data.data(),
                    vectorFbar1.data.data(), enzyme_const, f.ld, enzyme_const,
                    vectorReverseP.data(), vectorReverseX.data.data(),
                    vectorG0.data.data(), vectorG1.data.data(), enzyme_const,
                    b.ld, enzyme_const, clobber);
  auto halfDX = dx, halfBbar = bbar, halfFbar = fbar;
  for (int j = 0; j < nrhs; ++j)
    for (int i = 0; i < n; ++i) {
      halfDX(i, j) *= T(-0.5);
      halfBbar(i, j) *= T(-0.5);
    }
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i)
      halfFbar(i, j) = T(-0.5) * fbar(i, j) + (clobber ? T(0) : T(5) / T(64));
  const T nativeTolerance = T(256) * std::numeric_limits<T>::epsilon();
  if (maxDiff(nativeX, x) > nativeTolerance ||
      maxDiff(nativeDX, dx) > nativeTolerance ||
      maxDiff(reverseX, x) > nativeTolerance ||
      maxDiff(nativeBbar, bbar) > nativeTolerance ||
      maxDiff(nativeFbar, expectedFbar) > nativeTolerance ||
      maxDiff(constX, x) > nativeTolerance ||
      maxDiff(constDX, constExpectedDX) > nativeTolerance ||
      maxDiff(constReverseX, x) > nativeTolerance ||
      maxDiff(constBbar, bbar) > nativeTolerance ||
      maxDiff(vectorX, x) > nativeTolerance ||
      maxDiff(vectorD0, dx) > nativeTolerance ||
      maxDiff(vectorD1, halfDX) > nativeTolerance ||
      maxDiff(vectorReverseX, x) > nativeTolerance ||
      maxDiff(vectorG0, bbar) > nativeTolerance ||
      maxDiff(vectorG1, halfBbar) > nativeTolerance ||
      maxDiff(vectorFbar0, expectedFbar) > nativeTolerance ||
      maxDiff(vectorFbar1, halfFbar) > nativeTolerance ||
      !nativeX.paddingIntact() || !nativeDX.paddingIntact() ||
      !nativeBbar.paddingIntact() || !nativeFbar.paddingIntact() ||
      !vectorD0.paddingIntact() || !vectorD1.paddingIntact() ||
      !vectorG0.paddingIntact() || !vectorG1.paddingIntact() ||
      !vectorFbar0.paddingIntact() || !vectorFbar1.paddingIntact()) {
    std::cerr << "ENZYME_GETRS_MISMATCH n=" << n << " rhs=" << nrhs
              << " trans=" << trans << " pivoted=" << pivoted
              << " clobber=" << clobber << " JVP=" << maxDiff(nativeDX, dx)
              << " VJP_F=" << maxDiff(nativeFbar, expectedFbar)
              << " VJP_B=" << maxDiff(nativeBbar, bbar) << '\n';
    return false;
  }
#endif
  const bool pass = residual <= T(128) * eps && fdError <= fdTol &&
                    adjError <= T(128) * eps && f.paddingIntact() &&
                    b.paddingIntact() && x.paddingIntact() &&
                    dx.paddingIntact() && fbar.paddingIntact();
  std::cout << "type=" << (sizeof(T) == sizeof(float) ? "float" : "double")
            << " n=" << n << " rhs=" << nrhs << " trans=" << trans
            << " pivoted=" << pivoted << " clobber=" << clobber
            << " residual=" << residual << " fd_error=" << fdError
            << " adjoint_error=" << adjError << " pass=" << pass << '\n';
  return pass;
}

int main() {
  bool pass = true;
#if defined(DIFFERENTIATE_ENZYME) && !defined(COMPARE_LAPACK)
  // An illegal argument leaves the RHS unchanged. The derivative must be the
  // identity there, and must not allocate/free a tape or access invalid
  // extents.
  for (bool invalidSize : {false, true}) {
    double f = 3, df = 2, b = 4, db = 5;
    BlasInt pivot = 1;
    __enzyme_fwddiff((void *)getrs_primitive<double>, enzyme_const,
                     invalidSize ? 'N' : '?', enzyme_const,
                     invalidSize ? -1 : 1, enzyme_const, 1, &f, &df,
                     enzyme_const, 1, enzyme_const, &pivot, &b, &db,
                     enzyme_const, 1, enzyme_const, false);
    pass = pass && f == 3 && df == 2 && b == 4 && db == 5;
    __enzyme_autodiff((void *)getrs_primitive<double>, enzyme_const,
                      invalidSize ? 'N' : '?', enzyme_const,
                      invalidSize ? -1 : 1, enzyme_const, 1, &f, &df,
                      enzyme_const, 1, enzyme_const, &pivot, &b, &db,
                      enzyme_const, 1, enzyme_const, false);
    pass = pass && f == 3 && df == 2 && b == 4 && db == 5;
  }
#endif
  int cases = 0;
  for (int n : {0, 1, 2, 4, 7})
    for (int rhs : {0, 1, 3})
      for (char operation : {'N', 'n', 'T', 't', 'C', 'c'})
        for (bool pivoted : {false, true})
          for (bool clobber : {false, true}) {
            pass = check<float>(n, rhs, operation, pivoted, clobber) && pass;
            pass = check<double>(n, rhs, operation, pivoted, clobber) && pass;
            cases += 2;
          }
  std::cout << "CASES=" << cases << " GETRS_REFERENCE_PASS=" << pass << '\n';
  return pass ? 0 : 1;
}
