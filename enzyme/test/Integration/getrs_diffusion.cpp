// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// clang-format off
// RUN: %clang++ -std=c++2b -O1 -DGETRS_TEST_BACKEND_IMPLEMENTATION -x c++ -c %S/getrs_test_backend.h -o %t.backend.o
// RUN: if [ %llvmver -lt 17 ]; then flags="-Xclang -no-opaque-pointers"; else flags=""; fi; %clang++ -std=c++2b -O1 -fno-exceptions $flags -S -emit-llvm %s -o %t.input.ll && %opt %loadEnzyme %enzyme -S %t.input.ll -o %t.ad.ll && %clang++ -O2 %t.ad.ll %t.backend.o -o %t.exe && %t.exe
// RUN: if [ %llvmver -lt 17 ]; then flags="-Xclang -no-opaque-pointers"; else flags=""; fi; %clang++ -std=c++2b -O2 -fno-exceptions $flags -S -emit-llvm %s -o %t.input.ll && %opt %loadEnzyme %enzyme -S %t.input.ll -o %t.ad.ll && %clang++ -O2 %t.ad.ll %t.backend.o -o %t.exe && %t.exe
// clang-format on

// C++23 scientific consumer: three implicit diffusion steps reuse one LU.
// Factorization stays outside AD. Differentiate the source amplitude and the
// supplied packed LU factors, including accumulation across repeated solves.
#ifdef COMPARE_LAPACK
#include <lapacke.h>
#else
#include "getrs_test_backend.h"
#endif
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>

int enzyme_const;
extern "C" double __enzyme_autodiff(void *, ...);
extern "C" double __enzyme_fwddiff(void *, ...);
constexpr int N = 4;

__attribute__((noinline)) void simulate(double amplitude, const double *lu,
                                        const int *pivots, double *state) {
  for (int i = 0; i < N; ++i)
    state[i] = 0;
  for (int step = 0; step < 3; ++step) {
    for (int i = 0; i < N; ++i)
      state[i] += amplitude * (i + 1) * (step + 1) / 8.0;
    const char trans = 'N';
    const int n = N, one = 1;
    int info = 0;
    LAPACK_dgetrs(&trans, &n, &one, lu, &n, pivots, state, &n, &info);
  }
}

__attribute__((noinline)) double loss(double amplitude, const double *lu,
                                      const int *pivots) {
  double state[N];
  simulate(amplitude, lu, pivots, state);
  const double observations[N] = {0.2, -0.1, 0.3, 0.05};
  double value = 0;
  for (int i = 0; i < N; ++i) {
    double residual = state[i] - observations[i];
    value += 0.5 * residual * residual;
  }
  return value;
}

int main() {
  double matrix[N * N] = {};
  for (int i = 0; i < N; ++i) {
    matrix[i + N * i] = 3;
    if (i + 1 < N)
      matrix[i + 1 + N * i] = matrix[i + N * (i + 1)] = -1;
  }
  double factors[N * N];
  std::copy(matrix, matrix + N * N, factors);
  int pivots[N];
  int info = 0;
#ifdef COMPARE_LAPACK
  info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, N, N, factors, N, pivots);
  if (info)
    return 2;
#else
  // The strictly diagonally dominant diffusion matrix needs no row swaps.
  // This preparation is outside AD; GETRF differentiation is not being tested.
  for (int j = 0; j < N; ++j) {
    pivots[j] = j + 1;
    for (int i = j + 1; i < N; ++i) {
      factors[i + N * j] /= factors[j + N * j];
      for (int k = j + 1; k < N; ++k)
        factors[i + N * k] -= factors[i + N * j] * factors[j + N * k];
    }
  }
#endif
  const double amplitude = 0.7;
  double response[N];
  simulate(1, factors, pivots, response);
  const double observations[N] = {0.2, -0.1, 0.3, 0.05};
  double analytic = 0;
  for (int i = 0; i < N; ++i)
    analytic += (amplitude * response[i] - observations[i]) * response[i];
  double reverse = __enzyme_autodiff((void *)loss, amplitude, enzyme_const,
                                     factors, enzyme_const, pivots);
  double forward = __enzyme_fwddiff((void *)loss, amplitude, 1.0, enzyme_const,
                                    factors, enzyme_const, pivots);
  double factorBar[N * N] = {}, factorDirection[N * N];
  for (int i = 0; i < N * N; ++i)
    factorDirection[i] = (i % 5 - 2) / 64.0;
  double activeReverse = __enzyme_autodiff((void *)loss, amplitude, factors,
                                           factorBar, enzyme_const, pivots);
  double activeForward =
      __enzyme_fwddiff((void *)loss, amplitude, 1.0, factors, factorDirection,
                       enzyme_const, pivots);
  double expectedDirection = analytic;
  for (int i = 0; i < N * N; ++i)
    expectedDirection += factorBar[i] * factorDirection[i];
  const double h = 1e-5;
  double finite = (loss(amplitude + h, factors, pivots) -
                   loss(amplitude - h, factors, pivots)) /
                  (2 * h);
  double maxFactorError = 0;
  for (int i = 0; i < N * N; ++i) {
    double plus[N * N], minus[N * N];
    std::copy(factors, factors + N * N, plus);
    std::copy(factors, factors + N * N, minus);
    plus[i] += h;
    minus[i] -= h;
    double fd =
        (loss(amplitude, plus, pivots) - loss(amplitude, minus, pivots)) /
        (2 * h);
    maxFactorError = std::max(maxFactorError, std::abs(fd - factorBar[i]));
  }

  double state[N] = {}, maxResidual = 0;
  for (int step = 0; step < 3; ++step) {
    double rhs[N];
    for (int i = 0; i < N; ++i)
      rhs[i] = state[i] + amplitude * (i + 1) * (step + 1) / 8.0;
    std::copy(rhs, rhs + N, state);
    const char trans = 'N';
    const int rows = N, one = 1;
    LAPACK_dgetrs(&trans, &rows, &one, factors, &rows, pivots, state, &rows,
                  &info);
    if (info)
      return 3;
    for (int i = 0; i < N; ++i) {
      double product = 0;
      for (int j = 0; j < N; ++j)
        product += matrix[i + N * j] * state[j];
      maxResidual = std::max(maxResidual, std::abs(product - rhs[i]));
    }
  }
  bool pass = std::abs(reverse - analytic) < 1e-10 &&
              std::abs(forward - analytic) < 1e-10 &&
              std::abs(finite - analytic) < 1e-8 && maxResidual < 1e-12 &&
              std::abs(activeReverse - analytic) < 1e-10 &&
              std::abs(activeForward - expectedDirection) < 1e-10 &&
              maxFactorError < 1e-8;
  std::cout << std::setprecision(17)
            << "loss=" << loss(amplitude, factors, pivots) << '\n'
            << "reverse=" << reverse << " forward=" << forward
            << " analytic=" << analytic << " finite_difference=" << finite
            << '\n'
            << "max_primal_residual=" << maxResidual << '\n'
            << "max_factor_adjoint_error=" << maxFactorError
            << " active_forward=" << activeForward
            << " expected_direction=" << expectedDirection << '\n'
            << "IMPLICIT_DIFFUSION_PASS=" << pass << '\n';
  return pass ? 0 : 1;
}
