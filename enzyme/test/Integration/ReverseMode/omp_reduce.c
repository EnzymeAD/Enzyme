// RUN: %clang -fopenmp -std=c11 %O0TBAA -fno-vectorize -fno-unroll-loops %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %clang -fopenmp -x ir - -o %t && %t
// RUN: %clang -fopenmp -std=c11 -O1 -fno-vectorize -fno-unroll-loops %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %clang -fopenmp -x ir - -o %t && %t
// RUN: %clang -fopenmp -std=c11 -O2 -fno-vectorize -fno-unroll-loops %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %clang -fopenmp -x ir - -o %t && %t
// RUN: %clang -fopenmp -std=c11 -O3 -fno-vectorize -fno-unroll-loops %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %clang -fopenmp -x ir - -o %t && %t
// RUN: %clang -fopenmp -std=c11 %O0TBAA -fno-vectorize -fno-unroll-loops %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %clang -fopenmp -x ir - -o %t && %t
// RUN: %clang -fopenmp -std=c11 -O1 -fno-vectorize -fno-unroll-loops %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %clang -fopenmp -x ir - -o %t && %t
// RUN: %clang -fopenmp -std=c11 -O2 -fno-vectorize -fno-unroll-loops %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %clang -fopenmp -x ir - -o %t && %t
// RUN: %clang -fopenmp -std=c11 -O3 -fno-vectorize -fno-unroll-loops %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %clang -fopenmp -x ir - -o %t && %t

#include "../test_utils.h"

void omp_set_num_threads(int);

extern int enzyme_const;
extern int enzyme_dup;
void __enzyme_autodiff(void *, ...);

__attribute__((noinline))
void reduce_sum_double(const double *x, double *out, int n) {
  double s = 0.0;
#pragma omp parallel for reduction(+ : s)
  for (int i = 0; i < n; i++) {
    s += x[i] * x[i] + 3.0 * x[i];
  }
  *out += 2.0 * s;
}

__attribute__((noinline))
void reduce_sum_float(const float *x, float *out, int n) {
  float s = 0.0f;
#pragma omp parallel for reduction(+ : s)
  for (int i = 0; i < n; i++) {
    s += x[i] * x[i] * x[i];
  }
  *out = s;
}

__attribute__((noinline))
void reduce_max_double(const double *x, double *out, int n) {
  double m = *out;
#pragma omp parallel for reduction(max : m)
  for (int i = 0; i < n; i++) {
    double v = x[i] * x[i];
    if (v > m)
      m = v;
  }
  *out = m;
}

__attribute__((noinline))
void reduce_min_double(const double *x, double *out, int n) {
  double mn = *out;
#pragma omp parallel for reduction(min : mn)
  for (int i = 0; i < n; i++) {
    double d = x[i] - 3.5;
    double v = d * d;
    if (v < mn)
      mn = v;
  }
  *out = mn;
}

__attribute__((noinline))
void reduce_multi(const double *x, double *sum_out, double *max_out,
                  double *min_out, int n) {
  double s = 0.0;
  double mx = -1.0e30;
  double mn = 1.0e30;
#pragma omp parallel for reduction(+ : s) reduction(max : mx) reduction(min : mn)
  for (int i = 0; i < n; i++) {
    double v = x[i] * x[i] - 2.0 * x[i];
    s += v;
    if (v > mx)
      mx = v;
    if (v < mn)
      mn = v;
  }
  *sum_out = s;
  *max_out = mx;
  *min_out = mn;
}

__attribute__((noinline))
void reduce_blocking_and_sequential(const double *x, double *y, double *out,
                                    int n) {
  double s1 = 0.0;
  double mx = -1.0e30;
#pragma omp parallel
  {
    // Blocking reduction (no nowait) followed by all threads reading s1.
#pragma omp for reduction(+ : s1)
    for (int i = 0; i < n; i++) {
      s1 += x[i] * x[i];
    }
#pragma omp for
    for (int i = 0; i < n; i++) {
      y[i] = s1 * x[i];
    }
#pragma omp for reduction(max : mx)
    for (int i = 0; i < n; i++) {
      if (y[i] > mx)
        mx = y[i];
    }
  }
  *out = mx;
}

__attribute__((noinline))
void reduce_host_loop(const double *x, double *out, int n, int steps) {
  double total = 0.0;
  for (int st = 1; st <= steps; st++) {
    double s = 0.0;
#pragma omp parallel for reduction(+ : s)
    for (int i = 0; i < n; i++) {
      s += (double)st * x[i] * x[i];
    }
    total += s;
  }
  *out = total;
}

static void run_suite(int nthreads) {
  omp_set_num_threads(nthreads);
  const int n = 64;
  double x[64], dx[64];

  // 1. Double sum reduction (including seed propagation through += on *out).
  for (int i = 0; i < n; i++) {
    x[i] = 0.25 * (i + 1) - 4.0;
    dx[i] = 0.0;
  }
  double out = 5.0, dout = 1.5;
  __enzyme_autodiff((void *)reduce_sum_double, enzyme_dup, x, dx, enzyme_dup,
                    &out, &dout, enzyme_const, n);
  APPROX_EQ(dout, 1.5, 1e-10);
  for (int i = 0; i < n; i++) {
    double expected = 1.5 * 2.0 * (2.0 * x[i] + 3.0);
    APPROX_EQ(dx[i], expected, 1e-9);
  }

  // 2. Float sum reduction.
  float fx[64], dfx[64];
  for (int i = 0; i < n; i++) {
    fx[i] = 0.1f * (float)(i + 1);
    dfx[i] = 0.0f;
  }
  float fout = 0.0f, dfout = 2.0f;
  __enzyme_autodiff((void *)reduce_sum_float, enzyme_dup, fx, dfx, enzyme_dup,
                    &fout, &dfout, enzyme_const, n);
  for (int i = 0; i < n; i++) {
    float expected = 2.0f * 3.0f * fx[i] * fx[i];
    APPROX_EQ(dfx[i], expected, 1e-4f);
  }

  // 3a. Max reduction: winner on a non-zero thread (index 43 -> x^2 = 81).
  for (int i = 0; i < n; i++) {
    x[i] = 0.05 * (i + 1);
    dx[i] = 0.0;
  }
  x[43] = -9.0;
  out = 10.0;
  dout = 2.5;
  __enzyme_autodiff((void *)reduce_max_double, enzyme_dup, x, dx, enzyme_dup,
                    &out, &dout, enzyme_const, n);
  APPROX_EQ(out, 81.0, 1e-10);
  APPROX_EQ(dout, 0.0, 1e-10);
  for (int i = 0; i < n; i++) {
    double expected = (i == 43) ? (2.5 * 2.0 * (-9.0)) : 0.0;
    APPROX_EQ(dx[i], expected, 1e-10);
  }

  // 3b. Max reduction: initial shared *out beats all threads.
  for (int i = 0; i < n; i++) {
    x[i] = 0.05 * (i + 1);
    dx[i] = 0.0;
  }
  out = 100.0;
  dout = 3.25;
  __enzyme_autodiff((void *)reduce_max_double, enzyme_dup, x, dx, enzyme_dup,
                    &out, &dout, enzyme_const, n);
  APPROX_EQ(out, 100.0, 1e-10);
  APPROX_EQ(dout, 3.25, 1e-10);
  for (int i = 0; i < n; i++) {
    APPROX_EQ(dx[i], 0.0, 1e-10);
  }

  // 3c. Max reduction: winner on thread 0 (index 2 -> x^2 = 49).
  for (int i = 0; i < n; i++) {
    x[i] = 0.05 * (i + 1);
    dx[i] = 0.0;
  }
  x[2] = 7.0;
  out = -1.0;
  dout = 1.25;
  __enzyme_autodiff((void *)reduce_max_double, enzyme_dup, x, dx, enzyme_dup,
                    &out, &dout, enzyme_const, n);
  APPROX_EQ(out, 49.0, 1e-10);
  APPROX_EQ(dout, 0.0, 1e-10);
  for (int i = 0; i < n; i++) {
    double expected = (i == 2) ? (1.25 * 14.0) : 0.0;
    APPROX_EQ(dx[i], expected, 1e-10);
  }

  // 4. Min reduction: winner on index 51 (x[51] = 3.25 -> (x - 3.5)^2 = 0.0625).
  for (int i = 0; i < n; i++) {
    x[i] = -2.0 - 0.1 * i;
    dx[i] = 0.0;
  }
  x[51] = 3.25;
  out = 1000.0;
  dout = 4.0;
  __enzyme_autodiff((void *)reduce_min_double, enzyme_dup, x, dx, enzyme_dup,
                    &out, &dout, enzyme_const, n);
  APPROX_EQ(out, 0.0625, 1e-10);
  APPROX_EQ(dout, 0.0, 1e-10);
  for (int i = 0; i < n; i++) {
    double expected = (i == 51) ? (4.0 * 2.0 * (3.25 - 3.5)) : 0.0;
    APPROX_EQ(dx[i], expected, 1e-10);
  }

  // 5. Combined multi-variable reduction (+, max, min in one parallel loop).
  for (int i = 0; i < n; i++) {
    x[i] = 2.0 + 0.03 * (i + 1); // v_i = x^2 - 2x strictly increasing on x > 1
    dx[i] = 0.0;
  }
  x[19] = 10.0; // unique max: v = 80 at i = 19
  x[47] = 1.1;  // unique min: v = -0.99 at i = 47
  double sum_o = 0.0, dsum_o = 0.5;
  double max_o = 0.0, dmax_o = 2.0;
  double min_o = 0.0, dmin_o = -3.0;
  __enzyme_autodiff((void *)reduce_multi, enzyme_dup, x, dx, enzyme_dup,
                    &sum_o, &dsum_o, enzyme_dup, &max_o, &dmax_o, enzyme_dup,
                    &min_o, &dmin_o, enzyme_const, n);
  for (int i = 0; i < n; i++) {
    double dv = 2.0 * x[i] - 2.0;
    double coeff = 0.5;
    if (i == 19)
      coeff += 2.0;
    if (i == 47)
      coeff += -3.0;
    APPROX_EQ(dx[i], coeff * dv, 1e-9);
  }

  // 6. Blocking reduction followed by dependent parallel loop + max reduction.
  double y[64], dy[64];
  double sum_sq = 0.0;
  for (int i = 0; i < n; i++) {
    x[i] = 0.1 * (i + 1);
    if (i == 37)
      x[i] = 8.0; // unique max positive x[i], since s1 > 0
    sum_sq += x[i] * x[i];
    dx[i] = 0.0;
    y[i] = 0.0;
    dy[i] = 0.0;
  }
  out = 0.0;
  dout = 1.0;
  __enzyme_autodiff((void *)reduce_blocking_and_sequential, enzyme_dup, x, dx,
                    enzyme_dup, y, dy, enzyme_dup, &out, &dout, enzyme_const,
                    n);
  // out = s1 * x[37], where s1 = sum_j x[j]^2.
  // d(out)/dx[i] = x[37] * 2 * x[i] + (i == 37 ? s1 : 0).
  for (int i = 0; i < n; i++) {
    double expected = 8.0 * 2.0 * x[i] + ((i == 37) ? sum_sq : 0.0);
    APPROX_EQ(dx[i], expected, 1e-8);
  }

  // 7. Parallel reduction inside a host loop.
  for (int i = 0; i < n; i++) {
    x[i] = 0.2 * (i + 1);
    dx[i] = 0.0;
  }
  out = 0.0;
  dout = 1.0;
  __enzyme_autodiff((void *)reduce_host_loop, enzyme_dup, x, dx, enzyme_dup,
                    &out, &dout, enzyme_const, n, enzyme_const, 3);
  // sum_{st=1..3} st = 6 -> d(out)/dx[i] = 6 * 2 * x[i] = 12 * x[i].
  for (int i = 0; i < n; i++) {
    APPROX_EQ(dx[i], 12.0 * x[i], 1e-9);
  }
}

int main(void) {
  // 1 thread (empty_reduce_block), 3 threads (critical_reduce_block),
  // and 8 threads (tree_reduce_block in libomp).
  run_suite(1);
  run_suite(3);
  run_suite(8);
  return 0;
}
