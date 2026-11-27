// clang-format off
// RUN: %clang -fopenmp -std=c11 -fno-vectorize -fno-unroll-loops -O0 -Xclang -disable-O0-optnone %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %clang -fopenmp -x ir - -o %s.out && %s.out
// RUN: %clang -fopenmp -std=c11 -fno-vectorize -fno-unroll-loops -g -O0 -Xclang -disable-O0-optnone %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %clang -fopenmp -x ir - -o %s.out && %s.out
// RUN: %clang -fopenmp -std=c11 -fno-vectorize -fno-unroll-loops %O0TBAA %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %clang -fopenmp -x ir - -o %s.out && %s.out
// RUN: %clang -fopenmp -std=c11 -fno-vectorize -fno-unroll-loops -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %clang -fopenmp -x ir - -o %s.out && %s.out
// RUN: %clang -fopenmp -std=c11 -fno-vectorize -fno-unroll-loops -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %clang -fopenmp -x ir - -o %s.out && %s.out
// RUN: %clang -fopenmp -std=c11 -fno-vectorize -fno-unroll-loops -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %clang -fopenmp -x ir - -o %s.out && %s.out
// RUN: %clang -fopenmp -std=c11 -fno-vectorize -fno-unroll-loops -O0 -Xclang -disable-O0-optnone %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %clang -fopenmp -x ir - -o %s.out && %s.out
// RUN: %clang -fopenmp -std=c11 -fno-vectorize -fno-unroll-loops -g -O0 -Xclang -disable-O0-optnone %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %clang -fopenmp -x ir - -o %s.out && %s.out
// RUN: %clang -fopenmp -std=c11 -fno-vectorize -fno-unroll-loops %O0TBAA %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %clang -fopenmp -x ir - -o %s.out && %s.out
// RUN: %clang -fopenmp -std=c11 -fno-vectorize -fno-unroll-loops -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %clang -fopenmp -x ir - -o %s.out && %s.out
// RUN: %clang -fopenmp -std=c11 -fno-vectorize -fno-unroll-loops -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %clang -fopenmp -x ir - -o %s.out && %s.out
// RUN: %clang -fopenmp -std=c11 -fno-vectorize -fno-unroll-loops -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %clang -fopenmp -x ir - -o %s.out && %s.out
// clang-format on

#include <math.h>
#include <stddef.h>
#include <stdio.h>

#include "../test_utils.h"

extern int enzyme_dup;
extern int enzyme_dupnoneed;
extern int enzyme_const;
double __enzyme_autodiff(void *, ...);

double square_norm(double *vec, size_t count) {
  double sum = 0.0;
#pragma omp parallel for reduction(+ : sum)
  for (size_t i = 0; i < count; i++) {
    sum += vec[i] * vec[i];
  }
  return sum;
}

double firstprivate_reduction(double *arr, int count) {
  double sum = 0.0;
#pragma omp parallel for firstprivate(count) reduction(+ : sum)
  for (int i = 0; i < count; i++) {
    sum += arr[i] * arr[i] + 3.0 * arr[i];
  }
  return sum;
}

double multi_plus_minus(double *arr, int count, double init_sum,
                        double init_diff) {
  double sum = init_sum;
  double diff = init_diff;
#pragma omp parallel for reduction(+ : sum) reduction(- : diff)
  for (int i = 0; i < count; i++) {
    sum += arr[i] * arr[i];
    diff -= 2.0 * arr[i];
  }
  return 3.0 * sum + 5.0 * diff;
}

double blocking_two_loops(double *x, double *y, int count) {
  double sum = 1.5;
  double total = 0.0;
#pragma omp parallel
  {
#pragma omp for reduction(+ : sum)
    for (int i = 0; i < count; i++) {
      sum += x[i] * x[i];
    }
#pragma omp for reduction(+ : total)
    for (int i = 0; i < count; i++) {
      total += sum * y[i];
    }
  }
  return total;
}

int main(void) {
  {
    double vec[16], d_vec[16];
    for (int i = 0; i < 16; i++) {
      vec[i] = i + 1.0;
      d_vec[i] = 0.0;
    }
    __enzyme_autodiff((void *)square_norm, enzyme_dupnoneed, vec, d_vec,
                      enzyme_const, (size_t)16);
    for (int i = 0; i < 16; i++) {
      APPROX_EQ(d_vec[i], 2.0 * (i + 1.0), 1e-9);
    }
  }

  {
    double arr[17], d_arr[17];
    for (int i = 0; i < 17; i++) {
      arr[i] = 0.5 * i - 2.0;
      d_arr[i] = 0.0;
    }
    __enzyme_autodiff((void *)firstprivate_reduction, enzyme_dup, arr, d_arr,
                      enzyme_const, 17);
    for (int i = 0; i < 17; i++) {
      APPROX_EQ(d_arr[i], 2.0 * arr[i] + 3.0, 1e-9);
    }
  }

  {
    double arr[23], d_arr[23], scratch[23];
    for (int i = 0; i < 23; i++) {
      arr[i] = 1.25 * (i + 1);
      d_arr[i] = 0.0;
      scratch[i] = 0.0;
    }
    __enzyme_autodiff((void *)multi_plus_minus, enzyme_dup, arr, d_arr,
                      enzyme_const, 23, enzyme_const, 7.0, enzyme_const, -4.0);
    for (int i = 0; i < 23; i++) {
      APPROX_EQ(d_arr[i], 6.0 * arr[i] - 10.0, 1e-9);
    }
    double d_init_sum =
        __enzyme_autodiff((void *)multi_plus_minus, enzyme_dup, arr, scratch,
                          enzyme_const, 23, 7.0, enzyme_const, -4.0);
    for (int i = 0; i < 23; i++) {
      scratch[i] = 0.0;
    }
    double d_init_diff =
        __enzyme_autodiff((void *)multi_plus_minus, enzyme_dup, arr, scratch,
                          enzyme_const, 23, enzyme_const, 7.0, -4.0);
    APPROX_EQ(d_init_sum, 3.0, 1e-9);
    APPROX_EQ(d_init_diff, 5.0, 1e-9);
  }

  {
    double x[12], d_x[12], y[12], d_y[12];
    double expected_sum = 1.5, y_total = 0.0;
    for (int i = 0; i < 12; i++) {
      x[i] = 0.25 * (i + 1);
      d_x[i] = 0.0;
      y[i] = 1.5 - 0.1 * i;
      d_y[i] = 0.0;
      expected_sum += x[i] * x[i];
      y_total += y[i];
    }
    __enzyme_autodiff((void *)blocking_two_loops, enzyme_dup, x, d_x,
                      enzyme_dup, y, d_y, enzyme_const, 12);
    for (int i = 0; i < 12; i++) {
      APPROX_EQ(d_y[i], expected_sum, 1e-9);
      APPROX_EQ(d_x[i], 2.0 * x[i] * y_total, 1e-9);
    }
  }

  return 0;
}
