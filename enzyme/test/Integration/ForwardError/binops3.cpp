// RUN: %clang++ -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang++ -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang++ -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang++ -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -

#include "../test_utils.h"

#include <cmath>

extern double __enzyme_error_estimate(void *, ...);

// An example from https://dl.acm.org/doi/10.1145/3371128
double fun(double x) {
  double v1 = cos(x);
  double v2 = 1 - v1;
  double v3 = x * x;
  double v4 = v2 / v3;

  printf("v1 = %.18e, v2 = %.18e, v3 = %.18e, v4 = %.18e\n", v1, v2, v3, v4);

  return v4;
}

int main() {
  double res = fun(1e-7);
  __enzyme_error_estimate((void *)fun, 2e-7, 0.0);
  __enzyme_error_estimate((void *)fun, 7e-7, 0.0);
  double error = __enzyme_error_estimate((void *)fun, 1e-7, 0.0);
  printf("res = %.18e, abs error = %.18e, rel error = %.18e\n", res, error,
         fabs(error / res));
  APPROX_EQ(error, 2.2222222222e-2, 1e-4);
}
