// RUN: %clang++ -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %fpopt -enzyme-print-herbie -enzyme-print-fpopt -fpopt-target-func-regex=fun -S | %lli -
// RUN: %clang++ -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %fpopt -enzyme-print-herbie -enzyme-print-fpopt -fpopt-target-func-regex=fun -S | %lli -
// RUN: %clang++ -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %fpopt -enzyme-print-herbie -enzyme-print-fpopt -fpopt-target-func-regex=fun -S | %lli -
// RUN: %clang++ -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %fpopt -enzyme-print-herbie -enzyme-print-fpopt -fpopt-target-func-regex=fun -S | %lli -

#include "../test_utils.h"

#include <cmath>

double fun(double a, double b, double c) {
  double discriminant = b * b - 4 * a * c;
  printf("discriminant = %.18e\n", discriminant);
  double sqrt_discriminant = sqrt(discriminant);
  printf("sqrt_discriminant = %.18e\n", sqrt_discriminant);
  double numerator = -b - sqrt_discriminant;
  printf("numerator = %.18e\n", numerator);
  double result = numerator / (2 * a);
  return result;
}

int main() {
  // x^2 - 3x + 2 = 0 --> x1 = 1 (computed), x2 = 2
  double res1 = fun(1, -3, 2);
  printf("res1 = %.18e\n", res1);
  APPROX_EQ(res1, 1.0, 1e-4);

  // x^2 - 5x + 6 = 0 --> x1 = 2 (computed), x2 = 3
  double res2 = fun(1, -5, 6);
  printf("res2 = %.18e\n", res2);
  APPROX_EQ(res2, 2.0, 1e-4);

  return 0;
}
