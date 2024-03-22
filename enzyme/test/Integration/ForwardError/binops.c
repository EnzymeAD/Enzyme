// RUN: %clang -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S |
// %lli - RUN: %clang -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme
// -S | %lli - RUN: %clang -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme
// %enzyme -S | %lli - RUN: %clang -O3 %s -S -emit-llvm -o - | %opt -
// %OPloadEnzyme %enzyme -S | %lli -

#include "../test_utils.h"

double cos(double);

extern double __enzyme_error_estimate(void *, ...);

/*
double fun(double x1, double x2)
{
    double v2 = x2;
    double v3 = 333 * v2;
    double v5 = v3 + v2;
    return v5;
}
*/

double fun(double x1, double x2) { return cos(x1); }

int main() {
  double error = __enzyme_error_estimate((void *)fun, 1e-7, 0.0, 2.7, 0.0);
  printf("Found floating point error of %e\n", error);
  APPROX_EQ(error, 1.110223e-16, 1e-18);
}
