// No -ffast-math: nsz would make the sign of zero meaningless.
// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -

// fabs(-x*x) == x*x, so its second derivative at 0 is 2. At x = 0 the inner
// value is -0.0, so this needs fabs'(-0.0) == -1: the sign of the zero records
// the side from which the argument approaches 0.

#include "../test_utils.h"

extern double __enzyme_fwddiff(void *, ...);
extern double __enzyme_autodiff(void *, ...);

double absf(double x) { return __builtin_fabs(x); }
double f(double x) { return __builtin_fabs(-x * x); }

double df(double x) { return __enzyme_fwddiff((void *)f, x, 1.0); }
double gf(double x) { return __enzyme_autodiff((void *)f, x); }

int main() {
  // First derivative of fabs at signed zeros and away from zero.
  TEST_EQ(__enzyme_fwddiff((void *)absf, 0.0, 1.0), 1.0);
  TEST_EQ(__enzyme_fwddiff((void *)absf, -0.0, 1.0), -1.0);
  TEST_EQ(__enzyme_fwddiff((void *)absf, 2.0, 1.0), 1.0);
  TEST_EQ(__enzyme_fwddiff((void *)absf, -2.0, 1.0), -1.0);
  TEST_EQ(__enzyme_autodiff((void *)absf, 0.0), 1.0);
  TEST_EQ(__enzyme_autodiff((void *)absf, -0.0), -1.0);

  // First derivative of fabs(-x*x) at 0 is 0.
  APPROX_EQ(df(0.0), 0.0, 1e-10);
  APPROX_EQ(gf(0.0), 0.0, 1e-10);

  // Second derivative of fabs(-x*x) at 0 is 2.
  APPROX_EQ(__enzyme_fwddiff((void *)df, 0.0, 1.0), 2.0, 1e-10);
  APPROX_EQ(__enzyme_autodiff((void *)df, 0.0), 2.0, 1e-10);
  APPROX_EQ(__enzyme_fwddiff((void *)gf, 0.0, 1.0), 2.0, 1e-10);
  APPROX_EQ(__enzyme_autodiff((void *)gf, 0.0), 2.0, 1e-10);

  // And away from zero.
  APPROX_EQ(__enzyme_fwddiff((void *)df, 1.5, 1.0), 2.0, 1e-10);
  APPROX_EQ(__enzyme_fwddiff((void *)df, -1.5, 1.0), 2.0, 1e-10);
  return 0;
}
