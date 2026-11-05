// RUN: %clang -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -O1 -g %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -

// Batching a function that itself calls a defined function: here the callee
// is the forward-mode derivative Enzyme generates inside the batched
// function. Every vector operand of the inner call must reach the batched
// callee lane by lane.

#include "../test_utils.h"
#include <stdio.h>

struct Batch2 {
  double a, b;
};

extern Batch2 __enzyme_batch(...);
extern double __enzyme_fwddiff(void *, ...);
extern int enzyme_width;
extern int enzyme_vector;
extern int enzyme_scalar;

double f(double g, double x) { return g * x * x * x; }

// df/dx = 3 g x^2
double dfdx(double g, double x) {
  return __enzyme_fwddiff((void *)f, g, 0.0, x, 1.0);
}

int main() {
  const double g = 2.0, x0 = 1.0, x1 = 3.0;

  Batch2 p = __enzyme_batch((void *)f, enzyme_width, 2, enzyme_scalar, g,
                            enzyme_vector, x0, x1);
  Batch2 dv = __enzyme_batch((void *)dfdx, enzyme_width, 2, enzyme_vector, g, g,
                             enzyme_vector, x0, x1);
  Batch2 ds = __enzyme_batch((void *)dfdx, enzyme_width, 2, enzyme_scalar, g,
                             enzyme_vector, x0, x1);

  printf("primal (%g, %g) all-vector (%g, %g) scalar-g (%g, %g)\n", p.a, p.b,
         dv.a, dv.b, ds.a, ds.b);

  APPROX_EQ(p.a, g * x0 * x0 * x0, 1e-10);
  APPROX_EQ(p.b, g * x1 * x1 * x1, 1e-10);
  APPROX_EQ(dv.a, 3 * g * x0 * x0, 1e-10);
  APPROX_EQ(dv.b, 3 * g * x1 * x1, 1e-10);
  APPROX_EQ(ds.a, 3 * g * x0 * x0, 1e-10);
  APPROX_EQ(ds.b, 3 * g * x1 * x1, 1e-10);
  return 0;
}
