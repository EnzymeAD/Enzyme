// RUN: %clang -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -

// Second derivatives through a global: forward mode over reverse mode, each
// in a context of its own. The inner derivative accumulates into the shadow
// of g in its context; the outer one seeds and reads the shadow of g in its
// context, so the two never confuse their perturbations.

#include "../test_utils.h"

extern void *__enzyme_context(int);
extern void *__enzyme_shadow(void *, void *, int);
extern double __enzyme_fwddiff(double (*)(double), ...);
extern double __enzyme_autodiff(double (*)(double), ...);
extern int enzyme_context;

double g = 2.0;

double f(double x) { return g * x * x * x; }

// d f / d x = 3 g x^2
double df(double x) {
  void *inner = __enzyme_context(1);
  *(double *)__enzyme_shadow(inner, &g, 0) = 0.0;
  return __enzyme_autodiff(f, enzyme_context, inner, x);
}

int main() {
  void *outer = __enzyme_context(1);
  double *dg = (double *)__enzyme_shadow(outer, &g, 0);

  // d^2 f / d x^2 = 6 g x
  *dg = 0.0;
  double ddx = __enzyme_fwddiff(df, enzyme_context, outer, 2.0, 1.0);
  APPROX_EQ(ddx, 6 * 2.0 * 2.0, 1e-10);

  // d^2 f / d x d g = 3 x^2
  *dg = 1.0;
  double ddg = __enzyme_fwddiff(df, enzyme_context, outer, 2.0, 0.0);
  APPROX_EQ(ddg, 3 * 2.0 * 2.0, 1e-10);
  return 0;
}
