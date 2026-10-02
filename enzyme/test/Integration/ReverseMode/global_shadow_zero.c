// RUN: %clang -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -

// __enzyme_zero_shadows zeroes every shadow in a context, including ones the
// program never queries, so gradients in a loop do not accumulate.

#include "../test_utils.h"

extern void *__enzyme_context(int);
extern void *__enzyme_shadow(void *, void *, int);
extern void __enzyme_zero_shadows(void *);
extern double __enzyme_autodiff(double (*)(double), ...);
extern int enzyme_context;

double scale = 2.0;
double weights[2] = {3.0, 5.0};
double hidden = 7.0;

double f(double x) {
  return scale * (weights[0] + weights[1]) * x * x + hidden * x;
}

int main() {
  void *ctx = __enzyme_context(1);
  double *dscale = (double *)__enzyme_shadow(ctx, &scale, 0);
  double *dw = (double *)__enzyme_shadow(ctx, &weights, 0);
  for (int i = 1; i <= 3; i++) {
    __enzyme_zero_shadows(ctx);
    double x = i;
    double dx = __enzyme_autodiff(f, enzyme_context, ctx, x);
    APPROX_EQ(dx, 2 * 2.0 * 8.0 * x + 7.0, 1e-10);
    APPROX_EQ(*dscale, 8.0 * x * x, 1e-10);
    APPROX_EQ(dw[0], 2.0 * x * x, 1e-10);
    APPROX_EQ(dw[1], 2.0 * x * x, 1e-10);
  }
  return 0;
}
