// RUN: %clang -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -

// A global that gets memory from aligned_alloc outside differentiated code,
// as newer flang does for ALLOCATE, gets a zeroed allocation of the same size
// in its implicit shadow.

#include "../test_utils.h"
#include <stdlib.h>

extern void *__enzyme_context(int);
extern void *__enzyme_shadow(void *, void *, int);
extern void __enzyme_zero_shadows(void *);
extern double __enzyme_autodiff(double (*)(double), ...);
extern int enzyme_context;

double *a;

__attribute__((noinline)) void init() {
  a = (double *)aligned_alloc(64, 8 * sizeof(double));
  for (int i = 0; i < 8; i++)
    a[i] = i + 1;
}

double f(double x) {
  double sum = 0;
  for (int i = 0; i < 8; i++)
    sum += a[i] * a[i] * x;
  return sum;
}

int main() {
  init();
  void *ctx = __enzyme_context(1);
  double **da = (double **)__enzyme_shadow(ctx, &a, 0);
  for (int j = 1; j <= 2; j++) {
    __enzyme_zero_shadows(ctx);
    double x = j;
    double dx = __enzyme_autodiff(f, enzyme_context, ctx, x);
    APPROX_EQ(dx, 204.0, 1e-10);
    for (int i = 0; i < 8; i++)
      APPROX_EQ((*da)[i], 2.0 * (i + 1) * x, 1e-10);
  }
  free(a);
  return 0;
}
