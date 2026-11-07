// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli -
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli -
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli -
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli -

// A global without a marked shadow, written by a callee and read after the
// call in the same block. The load is visited before the call, so before the
// callee's derivative gives the global its shadow; the load must not be
// taken for a read of inactive memory.

#include "../test_utils.h"

void __enzyme_autodiff(void *, ...);

static double g = 1.0;

__attribute__((noinline)) static void f(double *x) { g = x[0] * x[0]; }

static double h(double *x) {
  f(x);
  return g;
}

int main(void) {
  double x = 3.0, dx = 0.0;
  __enzyme_autodiff((void *)h, &x, &dx);
  APPROX_EQ(dx, 6.0, 1e-10);
  return 0;
}
