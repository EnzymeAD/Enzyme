// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-global-activity=1 | %lli -
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-global-activity=1 | %lli -
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-global-activity=1 | %lli -
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-global-activity=1 | %lli -

// A global written by one callee and read by another. The reader must not
// give the global a local shadow in place of its real one: the writer's
// derivative reads the real shadow, and would never see the reader's adjoint.

#include "../test_utils.h"

void __enzyme_autodiff(void *, ...);

static double g[2] = {0};

__attribute__((noinline)) static void f(double *x) {
  for (int i = 0; i < 2; i++)
    g[i] = x[i] * x[i];
}

__attribute__((noinline)) static double loss(double *x) {
  double s = 0;
  for (int i = 0; i < 2; i++)
    s += g[i];
  return s;
}

static double h(double *x) {
  f(x);
  return loss(x);
}

int main(void) {
  double x[2] = {3.0, 4.0}, dx[2] = {0, 0};
  __enzyme_autodiff((void *)h, x, dx);
  APPROX_EQ(dx[0], 6.0, 1e-10);
  APPROX_EQ(dx[1], 8.0, 1e-10);
  return 0;
}
