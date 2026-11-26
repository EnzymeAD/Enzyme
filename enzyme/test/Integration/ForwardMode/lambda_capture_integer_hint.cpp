// RUN: %clang++ -std=c++17 -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang++ -std=c++17 -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang++ -std=c++17 -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang++ -std=c++17 -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -

// Reduced from MFEM dFEM: strided views captured by value in forall lambdas,
// with an __enzyme_integer hint on the stride. Until Enzyme lowers the hint
// it is an opaque call, so the lambda's closure stays in memory, and the
// first loop copies x into scratch through pointers loaded from it.

#include "../test_utils.h"
#include <stddef.h>

extern "C" void __enzyme_integer(void *, size_t);
extern int enzyme_dup, enzyme_const, enzyme_runtime_activity;
extern void __enzyme_fwddiff(void *, ...);

template <typename T> struct view {
  T *data;
  size_t stride;
  __attribute__((noinline)) void set_layout() { stride = 1; }
  T &operator()(int i) const {
    __enzyme_integer((void *)&stride, sizeof(stride));
    return data[i * stride];
  }
};

template <typename F> void forall(int n, F f) {
  for (int i = 0; i < n; i++)
    f(i);
}

__attribute__((noinline)) void cube(int n, const double *xp, double *yp,
                                    double *sp) {
  view<const double> x{xp};
  view<double> y{yp}, scratch{sp};
  x.set_layout();
  y.set_layout();
  scratch.set_layout();
  forall(n, [=](int q) { scratch(q) = x(q); });
  forall(n, [=](int q) { scratch(q) = scratch(q) * x(q); });
  forall(n, [=](int q) { y(q) = scratch(q) * x(q); });
}

int main() {
  const int n = 3;
  double x[n] = {1, 2, 3}, dx[n] = {1, 1, 1};
  double y[n] = {0}, dy[n] = {0}, s[n] = {0}, ds[n] = {0};
  __enzyme_fwddiff((void *)cube, enzyme_const, n, enzyme_dup, x, dx,
                   enzyme_dup, y, dy, enzyme_dup, s, ds,
                   enzyme_runtime_activity);
  for (int i = 0; i < n; i++) {
    printf("i=%d y=%f dy=%f ds=%f\n", i, y[i], dy[i], ds[i]);
    APPROX_EQ(dy[i], 3 * x[i] * x[i], 1e-10);
    APPROX_EQ(ds[i], 2 * x[i], 1e-10);
  }
  return 0;
}
