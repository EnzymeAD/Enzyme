// Separate compilation: main and its callee are compiled as different modules,
// each with Enzyme, and only linked afterwards (no LTO). The caller refers to
// the callee's derivative through an external symbol the defining module
// exports.
// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O0 -DMODULE_CALLEE %s -S -emit-llvm -o %t.callee.ll %newLoadClangEnzyme -mllvm -enzyme-separate-compilation -mllvm -enzyme-export-derivatives=reverse,forward; %clang -std=c11 -O0 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-separate-compilation | %lli --extra-module=%t.callee.ll - ; fi
// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O2 -DMODULE_CALLEE %s -S -emit-llvm -o %t.callee.ll %newLoadClangEnzyme -mllvm -enzyme-separate-compilation -mllvm -enzyme-export-derivatives=reverse,forward; %clang -std=c11 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-separate-compilation | %lli --extra-module=%t.callee.ll - ; fi
// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O2 -DMODULE_CALLEE %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-separate-compilation -mllvm -enzyme-export-derivatives=reverse | %FileCheck %s --check-prefix=CALLEE; fi
// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-separate-compilation | %FileCheck %s --check-prefix=CALLER; fi

#include "../test_utils.h"

#ifdef MODULE_CALLEE

extern double sin(double);

static double h(double v) { return sin(v) * v; }

void g(double *x, double *y, int n) {
  for (int i = 0; i < n; i++)
    y[i] = h(x[i]) * x[i];
}

#else

extern double sin(double);
extern double cos(double);

void g(double *x, double *y, int n);

double f(double *x, int n) {
  double y[4] = {0};
  g(x, y, n);
  double s = 0;
  for (int i = 0; i < n; i++)
    s += y[i] * x[i];
  return s;
}

void __enzyme_autodiff(void *, ...);
double __enzyme_fwddiff(void *, ...);

int main() {
  double x[4] = {1, 2, 3, 4}, dx[4] = {0};
  __enzyme_autodiff((void *)f, x, dx, 4);
  for (int i = 0; i < 4; i++) {
    double v = x[i];
    // f(x) = sum_i sin(x_i) x_i^3
    double expected = cos(v) * v * v * v + 3 * sin(v) * v * v;
    APPROX_EQ(dx[i], expected, 1e-10);
    double tx[4] = {0};
    tx[i] = 1;
    APPROX_EQ(__enzyme_fwddiff((void *)f, x, tx, 4), expected, 1e-10);
  }
  return 0;
}

#endif

// CALLEE: @__enzyme_sep_rev_w1_g = {{.*}}constant { ptr, ptr } { ptr @augmented_g, ptr @diffeg }

// CALLER-DAG: @__enzyme_sep_rev_w1_g = external {{.*}}constant { ptr, ptr }
// CALLER-DAG: @__enzyme_sep_fwd_w1_g = external {{.*}}constant ptr
