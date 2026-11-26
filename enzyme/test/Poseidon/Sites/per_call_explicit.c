// REQUIRES: enzyme
// RUN: rm -rf %t && mkdir %t
// RUN: %poseidon_clang -O2 -poseidon-profile-generate -poseidon-cache=%t/cg %s -o %t/gen
// RUN: env POSEIDON_PROFILE_DIR=%t/prof %t/gen | FileCheck --check-prefix=SAME %s
// RUN: ls %t/prof | FileCheck --check-prefix=PROFILES %s
// RUN: %poseidon_clang -O2 -poseidon-profile-use=%t/prof -poseidon-early-prune -poseidon-tau=1e-6 -poseidon-cost-model=%host_cost_model -poseidon-enable-rewrite=0 -poseidon-cache=%t/c %s -poseidon-print -o %t/use 2>&1 | FileCheck --check-prefix=USE %s
// RUN: %t/use | FileCheck --check-prefix=RUNS %s
// RUN: not %clang -fsyntax-only -I%poseidon_profile_inc -DMACRO %s 2>&1 | FileCheck --check-prefix=MACRO %s

// In C __poseidon_optimize takes the automatic list only; an explicit list is
// a marker call under either name and reads as in C++. A per-call form returns
// the call's value with C's argument conversions, and its target may be a macro.

#include "poseidon/poseidon.h"
#include <math.h>
#include <stdio.h>

#define LOOSE 1e-3

int enzyme_dup, enzyme_const;
void __poseidon_fp_optimize(void *, ...);
double __enzyme_fp_optimize(void *, ...);

void wave(const double *x, double *y, int n) {
  for (int i = 0; i < n; ++i)
    y[i] = exp(-0.5 * x[i] * x[i]) * cos(4.0 * x[i]);
}

void scale(const double *x, double *y, int n) {
  for (int i = 0; i < n; ++i)
    y[i] = sqrt(x[i] * x[i] + 1.0);
}

double sq(double x, double y) { return x * x + y * y; }

double damp(double x, double t) { return exp(-x * t) * cos(x); }

int main(void) {
  enum { n = 256 };
  static double x[n], y1[n], y2[n], y3[n], y4[n], dy[n];
  for (int i = 0; i < n; ++i) {
    x[i] = -2.0 + 4.0 * i / (n - 1);
    dy[i] = 1.0;
  }
  __poseidon_fp_optimize((void *)wave, enzyme_const, x, enzyme_dup, y1, dy, n);
  wave(x, y2, n);
  __enzyme_fp_optimize((void *)scale, x, y3, n);
  scale(x, y4, n);
  double s = __enzyme_fp_optimize((void *)sq, x[3], x[7]);
#ifdef MACRO
  // MACRO: per_call_explicit.c:[[@LINE+2]]:{{[0-9]+}}: error: too many arguments to function call, expected 3, have 6
  // MACRO: poseidon.h:{{[0-9]+}}:{{[0-9]+}}: note: expanded from macro '__poseidon_optimize'
  __poseidon_optimize(wave, enzyme_const, x, enzyme_dup, y1, dy, n);
  // MACRO: 1 error generated.
#endif
  double a = 0, b = 0;
  for (int i = 0; i < n; ++i) {
    float f = 0.01f * (float)i;
    a += __poseidon_optimize_tau(LOOSE, damp, f, 2);
    b += damp(f, 2);
  }
  printf("%d %d %d %d\n", y1[100] == y2[100], y3[100] == y4[100],
         s == sq(x[3], x[7]), a == b);
  return 0;
}

// SAME: 1 1 1 1
// PROFILES: damp.poseidon.main.1.fpprofile
// PROFILES-NEXT: scale.poseidon.main.0.fpprofile
// PROFILES-NEXT: sq.fpprofile
// PROFILES-NEXT: wave.fpprofile
// USE-DAG: [poseidon] preprocess_wave: tau=1.000000e-06
// USE-DAG: [poseidon] pin: wave:P0=All FP64(0%) + FP32(100%)
// USE-DAG: [poseidon] preprocess_scale.poseidon.main.0: tau=1.000000e-06
// USE-DAG: [poseidon] pin: scale.poseidon.main.0:P0=All FP64(0%) + FP32(100%)
// USE-DAG: [poseidon] preprocess_sq: tau=1.000000e-06
// USE-DAG: [poseidon] pin: sq:P0=All FP64(0%) + FP32(100%)
// USE-DAG: [poseidon] preprocess_damp.poseidon.main.1: tau=1.000000e-03
// USE-DAG: [poseidon] pin: damp.poseidon.main.1:P0=All FP64(0%) + FP32(100%)
// RUNS: 0 0 0 0
