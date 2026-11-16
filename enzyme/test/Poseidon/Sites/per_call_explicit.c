// REQUIRES: enzyme
// RUN: rm -rf %t && mkdir %t
// RUN: %poseidon_clang -O2 -poseidon-profile-generate -poseidon-cache=%t/cg %s -lstdc++ -lm -o %t/gen
// RUN: env POSEIDON_PROFILE_DIR=%t/prof %t/gen | FileCheck --check-prefix=SAME %s
// RUN: ls %t/prof | FileCheck --check-prefix=PROFILES %s
// RUN: %poseidon_clang -O2 -poseidon-profile-use=%t/prof -poseidon-tau=1e-6 -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-enable-herbie=0 -poseidon-cache=%t/c %s -lstdc++ -lm -o %t/use 2>&1 | FileCheck --check-prefix=USE %s
// RUN: %t/use | FileCheck --check-prefix=RUNS %s
// RUN: not %clang -fsyntax-only -I%poseidon_profile_inc -DMACRO %s 2>&1 | FileCheck --check-prefix=MACRO %s

// C has no overloading, so __poseidon_optimize takes the automatic list only,
// and an explicit list there does not compile. An explicit list is a call of
// the marker the program declares, under either of its names, and reads as
// it reads in C++: with activity markers it is a site over the function, and
// an automatic list with a pointer is a per-call site.

#include "poseidon/poseidon.h"
#include <math.h>
#include <stdio.h>

int enzyme_dup, enzyme_const;
void __poseidon_fp_optimize(void *, ...);
void __enzyme_fp_optimize(void *, ...);

void wave(const double *x, double *y, int n) {
  for (int i = 0; i < n; ++i)
    y[i] = exp(-0.5 * x[i] * x[i]) * cos(4.0 * x[i]);
}

void scale(const double *x, double *y, int n) {
  for (int i = 0; i < n; ++i)
    y[i] = sqrt(x[i] * x[i] + 1.0);
}

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
#ifdef MACRO
  // MACRO: error: too many arguments to function call
  __poseidon_optimize(wave, enzyme_const, x, enzyme_dup, y1, dy, n);
#endif
  printf("%d %d\n", y1[100] == y2[100], y3[100] == y4[100]);
  return 0;
}

// SAME: 1 1
// PROFILES: preprocess_scale.poseidon.main.0.fpprofile
// PROFILES-NEXT: preprocess_wave.fpprofile
// USE-DAG: [poseidon] preprocess_wave: tau=1.000000e-06
// USE-DAG: [poseidon] preprocess_scale.poseidon.main.0: tau=1.000000e-06
// RUNS: {{[01] [01]}}
