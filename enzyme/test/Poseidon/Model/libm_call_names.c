// REQUIRES: enzyme
// RUN: rm -rf %t && mkdir %t
// RUN: %poseidon_clang -O2 -poseidon-profile-generate -poseidon-cache=%t/cg %s -o %t/gen
// RUN: env POSEIDON_PROFILE_DIR=%t/prof %t/gen > /dev/null
// RUN: env FAKE_HERBIE_DIR=%S/../Inputs/fmod_operator %poseidon_clang -O2 -poseidon-profile-use=%t/prof -poseidon-cost-model=%host_cost_model -poseidon-enable-herbie=1 -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh -poseidon-enable-pt=1 -poseidon-cost-budget=-0.5 -poseidon-print -poseidon-cache=%t/c %s -S -emit-llvm -o %t/use.ll 2>&1 | FileCheck %s
// RUN: FileCheck --check-prefix=IR %s < %t/use.ll

// A host call to erf is the math function erf, though its name ends in f as
// the float form of a function would: Herbie is given (erf ...), the cost model
// prices the call by the erf row (8 in double), beside the fadd (4), and the
// precision change that runs the site in float calls erff, priced at 2, with
// the fadd (1) and the two casts (1 each).

#include "poseidon/poseidon.h"
#include <math.h>
#include <stdio.h>

POSEIDON_OPTIMIZE void kernel(const double *x, double *y, int n) {
  for (int i = 0; i < n; ++i)
    y[i] = erf(x[i]) + x[i];
}

int main(void) {
  double x[64], y[64];
  for (int i = 0; i < 64; ++i)
    x[i] = -2.0 + i / 16.0;
  kernel(x, y, 64);
  printf("%g\n", y[40]);
  return 0;
}

// CHECK: :name "0" (+ v0 (erf v0)))
// CHECK: Initial ComputationCost: 1.200000e+01
// CHECK-NEXT: Initial HerbieAccuracy:
// CHECK-NEXT: Initial Expression: (+ v0 (erf v0))
// CHECK: [poseidon] precision candidate 'All FP64(0%) + FP32(100%)': initial=1.200000e+01 cand=5.000000e+00
// CHECK: Applying solution for precision change All FP64(0%) + FP32(100%)

// IR-LABEL: define {{.*}} @kernel(
// IR: call float @erff(float
