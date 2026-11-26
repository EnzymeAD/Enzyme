// REQUIRES: enzyme
// RUN: rm -rf %t && mkdir %t
// RUN: %poseidon_clang -O2 -poseidon-profile-generate -poseidon-cache=%t/cg %s -o %t/gen
// RUN: env POSEIDON_PROFILE_DIR=%t/prof %t/gen > /dev/null
// RUN: env FAKE_HERBIE_DIR=%S/Inputs/fmod_operator %poseidon_clang -O2 -poseidon-profile-use=%t/prof -poseidon-early-prune -poseidon-cost-model=%host_cost_model -poseidon-enable-algebraic=1 -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh -poseidon-enable-precision=1 -poseidon-cost-budget=-0.5 -poseidon-print -poseidon-cache=%t/c %s -S -emit-llvm -o %t/use.ll 2>&1 | FileCheck %s
// RUN: FileCheck --check-prefix=IR %s < %t/use.ll

// A host call to erf is erf, not the float form of "er": Herbie gets (erf ...),
// the erf row prices it (8, fadd 4) and the float site calls erff (2 + 1 + 2).

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
