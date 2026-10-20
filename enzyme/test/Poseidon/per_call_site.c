// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -O2 -ffp-contract=on -I%FPProfileInc %s -lm -o %t/stock
// RUN: %t/stock > %t/stock.out
// RUN: %poseidon_clang -O2 -poseidon-profile-generate -poseidon-cache=%t/cg %s -lstdc++ -lm -o %t/gen 2>/dev/null
// RUN: env POSEIDON_PROFILE_DIR=%t/prof %t/gen > %t/gen.out
// RUN: diff %t/stock.out %t/gen.out
// RUN: %poseidon_clang -O2 -poseidon-profile-use=%t/prof -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-enable-herbie=0 -poseidon-cache=%t/c %s -lstdc++ -lm -o %t/use 2>&1 | FileCheck %s
// RUN: %t/use > %t/use.out
// RUN: paste -d' ' %t/stock.out %t/use.out | awk '{ print ($1 == $3), ($2 == $4) }' | FileCheck --check-prefix=SAME %s
// REQUIRES: poseidon, enzyme

// The per-call forms in C: the return value comes back, C's argument
// conversions apply (a float and an int reach a function of doubles), and the
// target may be a macro.

#include "poseidon/poseidon.h"
#include <math.h>
#include <stdio.h>

#define LOOSE 1e-3

double damp(double x, double t) { return exp(-x * t) * cos(x); }

int main(void) {
  double a = 0, b = 0;
  for (int i = 0; i < 256; ++i) {
    float x = 0.01f * (float)i;
    a += __poseidon_optimize_tau(LOOSE, damp, x, 2);
    b += damp(x, 2);
  }
  printf("%.17g %.17g\n", a, b);
  return 0;
}

// CHECK: [poseidon] preprocess_damp.poseidon.main.0: tau=1.000000e-03
// CHECK: Applying solution for CS: All FP64(0%) + FP32(100%)
// SAME: 0 1
