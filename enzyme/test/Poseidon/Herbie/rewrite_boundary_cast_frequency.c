// REQUIRES: enzyme
// RUN: rm -rf %t && mkdir -p %t
// RUN: %poseidon_clang -O2 -poseidon-profile-generate -poseidon-cache=%t/cg %s -lstdc++ -lm -o %t/gen 2>/dev/null
// RUN: env POSEIDON_PROFILE_DIR=%t/prof %t/gen > /dev/null
// RUN: env FAKE_HERBIE_DIR=%S/../Inputs/rewrite_boundary_cast_frequency %poseidon_clang -O2 -poseidon-profile-use=%t/prof -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-enable-herbie=1 -poseidon-herbie-binary=%S/../Inputs/rewrite_boundary_cast_frequency/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-pt=0 -poseidon-min-uses-split=2 -poseidon-min-ops-split=1 -poseidon-cost-budget=-0.01 -poseidon-print -poseidon-cache=%t/c %s -S -emit-llvm -o %t/use.ll 2> %t/use.err
// RUN: FileCheck --check-prefix=PRICE %s < %t/use.err
// RUN: FileCheck --check-prefix=IR %s < %t/use.ll

// A binary32 rewrite of the value a loop carries, with Herbie's answer canned
// in Inputs/rewrite_boundary_cast_frequency. Written after the loop's output
// x * a + b * a, the rewrite truncates x, p and q and extends its result; in
// the emitted code x stays binary32 across the back edge, truncated once
// before the loop and extended once after it. The price follows: per
// execution of the output, three binary32 operations (1 each), the two
// conversions of the arguments (1 each: nothing measured says how often they
// run) and the conversions of x and of the result at the loop's one entry per
// eight iterations (1/8 each), against the 12 the rewrite erases, over 512
// executions.

#include "poseidon/poseidon.h"
#include <stdio.h>

double iterate(double p, double q) {
  double a = p * q;
  double b = p + q;
  double x = (p - q) / (p - q * q);
#pragma clang loop unroll(disable)
  for (int i = 0; i < 8; ++i)
    x = x * a + b * a;
  return x / p;
}

int main(void) {
  double s = 0;
  for (int i = 1; i <= 64; ++i)
    s += __poseidon_optimize(iterate, 0.5 + 0.001 * i, 0.25);
  printf("%.17g\n", s);
  return 0;
}

// PRICE: Initial ComputationCost: 1.200000e+01
// PRICE: -2.944000e+03{{.*}}(*.f32 (*.f32 v0 v1) (+.f32 v3 (+.f32 v0 v1)))
// PRICE: [poseidon] pin: iterate.poseidon.main.0:R3=(*.f32 (*.f32 v0 v1) (+.f32 v3 (+.f32 v0 v1)))

// IR-LABEL: define {{.*}} @main(
// IR: fptrunc double
// IR: phi float
// IR-NOT: fptrunc
// IR-NOT: fpext
// IR: br i1 {{.*}} !llvm.loop
// IR: fpext float
