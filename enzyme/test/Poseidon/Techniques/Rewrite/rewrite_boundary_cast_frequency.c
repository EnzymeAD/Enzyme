// REQUIRES: enzyme
// RUN: rm -rf %t && mkdir %t
// RUN: %poseidon_clang -O2 -poseidon-profile-generate -poseidon-cache=%t/cg %s -o %t/gen
// RUN: env POSEIDON_PROFILE_DIR=%t/prof %t/gen > /dev/null
// RUN: env FAKE_HERBIE_DIR=%S/Inputs/rewrite_boundary_cast_frequency %poseidon_clang -O2 -poseidon-profile-use=%t/prof -poseidon-early-prune -poseidon-cost-model=%host_cost_model -poseidon-enable-rewrite=1 -poseidon-herbie-binary=%S/../../Inputs/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-precision=0 -poseidon-subgraph-split-uses=2 -poseidon-subgraph-split-ops=1 -poseidon-cost-budget=-0.01 -poseidon-print -poseidon-cache=%t/c %s -S -emit-llvm -o %t/use.ll 2>&1 | FileCheck --check-prefix=PRICE %s
// RUN: FileCheck --check-prefix=IR %s < %t/use.ll

// A binary32 rewrite of the value a loop carries, with Herbie's answer canned
// in Techniques/Rewrite/Inputs/rewrite_boundary_cast_frequency. Written after the loop's output
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
// PRICE: {{^}}1.113646e-06 -2.944000e+03 7.500000e-01 9.921875e-01 (*.f32 (*.f32 v0 v1) (+.f32 v3 (+.f32 v0 v1))){{$}}
// PRICE: [poseidon] pin: iterate.poseidon.main.0:R3=(*.f32 (*.f32 v0 v1) (+.f32 v3 (+.f32 v0 v1))){{$}}

// IR-LABEL: define {{.*}} @main(
// IR: [[X:%.+]] = fdiv double
// IR: [[XF:%.+]] = fptrunc double [[X]] to float
// IR: [[XPHI:%.+]] = phi float {{.*}}[ [[XF]], %
// IR-NOT: fptrunc
// IR-NOT: fpext
// IR: [[S:%.+]] = fadd float {{.*}}[[XPHI]]{{(,|$)}}
// IR-NOT: fptrunc
// IR-NOT: fpext
// IR: [[XN:%.+]] = fmul float {{.*}}[[S]]{{(,|$)}}
// IR-NOT: fptrunc
// IR-NOT: fpext
// IR: br i1 {{.*}} !llvm.loop
// IR-NOT: fptrunc
// IR: fpext float [[XN]] to double
