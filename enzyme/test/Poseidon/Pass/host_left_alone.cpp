// REQUIRES: enzyme
// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -x c++ -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon %s -o %t/plain.ll 2>/dev/null
// RUN: %clang -x c++ -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites -mllvm -poseidon-print -mllvm -poseidon-cache=%t/u %s -o %t/left.ll 2>&1 | FileCheck --check-prefix=LEFT %s
// RUN: diff %t/plain.ll %t/left.ll
//
// RUN: %clang -x c++ -O2 -ffp-contract=on -I%poseidon_profile_inc %s -lstdc++ -lm -o %t/stock
// RUN: %t/stock > %t/stock.out
// RUN: %poseidon_clangxx -O2 -poseidon-profile-generate -poseidon-cache=%t/cg %s -o %t/gen 2>/dev/null
// RUN: env POSEIDON_PROFILE_DIR=%t/prof %t/gen > /dev/null
// RUN: %poseidon_clangxx -O2 -poseidon-profile-use=%t/prof -poseidon-tau=1e-6 -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-enable-herbie=0 -poseidon-print -poseidon-cache=%t/c %s -o %t/use 2> %t/use.err
// RUN: FileCheck --check-prefix=USE %s < %t/use.err
// RUN: %t/use > %t/use.out
// RUN: paste -d' ' %t/stock.out %t/use.out | awk '{ print ($1 == $4), ($2 == $5), ($3 == $6) }' | FileCheck --check-prefix=SAME %s

// A host module whose sites the solve all leaves alone is the code as written:
// the sites' bodies were kept out of line and the module went through the
// canonical prefix, so the module is compiled once more from its state before
// that, as a build without a profile compiles it, and that result is emitted.
// Without it the loop over qpoint is no longer vectorized. When a site is
// rewritten, every function the rewrite reached comes from the rewritten
// module: here both callers the annotated function's wrapper was inlined into.

#include "poseidon/poseidon.h"
#include <cstdio>

void vol(const double *a, double *v) {
  double J = a[0] * a[1] - a[2] * a[3];
  *v = J * (J - 1.0);
}

double qpoint(const double *g) {
  double a[4] = {g[0], g[1], g[2], g[3]}, v;
  __poseidon_optimize_tau(1e-12, vol, a, &v);
  return 2.0 * v;
}

POSEIDON_OPTIMIZE double sq(double x, double y) { return x * x + y * y; }

__attribute__((noinline)) double first(int n) {
  double s = 0.0;
  for (int i = 0; i < n; ++i)
    s += sq(1.0 + 0.01 * i, 2.0 + 0.02 * i);
  return s;
}

__attribute__((noinline)) double second(int n) {
  double s = 0.0;
  for (int i = n - 1; i >= 0; --i)
    s += sq(1.0 + 0.01 * i, 2.0 + 0.02 * i);
  return s;
}

int main() {
  const int n = 64;
  static double g[4 * n], r[n];
  for (int i = 0; i < 4 * n; ++i)
    g[i] = 1.0 + 1e-3 * (i % 7);
  double q = 0.0;
  for (int i = 0; i < n; ++i)
    q += qpoint(g + 4 * i);
  printf("%.17g %.17g %.17g\n", first(256), second(256), q);
  return 0;
}

// LEFT: [poseidon] every site of {{.*}}host_left_alone.cpp was left alone; emitting the module as written

// USE-DAG: no rewrite applied for preprocess__Z3volPKdPd.poseidon._Z6qpointPKd.0
// USE-DAG: Applying solution for CS: All FP64(0%) + FP32(100%)
// USE: [poseidon] {{.*}}host_left_alone.cpp: {{[0-9]+}} function(s) a rewrite reached; every other function as written

// The two callers of the rewritten sq differ from plain clang's program; the
// site left alone does not.
// SAME: 0 0 1
