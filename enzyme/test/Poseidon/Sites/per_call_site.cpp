// REQUIRES: enzyme
// RUN: rm -rf %t && mkdir %t
// RUN: %clang -x c++ -O2 -ffp-contract=on -I%poseidon_profile_inc %s -lstdc++ -lm -o %t/stock
// RUN: %t/stock > %t/stock.out
// RUN: %poseidon_clangxx -O2 -poseidon-profile-generate -poseidon-cache=%t/cg -poseidon-print %s -o %t/gen 2>&1 | FileCheck --check-prefix=GEN %s
// RUN: env POSEIDON_PROFILE_DIR=%t/prof %t/gen > %t/gen.out
// RUN: diff %t/stock.out %t/gen.out
// RUN: ls %t/prof | FileCheck --check-prefix=PROFILES %s
// RUN: %poseidon_clangxx -O2 -g -poseidon-profile-use=%t/prof -poseidon-early-prune -poseidon-tau=1e-6 -poseidon-cost-model=%host_cost_model -poseidon-enable-rewrite=0 -poseidon-print -poseidon-cache=%t/c2 -poseidon-explain=%t/x -Rpass-analysis=poseidon %s -o %t/use 2> %t/use.err
// RUN: FileCheck --check-prefix=USE %s < %t/use.err
// RUN: FileCheck --check-prefix=REMARK %s < %t/use.err
// RUN: %poseidon_explain %t/x --format report | FileCheck --check-prefix=REPORT %s
// RUN: %t/use > %t/use.out
// RUN: paste -d' ' %t/stock.out %t/use.out | awk '{ print ($1 == $4), ($2 == $5), ($3 == $6) }' | FileCheck --check-prefix=SAME %s
// The profile of each call is keyed by the call, the same at every -O level.
// RUN: %poseidon_clangxx -O0 -poseidon-profile-use=%t/prof -poseidon-early-prune -poseidon-tau=1e-6 -poseidon-cost-model=%host_cost_model -poseidon-enable-rewrite=0 -poseidon-cache=%t/c0 %s -poseidon-print -o %t/use0 2>&1 | FileCheck --check-prefix=LEVEL %s
// RUN: %poseidon_clangxx -O3 -poseidon-profile-use=%t/prof -poseidon-early-prune -poseidon-tau=1e-6 -poseidon-cost-model=%host_cost_model -poseidon-enable-rewrite=0 -poseidon-cache=%t/c3 %s -poseidon-print -o %t/use3 2>&1 | FileCheck --check-prefix=LEVEL %s
// RUN: %poseidon_clangxx -O0 -poseidon-profile-generate -poseidon-cache=%t/cg0 %s -o %t/gen0
// RUN: env POSEIDON_PROFILE_DIR=%t/prof0 %t/gen0 > /dev/null
// RUN: %poseidon_clangxx -O2 -poseidon-profile-use=%t/prof0 -poseidon-early-prune -poseidon-tau=1e-6 -poseidon-cost-model=%host_cost_model -poseidon-enable-rewrite=0 -poseidon-cache=%t/c20 %s -poseidon-print -o %t/use20 2>&1 | FileCheck --check-prefix=LEVEL %s
// RUN: %t/use0 > %t/use0.out
// RUN: diff %t/use.out %t/use0.out
// RUN: not %clang -x c++ -O1 -S -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/unrelated_site_profile -mllvm -poseidon-early-prune -DNONCONST %s -o /dev/null 2>&1 | FileCheck --check-prefix=NONCONST %s
// RUN: not %clang -x c++ -O1 -S -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/unrelated_site_profile -mllvm -poseidon-early-prune -DMARKED %s -o /dev/null 2>&1 | FileCheck --check-prefix=MARKED %s

// __poseidon_optimize makes one call the site, with its own profile, target and
// rewritten copy; only the call of sq whose range FP32 holds is rewritten.

#include "poseidon/poseidon.h"
#include <cmath>
#include <cstdio>

double sq(double x, double y) { return x * x + y * y; }

#ifdef MARKED
POSEIDON_OPTIMIZE
#endif
void wave(const double *x, double *y, int n) {
  for (int i = 0; i < n; ++i)
    y[i] = std::exp(-0.5 * x[i] * x[i]) * std::cos(4.0 * x[i]);
}

int main(int argc, char **argv) {
  static double x[256], y1[256], y2[256], y3[256];
  double small = 0, huge = 0, plain = 0;
  for (int i = 0; i < 256; ++i) {
    double s = 1.0 + 0.01 * i, h = 1e30 * s;
    // REMARK: per_call_site.cpp:[[@LINE+2]]:{{[0-9]+}}: remark: Poseidon site sq: target 1e-06
    // REPORT: called at per_call_site.cpp:[[@LINE+1]]:
    small += __poseidon_optimize(sq, s, 2.0 * s);
    // REMARK: per_call_site.cpp:[[@LINE+2]]:{{[0-9]+}}: remark: Poseidon site sq: target 1e-06
    // REPORT: called at per_call_site.cpp:[[@LINE+1]]:
    huge += __poseidon_optimize(sq, h, 2.0 * h);
    plain += sq(s, 2.0 * s);
  }
  for (int i = 0; i < 256; ++i)
    x[i] = -2.0 + 4.0 * i / 255;
#ifdef NONCONST
  double t = argc * 1e-3;
  __poseidon_optimize_tau(t, wave, x, y1, 256);
#endif
  // REMARK: per_call_site.cpp:[[@LINE+2]]:{{[0-9]+}}: remark: Poseidon site wave: target 0.001
  // REPORT: called at per_call_site.cpp:[[@LINE+1]]:
  __poseidon_optimize_tau(1e-3, wave, x, y1, 256);
  // REMARK: per_call_site.cpp:[[@LINE+2]]:{{[0-9]+}}: remark: Poseidon site wave: target 1e-12
  // REPORT: called at per_call_site.cpp:[[@LINE+1]]:
  __poseidon_optimize_tau(1e-12, wave, x, y2, 256);
  wave(x, y3, 256);
  printf("%.17g %.17g %.17g\n%.17g %.17g %.17g\n", small, huge, plain, y1[100],
         y2[100], y3[100]);
}

// GEN: [poseidon] sq(double, double) at call 0 in main: site 0, 0 shadow buffer(s), output(s) return, 1 call(s)
// GEN: [poseidon] sq(double, double) at call 1 in main: site 1, 0 shadow buffer(s), output(s) return, 1 call(s)
// GEN: [poseidon] wave(double const*, double*, int) at call 2 in main: site 2, 2 shadow buffer(s), output(s) 1, 1 call(s)
// GEN: [poseidon] wave(double const*, double*, int) at call 3 in main: site 3, 2 shadow buffer(s), output(s) 1, 1 call(s)

// PROFILES: _Z2sqdd.poseidon.main.0.fpprofile
// PROFILES-NEXT: _Z2sqdd.poseidon.main.1.fpprofile
// PROFILES-NEXT: _Z4wavePKdPdi.poseidon.main.2.fpprofile
// PROFILES-NEXT: _Z4wavePKdPdi.poseidon.main.3.fpprofile

// USE: [poseidon] preprocess__Z2sqdd.poseidon.main.0: tau=1.000000e-06
// USE: Applying solution for precision change All FP64(0%) + FP32(100%)
// USE: [poseidon] preprocess__Z2sqdd.poseidon.main.1: tau=1.000000e-06
// USE: no rewrite applied for preprocess__Z2sqdd.poseidon.main.1
// USE: preprocess__Z4wavePKdPdi.poseidon.main.2: the site's own accuracy target 1.000000e-03 overrides -poseidon-tau=1.000000e-06
// USE: Applying solution for precision change All FP64(0%) + FP32(100%)
// USE: preprocess__Z4wavePKdPdi.poseidon.main.3: the site's own accuracy target 1.000000e-12 overrides -poseidon-tau=1.000000e-06
// USE: no rewrite applied for preprocess__Z4wavePKdPdi.poseidon.main.3

// SAME: 0 1 1
// SAME-NEXT: 0 1 1

// LEVEL-COUNT-2: Applying solution for precision change All FP64(0%) + FP32(100%)
// LEVEL-NOT: Applying solution

// NONCONST: error: {{.*}}Poseidon: the accuracy target of wave(double const*, double*, int) must be a compile-time floating-point constant
// MARKED: error: {{.*}}Poseidon: wave(double const*, double*, int) carries POSEIDON_OPTIMIZE, so every call of it is already a site; remove this __poseidon_optimize or the annotation
