// REQUIRES: enzyme
// RUN: rm -rf %t && mkdir %t
// RUN: %poseidon_clangxx -O2 -poseidon-profile-generate -poseidon-cache=%t/cg %s -o %t/gen
// RUN: env POSEIDON_PROFILE_DIR=%t/prof %t/gen | FileCheck --check-prefix=SAME %s
// RUN: ls %t/prof | FileCheck --check-prefix=PROFILES %s
// RUN: %poseidon_clangxx -O2 -poseidon-profile-use=%t/prof -poseidon-early-prune -poseidon-tau=1e-6 -poseidon-cost-model=%host_cost_model -poseidon-enable-algebraic=0 -poseidon-cache=%t/c %s -poseidon-print -Rpass-missed=poseidon -o %t/use 2>&1 | FileCheck --check-prefix=USE %s
// RUN: %t/use | FileCheck --check-prefix=RUNS %s
// RUN: not %clang -x c++ -O2 -I%poseidon_profile_inc %s -o %t/noplugin 2>&1 | FileCheck --check-prefix=LINK %s
// RUN: %clang -x c++ -O2 -I%poseidon_profile_inc %loadClangPoseidon %s -lstdc++ -lm -o %t/plain
// RUN: %t/plain | FileCheck --check-prefix=SAME %s
// RUN: not %clang -x c++ -O1 -gline-tables-only -S -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/unrelated_site_profile -mllvm -poseidon-early-prune -DLISTS %s -o /dev/null 2>&1 | FileCheck --check-prefix=LISTS %s

// __poseidon_optimize takes either list: an explicit one makes the call a site
// over the function itself, an automatic one with a pointer a per-call site;
// __poseidon_fp_optimize and __enzyme_fp_optimize read their lists the same
// way. Without the plugin an explicit list does not link; with it and no
// profile action every call runs as written.

#include "poseidon/poseidon.h"
#include <cmath>
#include <cstdio>

int enzyme_dup, enzyme_const, poseidon_tau;
template <typename R, typename... T> R __poseidon_fp_optimize(void *, T...);
template <typename R, typename... T> R __enzyme_fp_optimize(void *, T...);

void wave(const double *x, double *y, int n) {
  for (int i = 0; i < n; ++i)
    y[i] = std::exp(-0.5 * x[i] * x[i]) * std::cos(4.0 * x[i]);
}

double energy(const double *x, int n) {
  double s = 0;
  for (int i = 0; i < n; ++i)
    s += 0.5 * x[i] * x[i] + std::sin(x[i]);
  return s;
}

double damp(const double *x, double t) { return std::exp(-x[0] * t) * x[1]; }

void scale(const double *x, double *y, int n) {
  for (int i = 0; i < n; ++i)
    y[i] = std::sqrt(x[i] * x[i] + 1.0);
}

double sq(double x, double y) { return x * x + y * y; }

void shift(const double *x, double *y, int n) {
  for (int i = 0; i < n; ++i)
    y[i] = std::log1p(x[i] * x[i]) - 0.25 * x[i];
}

int pick(int mode, int n) { return mode + n; }

int main() {
  const int n = 256;
  static double x[n], dx[n], y1[n], y2[n], y3[n], y4[n], y5[n], dy[n];
  static double y6[n], y7[n], y8[n];
  for (int i = 0; i < n; ++i) {
    x[i] = -2.0 + 4.0 * i / (n - 1);
    dy[i] = 1.0;
  }
  __poseidon_optimize(wave, enzyme_const, x, enzyme_dup, y1, dy, n);
  wave(x, y2, n);
  double e = __poseidon_optimize_tau(1e-3, energy, enzyme_dup, x, dx, n);
  double d = __poseidon_optimize_tau(1e-12, damp, enzyme_const, x, 2.0);
  __enzyme_fp_optimize<void>((void *)scale, x, y3, n);
  scale(x, y4, n);
  __enzyme_fp_optimize<void>((void *)wave, enzyme_const, x, enzyme_dup, y5, dy,
                             n);
  double s = __poseidon_fp_optimize<double>((void *)sq, x[3], x[7]);
  __poseidon_optimize_tau(1e-9, shift, x, y6, n);
  __poseidon_fp_optimize<void>((void *)shift, poseidon_tau, 1e-8, x, y7, n);
  shift(x, y8, n);
  printf("%d %d %d %d %d %d %d %d\n", y1[100] == y2[100], e == energy(x, n),
         d == damp(x, 2.0), y3[100] == y4[100], y5[100] == y2[100],
         s == sq(x[3], x[7]), y6[100] == y8[100], y7[100] == y8[100]);
  printf("%.6g %.6g %.6g %.6g %.6g\n", y1[100], e, d, y3[100], s);

#ifdef LISTS
  // LISTS-DAG: per_call_explicit.cpp:[[@LINE+1]]:{{[0-9]+}}: error: Poseidon: pick(int, int) takes the arguments of this call as written, but enzyme_const among them is an activity marker, which makes them an explicit list; pass pick(int, int)'s own arguments, or an explicit list that puts enzyme_dup before each pointer and its shadow after it, or enzyme_const before it
  int p = __poseidon_optimize(pick, enzyme_const, n);
  // LISTS-DAG: per_call_explicit.cpp:[[@LINE+1]]:{{[0-9]+}}: error: Poseidon: wave(double const*, double*, int) does not take the arguments of this call as written, and none of them is an activity marker; pass wave(double const*, double*, int)'s own arguments, or an explicit list that puts enzyme_dup before each pointer and its shadow after it, or enzyme_const before it
  __poseidon_optimize(wave, x, y1);
  // LISTS-DAG: per_call_explicit.cpp:[[@LINE+1]]:{{[0-9]+}}: error: Poseidon: this call passes 2 argument(s) to scale(double const*, double*, int), which takes 3, and none of them is an activity marker; pass scale(double const*, double*, int)'s own arguments, or an explicit list that puts enzyme_dup before each pointer and its shadow after it, or enzyme_const before it
  __enzyme_fp_optimize<void>((void *)scale, x, y3);
  // LISTS-DAG: per_call_explicit.cpp:[[@LINE+1]]:{{[0-9]+}}: error: Poseidon: parameter 1 of _Z4wavePKdPdi is a pointer without an activity marker in an explicit list; put enzyme_dup before it and its shadow after it, or enzyme_const before it
  __poseidon_optimize(wave, enzyme_const, x, y1, dy, n);
  // LISTS-DAG: per_call_explicit.cpp:[[@LINE+1]]:{{[0-9]+}}: error: Poseidon: activity marker without an argument
  __poseidon_optimize(wave, enzyme_const, x, enzyme_dup, y1, dy, enzyme_const);
  // LISTS-DAG: per_call_explicit.cpp:[[@LINE+1]]:{{[0-9]+}}: error: Poseidon: poseidon_tau among the arguments of this call of shift(double const*, double*, int) gives the accuracy target only of a marker call; give the target as the first argument of __poseidon_optimize_tau
  __poseidon_optimize(shift, poseidon_tau, 1e-6, x, y6, n);
  // LISTS-DAG: per_call_explicit.cpp:[[@LINE+1]]:{{[0-9]+}}: error: Poseidon: this call passes 2 argument(s) to shift(double const*, double*, int), which takes 3, and none of them is an activity marker; pass shift(double const*, double*, int)'s own arguments, or an explicit list that puts enzyme_dup before each pointer and its shadow after it, or enzyme_const before it
  __poseidon_fp_optimize<void>((void *)shift, poseidon_tau, 1e-6, x, y6);
  // LISTS-DAG: per_call_explicit.cpp:[[@LINE+1]]:{{[0-9]+}}: error: Poseidon: the accuracy target of shift(double const*, double*, int) must be a compile-time floating-point constant (a literal, a macro or a constant expression)
  __poseidon_fp_optimize<void>((void *)shift, poseidon_tau, d, x, y6, n);
  // LISTS-DAG: per_call_explicit.cpp:[[@LINE+1]]:{{[0-9]+}}: error: Poseidon: this call uses the value of shift(double const*, double*, int) as another type than it returns; give the call shift(double const*, double*, int)'s return type
  d += __poseidon_fp_optimize<double>((void *)shift, x, y6, n);
  printf("%d %g\n", p, d);
  // LISTS: 9 errors generated.
#endif
}

// SAME: 1 1 1 1 1 1 1 1

// PROFILES: _Z2sqdd.fpprofile
// PROFILES-NEXT: _Z4dampPKdd.fpprofile
// PROFILES-NEXT: _Z4wavePKdPdi.fpprofile
// PROFILES-NEXT: _Z5scalePKdPdi.poseidon.main.0.fpprofile
// PROFILES-NEXT: _Z5shiftPKdPdi.poseidon.main.1.fpprofile
// PROFILES-NEXT: _Z5shiftPKdPdi.poseidon.main.2.fpprofile
// PROFILES-NEXT: _Z6energyPKdi.fpprofile

// Each site is solved against its own target.
// USE-DAG: [poseidon] preprocess__Z4wavePKdPdi: tau=1.000000e-06
// USE-DAG: [poseidon] pin: _Z4wavePKdPdi:P0=All FP64(0%) + FP32(100%)
// USE-DAG: [poseidon] _Z4wavePKdPdi: same body and target as an earlier site; reusing its solve (preprocess__Z4wavePKdPdi)
// USE-DAG: [poseidon] preprocess__Z6energyPKdi: the site's own accuracy target 1.000000e-03 overrides -poseidon-tau=1.000000e-06
// USE-DAG: [poseidon] pin: _Z6energyPKdi:P0=All FP64(0%) + FP32(100%)
// USE-DAG: [poseidon] preprocess__Z4dampPKdd: the site's own accuracy target 1.000000e-12 overrides -poseidon-tau=1.000000e-06
// USE-DAG: per_call_explicit.cpp:[[@LINE-55]]:14: remark: Poseidon left site damp as written: the selected frontier point leaves the site as written [-Rpass-missed=poseidon]
// USE-DAG: [poseidon] preprocess__Z5scalePKdPdi.poseidon.main.0: tau=1.000000e-06
// USE-DAG: [poseidon] pin: _Z5scalePKdPdi.poseidon.main.0:P0=All FP64(0%) + FP32(100%)
// USE-DAG: [poseidon] preprocess__Z2sqdd: tau=1.000000e-06
// USE-DAG: [poseidon] pin: _Z2sqdd:P0=All FP64(0%) + FP32(100%)
// USE-DAG: [poseidon] preprocess__Z5shiftPKdPdi.poseidon.main.1: the site's own accuracy target 1.000000e-09 overrides -poseidon-tau=1.000000e-06
// USE-DAG: per_call_explicit.cpp:[[@LINE-55]]:3: remark: Poseidon left site shift as written: the selected frontier point leaves the site as written [-Rpass-missed=poseidon]
// USE-DAG: [poseidon] preprocess__Z5shiftPKdPdi.poseidon.main.2: the site's own accuracy target 1.000000e-08 overrides -poseidon-tau=1.000000e-06
// USE-DAG: [poseidon] pin: _Z5shiftPKdPdi.poseidon.main.2:P0=Funcs FP64(50%) + FP32(50%)
// USE-NOT: warning

// RUNS: 0 0 1 0 0 0 1 0

// LINK: undefined reference to `__poseidon_site_explicit'
// LINK: undefined reference to `__poseidon_site_explicit_tau'
