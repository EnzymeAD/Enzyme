// RUN: rm -rf %t && mkdir -p %t
// RUN: %poseidon_clangxx -O2 -poseidon-profile-generate -poseidon-cache=%t/c %s -o %t/gen 2>/dev/null
// RUN: env POSEIDON_PROFILE_DIR=%t/prof %t/gen > /dev/null
// RUN: %poseidon_clangxx -O2 -poseidon-profile-use=%t/prof -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-enable-herbie=0 -poseidon-cache=%t/c -S %s -o %t/pose.s 2> %t/use.err
// RUN: FileCheck %s < %t/use.err
// RUN: %clang -x c++ -O2 -ffp-contract=on -I%FPProfileInc -S %s -o %t/stock.s
// RUN: awk '/^_Z4wavePKdPdi:/{p=1} p{print} p&&/^\.Lfunc_end/{exit}' %t/stock.s | sed -E 's/\.(LBB|LCPI|Ltmp|Lfunc_end)[0-9_]+/.L/g' > %t/stock.fn
// RUN: awk '/^_Z4wavePKdPdi:/{p=1} p{print} p&&/^\.Lfunc_end/{exit}' %t/pose.s | sed -E 's/\.(LBB|LCPI|Ltmp|Lfunc_end)[0-9_]+/.L/g' > %t/pose.fn
// RUN: grep -q ret %t/stock.fn
// RUN: diff %t/stock.fn %t/pose.fn
// REQUIRES: poseidon, enzyme

// A site its target refuses is left as written: the function is the code plain
// clang emits for it.

#include "poseidon/poseidon.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

POSEIDON_OPTIMIZE_TAU(1e-30) void wave(const double *x, double *y, int n) {
  for (int i = 0; i < n; ++i)
    y[i] = std::exp(-0.5 * x[i] * x[i]) * std::cos(4.0 * x[i]);
}

int main(int argc, char **argv) {
  int n = argc > 1 ? atoi(argv[1]) : 1024;
  double *x = (double *)malloc(n * sizeof(double));
  double *y = (double *)malloc(n * sizeof(double));
  for (int i = 0; i < n; ++i)
    x[i] = -2.0 + 4.0 * i / (n - 1);
  wave(x, y, n);
  printf("%.17g\n", y[n / 3]);
}

// CHECK: No solution found that meets accuracy tolerance 1.000000e-30!
// CHECK: [poseidon] no rewrite applied for preprocess__Z4wavePKdPdi_poseidon_body; calling original body _Z4wavePKdPdi_poseidon_body (bit-transparent no-op)
