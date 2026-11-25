// REQUIRES: enzyme
// RUN: rm -rf %t && mkdir %t
// RUN: %poseidon_clangxx -O2 -poseidon-profile-generate=%t/named -poseidon-cache=%t/c %s -o %t/gen
// RUN: cd %t && env -u POSEIDON_PROFILE_DIR %t/gen
// RUN: ls %t/named | FileCheck --check-prefix=PROFILE %s
// RUN: not ls %t/poseidon.profile
// RUN: env POSEIDON_PROFILE_DIR=%t/env %t/gen
// RUN: ls %t/env | FileCheck --check-prefix=PROFILE %s
// RUN: %poseidon_clangxx -O2 -poseidon-profile-use=%t/named -poseidon-tau=1e-6 -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-enable-herbie=0 -poseidon-cache=%t/c -poseidon-print -c %s -o %t/use.o 2>&1 | FileCheck --check-prefix=USE %s
// RUN: %poseidon_clangxx -O2 -poseidon-profile-generate -poseidon-cache=%t/c -S -emit-llvm %s -o - | FileCheck --check-prefix=DEFAULT %s

// A program built with -poseidon-profile-generate=<dir> writes its profiles
// there, unless POSEIDON_PROFILE_DIR names another directory when it runs, as
// clang's -fprofile-generate=<dir> and LLVM_PROFILE_FILE do. Built without a
// directory, it defines none and writes to the default.

#include "poseidon/poseidon.h"
#include <cmath>
#include <cstdio>
#include <vector>

POSEIDON_OPTIMIZE void wave(const double *x, double *y, int n) {
  for (int i = 0; i < n; ++i)
    y[i] = std::sin(x[i]) * x[i] + 1.0;
}

int main() {
  std::vector<double> x(64), y(64);
  for (int i = 0; i < 64; ++i)
    x[i] = i / 16.0;
  wave(x.data(), y.data(), 64);
  printf("%.6f\n", y[40]);
}

// PROFILE: preprocess__Z4wavePKdPdi_poseidon_body.fpprofile

// USE: [poseidon] Optimizing preprocess__Z4wavePKdPdi_poseidon_body with relative error tolerance: 1.000000e-06

// DEFAULT-NOT: __poseidon_profile_dir
