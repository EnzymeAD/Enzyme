// RUN: rm -rf %t && mkdir -p %t
// RUN: %poseidon_clangxx -O2 -poseidon-profile-generate -poseidon-cache=%t/c %s -o %t/gen 2>/dev/null
// RUN: env POSEIDON_PROFILE_DIR=%t/prof %t/gen > /dev/null
// RUN: grep -E '^[0-9]+$' %t/prof/preprocess__Z4wavePKdPdi_poseidon_body.fpprofile > %t/idx
// RUN: FileCheck %s < %t/idx
// RUN: sort -n -c %t/idx
// REQUIRES: poseidon, enzyme

// The host profiler writes a site's records in slot order, whatever order the
// run first reached them in: the profile's bytes are part of the key of the
// site's cached DP table.

#include "poseidon/poseidon.h"
#include <cmath>
#include <cstdio>
#include <vector>

POSEIDON_OPTIMIZE void wave(const double *x, double *y, int n) {
  for (int i = 0; i < n; ++i)
    y[i] = std::exp(-0.5 * x[i] * x[i]) * std::cos(4.0 * x[i]) + 1.0;
}

int main() {
  const int n = 64;
  std::vector<double> x(n), y(n);
  for (int i = 0; i < n; ++i)
    x[i] = -2.0 + 4.0 * i / (n - 1);
  wave(x.data(), y.data(), n);
  printf("%.17g\n", y[n / 3]);
}

// CHECK: {{^}}0{{$}}
// CHECK-NEXT: {{^}}1{{$}}
// CHECK-NEXT: {{^}}2{{$}}
