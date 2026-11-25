// REQUIRES: enzyme
// RUN: rm -rf %t && mkdir %t
// RUN: %poseidon_clangxx -O2 -poseidon-profile-generate -poseidon-cache=%t/c %s -o %t/a
// RUN: %poseidon_clangxx -O2 -poseidon-profile-generate -poseidon-cache=%t/c -DPROGRAM_B %s -o %t/b
// RUN: env POSEIDON_PROBE=1 POSEIDON_PROBE_LADDER=1e-6 POSEIDON_PROFILE_DIR=%t/prof %t/a > /dev/null
// RUN: cp %t/prof/preprocess__Z1fPKdPdi_poseidon_body.fpprofile %t/f.a
// RUN: env POSEIDON_PROBE=1 POSEIDON_PROBE_LADDER=1e-6 POSEIDON_PROFILE_DIR=%t/prof %t/b > /dev/null 2> %t/b.err
// RUN: cmp %t/f.a %t/prof/preprocess__Z1fPKdPdi_poseidon_body.fpprofile
// RUN: FileCheck --check-prefix=MEASURED %s < %t/prof/preprocess__Z1fPKdPdi_poseidon_body.fpprofile
// RUN: FileCheck --check-prefix=MEASURED %s < %t/prof/preprocess__Z1gPKdPdi_poseidon_body.fpprofile
// RUN: FileCheck --check-prefix=PROBE %s < %t/b.err
// RUN: env POSEIDON_PROBE=0 POSEIDON_PROFILE_DIR=%t/declined %t/a > /dev/null
// RUN: env POSEIDON_PROBE=1 POSEIDON_PROBE_LADDER=1e-6 POSEIDON_PROFILE_DIR=%t/declined %t/b > /dev/null
// RUN: FileCheck --check-prefix=NOKAPPA %s < %t/declined/preprocess__Z1fPKdPdi_poseidon_body.fpprofile
// RUN: FileCheck --check-prefix=MEASURED %s < %t/declined/preprocess__Z1gPKdPdi_poseidon_body.fpprofile
// RUN: env POSEIDON_PROFILE_DIR=%t/only %t/b > /dev/null
// RUN: env POSEIDON_PROFILE_DIR=%t/only %t/b > /dev/null
// RUN: FileCheck --check-prefix=NEVER %s < %t/only/preprocess__Z1fPKdPdi_poseidon_body.fpprofile

// Two programs share the sites f, g and h and write one profile directory:
// program a executes f and h, program b g and h. The run of b keeps the
// profile of f that the run of a wrote, executions and condition number or
// its lack of one, and measures its own sites only. A site no run executed is
// written as never executed.

#include "poseidon/poseidon.h"
#include <cmath>
#include <cstdio>
#include <vector>

POSEIDON_OPTIMIZE void f(const double *x, double *y, int n) {
  for (int i = 0; i < n; ++i)
    y[i] = std::exp(-0.5 * x[i] * x[i]) * std::cos(4.0 * x[i]);
}

POSEIDON_OPTIMIZE void g(const double *x, double *y, int n) {
  for (int i = 0; i < n; ++i)
    y[i] = std::sin(x[i]) * x[i] + 1.0;
}

POSEIDON_OPTIMIZE void h(const double *x, double *y, int n) {
  for (int i = 0; i < n; ++i)
    y[i] = x[i] * x[i] - 0.5 * x[i];
}

int main() {
  const int n = 256;
  std::vector<double> x(n), y(n), z(n);
  for (int i = 0; i < n; ++i)
    x[i] = -2.0 + 4.0 * i / (n - 1);
#ifdef PROGRAM_B
  g(x.data(), y.data(), n);
#else
  f(x.data(), y.data(), n);
#endif
  h(x.data(), z.data(), n);
  poseidon_metric("y", y[n / 3] + z[n / 3]);
  printf("%.17g\n", y[n / 3] + z[n / 3]);
}

// MEASURED: SiteId = {{[0-9]+}}
// MEASURED: Kappa =
// MEASURED-NOT: Executed
// MEASURED: MinRes =

// PROBE: [probe] measuring the condition numbers of metric "y" at 2 sites: 3 more runs of this program

// NOKAPPA-NOT: Kappa
// NOKAPPA: MinRes =

// NEVER: SiteId = {{[0-9]+}}
// NEVER: CanonicalHash = {{[0-9a-f]+}}
// NEVER-NEXT: Executed = 0
// NEVER-NOT: {{.}}
