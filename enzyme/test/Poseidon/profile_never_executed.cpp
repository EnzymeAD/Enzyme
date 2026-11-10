// RUN: rm -rf %t && mkdir -p %t
// RUN: %poseidon_clangxx -O2 -poseidon-profile-generate -poseidon-cache=%t/c %s -o %t/gen 2>/dev/null
// RUN: env POSEIDON_PROFILE_DIR=%t/prof %t/gen > /dev/null 2> %t/gen.err
// RUN: FileCheck --check-prefix=NEVER %s < %t/prof/preprocess__Z7skippedPKdPdi_poseidon_body.fpprofile
// RUN: FileCheck --check-prefix=RAN %s < %t/prof/preprocess__Z3ranPKdPdi_poseidon_body.fpprofile
// RUN: FileCheck --check-prefix=NOPROBE --allow-empty %s < %t/gen.err
// RUN: env POSEIDON_PROFILE_DIR=%t/flush %t/gen flush > /dev/null
// RUN: FileCheck --check-prefix=NEVER %s < %t/flush/preprocess__Z7skippedPKdPdi_poseidon_body.fpprofile
// RUN: FileCheck --check-prefix=RAN %s < %t/flush/preprocess__Z3ranPKdPdi_poseidon_body.fpprofile
// RUN: %poseidon_clangxx -O2 -poseidon-profile-use=%t/prof -poseidon-tau=1e-6 -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-enable-herbie=0 -poseidon-cache=%t/c -c %s -o %t/use.o 2> %t/use.err
// RUN: FileCheck --check-prefix=USE --implicit-check-not=warning %s < %t/use.err
// REQUIRES: poseidon, enzyme

// The profiling runtime writes a profile for every site it instrumented, run or
// not: one the run never executed gets its header and a line saying so, at
// exit and from ProfilerWrite alike, and profile use leaves it as written with
// a note. A site that never ran is no site for the condition-number probe,
// which needs two.

#include "poseidon/poseidon.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <vector>

extern "C" void ProfilerWrite();

POSEIDON_OPTIMIZE void ran(const double *x, double *y, int n) {
  for (int i = 0; i < n; ++i)
    y[i] = std::exp(-0.5 * x[i] * x[i]) * std::cos(4.0 * x[i]);
}

POSEIDON_OPTIMIZE void skipped(const double *x, double *y, int n) {
  for (int i = 0; i < n; ++i)
    y[i] = std::sin(x[i]) * x[i] + 1.0;
}

int main(int argc, char **argv) {
  const int n = 256;
  std::vector<double> x(n), y(n);
  for (int i = 0; i < n; ++i)
    x[i] = -2.0 + 4.0 * i / (n - 1);
  ran(x.data(), y.data(), n);
  if (argc > 5)
    skipped(x.data(), y.data(), n);
  poseidon_metric("y", y[n / 3]);
  printf("%.17g\n", y[n / 3]);
  if (argc > 1 && !strcmp(argv[1], "flush")) {
    ProfilerWrite();
    _exit(0);
  }
}

// NEVER: SiteId = {{[0-9]+}}
// NEVER: CanonicalHash = {{[0-9a-f]+}}
// NEVER-NEXT: Executed = 0
// NEVER-NOT: {{.}}

// RAN: SiteId = {{[0-9]+}}
// RAN: CanonicalHash = {{[0-9a-f]+}}
// RAN-NOT: Executed
// RAN: MinRes =

// NOPROBE-NOT: [probe]

// USE-DAG: [poseidon] _Z7skippedPKdPdi_poseidon_body: the profiling run instrumented this site and never executed it ({{.*}}/prof/preprocess__Z7skippedPKdPdi_poseidon_body.fpprofile); left unchanged
// USE-DAG: preprocess__Z3ranPKdPdi_poseidon_body: tau=1.000000e-06
