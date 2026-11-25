// REQUIRES: enzyme
// RUN: rm -rf %t && mkdir %t
// RUN: %poseidon_clangxx -O2 -poseidon-profile-generate -poseidon-cache=%t/c %s -o %t/gen
// RUN: env POSEIDON_PROFILE_DIR=%t/prof %t/gen > /dev/null 2> %t/gen.err
// RUN: FileCheck --check-prefix=NEVER %s < %t/prof/_Z7skippedPKdPdi.fpprofile
// RUN: FileCheck --check-prefix=RAN %s < %t/prof/_Z3ranPKdPdi.fpprofile
// RUN: grep -E '^[0-9]+$' %t/prof/_Z3ranPKdPdi.fpprofile > %t/idx
// RUN: FileCheck --check-prefix=ORDER %s < %t/idx
// RUN: sort -n -c %t/idx
// RUN: FileCheck --check-prefix=NOPROBE --allow-empty %s < %t/gen.err
// RUN: %poseidon_clangxx -O2 -poseidon-profile-use=%t/prof -poseidon-tau=1e-6 -poseidon-cost-model=%host_cost_model -poseidon-enable-herbie=0 -poseidon-cache=%t/c -c %s -poseidon-print -o %t/use.o 2>&1 | FileCheck --check-prefix=USE --implicit-check-not=warning %s
// RUN: %poseidon_clangxx -poseidon-profile-use=%t/prof %t/use.o -o %t/use
// RUN: mkdir %t/deployed && cd %t/deployed && %t/use
// RUN: ls -A %t/deployed | count 0

// The profiling runtime writes a profile for every site it instrumented, run or
// not: one the run never executed gets its header and a line saying so, and
// profile use leaves it as written with a note. A site that never ran is no
// site for the condition-number probe, which needs two. The records of a site
// that ran are in slot order, whatever order the run first reached them in:
// the profile's bytes are part of the key of the site's cached DP table. The
// optimized program links the runtime for its poseidon_metric call and writes
// no profile.

#include "poseidon/poseidon.h"
#include <cmath>
#include <cstdio>
#include <vector>

POSEIDON_OPTIMIZE void ran(const double *x, double *y, int n) {
  for (int i = 0; i < n; ++i)
    y[i] = std::exp(-0.5 * x[i] * x[i]) * std::cos(4.0 * x[i]) + 1.0;
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
}

// NEVER: SiteId = {{[0-9]+}}
// NEVER: CanonicalHash = {{[0-9a-f]+}}
// NEVER-NEXT: Executed = 0
// NEVER-NOT: {{.}}

// RAN: SiteId = {{[0-9]+}}
// RAN: CanonicalHash = {{[0-9a-f]+}}
// RAN-NOT: Executed
// RAN: MinRes =

// ORDER: {{^}}0{{$}}
// ORDER-NEXT: {{^}}1{{$}}
// ORDER-NEXT: {{^}}2{{$}}

// NOPROBE-NOT: [probe]

// USE-DAG: [poseidon] _Z7skippedPKdPdi_poseidon_body: the profiling run instrumented this site and never executed it ({{.*}}/prof/_Z7skippedPKdPdi.fpprofile); left unchanged
// USE-DAG: preprocess__Z3ranPKdPdi_poseidon_body: tau=1.000000e-06
