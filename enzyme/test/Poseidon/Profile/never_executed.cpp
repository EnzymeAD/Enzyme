// REQUIRES: enzyme
// RUN: rm -rf %t && mkdir %t
// RUN: %poseidon_clangxx -O2 -poseidon-profile-generate=%t/prof -poseidon-cache=%t/c %s -o %t/gen
// RUN: cd %t && env -u POSEIDON_PROFILE_DIR %t/gen > /dev/null 2> %t/gen.err
// RUN: not ls %t/poseidon.profile
// RUN: env POSEIDON_PROFILE_DIR=%t/env %t/gen > /dev/null 2>&1
// RUN: ls %t/env | FileCheck --check-prefix=DIR %s
// RUN: %poseidon_clangxx -O2 -poseidon-profile-generate -poseidon-cache=%t/c -S -emit-llvm %s -o - | FileCheck --check-prefix=DEFAULT %s
// RUN: FileCheck --check-prefix=NEVER %s < %t/prof/_Z7skippedPKdPdi.fpprofile
// RUN: FileCheck --check-prefix=RAN %s < %t/prof/_Z3ranPKdPdi.fpprofile
// RUN: grep -E '^[0-9]+$' %t/prof/_Z3ranPKdPdi.fpprofile > %t/idx
// RUN: FileCheck --check-prefix=ORDER %s < %t/idx
// RUN: sort -n -c %t/idx
// RUN: FileCheck --check-prefix=NOKAPPA --allow-empty %s < %t/gen.err
// RUN: %poseidon_clangxx -O2 -poseidon-profile-use=%t/prof -poseidon-early-prune -poseidon-tau=1e-6 -poseidon-cost-model=%host_cost_model -poseidon-enable-rewrite=0 -poseidon-cache=%t/c -c %s -poseidon-print -o %t/use.o 2>&1 | FileCheck --check-prefix=USE --implicit-check-not=warning %s
// RUN: %poseidon_clangxx -poseidon-profile-use=%t/prof -poseidon-early-prune %t/use.o -o %t/use
// RUN: mkdir %t/deployed && cd %t/deployed && %t/use
// RUN: ls -A %t/deployed | count 0

// -poseidon-profile-generate=<dir> names where profiles go unless
// POSEIDON_PROFILE_DIR overrides it, as -fprofile-generate=<dir> and
// LLVM_PROFILE_FILE do; without a directory the build defines none.
// A site the run never executed still gets a profile (header and
// Executed = 0), left as written by profile use with a note, and is no site
// for a kappa run. Records are in slot order (the profile's bytes key the DP
// cache). The optimized program links the runtime for poseidon_metric and
// writes no profile.

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

// DIR-DAG: _Z3ranPKdPdi.fpprofile
// DIR-DAG: _Z7skippedPKdPdi.fpprofile
// DEFAULT-NOT: __poseidon_profile_dir
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
// NOKAPPA-NOT: [poseidon]
// USE-DAG: [poseidon] _Z7skippedPKdPdi_poseidon_body: the profiling run instrumented this site and never executed it ({{.*}}/prof/_Z7skippedPKdPdi.fpprofile); left unchanged
// USE-DAG: preprocess__Z3ranPKdPdi_poseidon_body: tau=1.000000e-06
