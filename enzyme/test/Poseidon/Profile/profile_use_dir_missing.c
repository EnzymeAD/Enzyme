// RUN: rm -rf %t && mkdir -p %t/tmp
// RUN: not env TMPDIR=%t/tmp %clang -x c -O1 -c -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%t/none -mllvm -poseidon-cost-model=%cost_models/cm_cpu_native.csv %s -o %t/site.o 2>&1 | FileCheck %s --implicit-check-not="error in backend" --implicit-check-not="Stack dump" --implicit-check-not=PLEASE
// RUN: not ls %t/site.o
// RUN: ls %t/tmp | count 0

// A profile directory no profiling run wrote is a mistake in the compile's
// input: an error at the site, exit status 1, no object, and none of clang's
// crash reporting.

#include "poseidon/poseidon.h"

POSEIDON_OPTIMIZE double f(double x, double y) { return (x + y) * x; }

// CHECK: profile_use_dir_missing.c:[[@LINE-2]]:{{[0-9]+}}: error: Poseidon: the profile directory {{.*}}none (-poseidon-profile-use) does not exist; no profiling run wrote into it. Run the program built with -poseidon-profile-generate first, or pass the directory it wrote
