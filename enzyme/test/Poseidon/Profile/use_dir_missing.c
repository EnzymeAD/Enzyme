// RUN: rm -rf %t && mkdir -p %t/tmp
// RUN: (env TMPDIR=%t/tmp %clang -x c -O1 -c -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%t/none -mllvm -poseidon-cost-model=%host_cost_model %s -o %t/site.o 2>&1; echo "exit status $?") | FileCheck %s --implicit-check-not="error in backend" --implicit-check-not="Stack dump" --implicit-check-not=PLEASE
// RUN: not ls %t/site.o
// RUN: ls %t/tmp | count 0

// A profile directory no profiling run wrote is a mistake in the compile's
// input: an error at the site, exit status 1, no object, and none of clang's
// crash reporting.

#include "poseidon/poseidon.h"

POSEIDON_OPTIMIZE double f(double x, double y) { return (x + y) * x; }

// CHECK: use_dir_missing.c:[[@LINE-2]]:26: error: Poseidon: the profile directory {{.*}}/none (-poseidon-profile-use) does not exist; no profiling run wrote into it. Run the program built with -poseidon-profile-generate first, or pass the directory it wrote
// CHECK-NEXT: POSEIDON_OPTIMIZE double f(double x, double y) { return (x + y) * x; }
// CHECK-NEXT: {{^ *\| *\^$}}
// CHECK-NEXT: {{^}}1 error generated.{{$}}
// CHECK-NEXT: exit status 1
