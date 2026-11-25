// REQUIRES: enzyme
// DEFINE: %{gen} = %poseidon_clangxx -O2 -poseidon-profile-generate -poseidon-cache=%t/c
// DEFINE: %{use} = %poseidon_clangxx -O2 -poseidon-profile-use=%t/prof -poseidon-tau=1e-6 -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-enable-herbie=0 -poseidon-cache=%t/c -poseidon-explain=%t/explain -poseidon-print
// RUN: rm -rf %t && mkdir %t
// RUN: %{gen} -c %S/../Inputs/local_name/a.cpp -o %t/a.o
// RUN: %{gen} -c %S/../Inputs/local_name/b.cpp -o %t/b.o
// RUN: %{gen} %s %t/a.o %t/b.o -o %t/gen
// RUN: env POSEIDON_PROFILE_DIR=%t/prof %t/gen | FileCheck --check-prefix=OUT %s
// RUN: ls %t/prof | FileCheck --check-prefix=FILES %s
// RUN: %{use} -c %S/../Inputs/local_name/a.cpp -o %t/ua.o 2>&1 | FileCheck --check-prefix=USE --implicit-check-not=error: %s
// RUN: %{use} -c %S/../Inputs/local_name/b.cpp -o %t/ub.o 2>&1 | FileCheck --check-prefix=USE --implicit-check-not=error: %s
// RUN: ls %t/explain | FileCheck --check-prefix=EXPLAIN %s

// Two units each define a static site function named step, with different
// bodies. A static function is known by its source file and its name, as LLVM's
// PGO knows it, so each unit's site has a profile of its own, named after the
// function and the GUID of that identity, and profile use of each unit finds
// its own and explains its solve in a file of its own.

#include <cstdio>
#include <vector>

void run_a(double *x, int n);
void run_b(double *x, int n);

int main() {
  std::vector<double> x(64), y(64);
  for (int i = 0; i < 64; ++i)
    x[i] = y[i] = 0.25 + i / 64.0;
  run_a(x.data(), 64);
  run_b(y.data(), 64);
  printf("%.6f %.6f\n", x[63], y[63]);
}

// OUT: 2.523682 0.381643

// FILES-COUNT-2: preprocess__ZL4stepPdi_poseidon_body~{{[0-9a-f]+}}.fpprofile
// FILES-NOT: fpprofile

// USE: [poseidon] Optimizing preprocess__ZL4stepPdi_poseidon_body with relative error tolerance: 1.000000e-06

// EXPLAIN-COUNT-2: preprocess__ZL4stepPdi_poseidon_body~{{[0-9a-f]+}}.explain.json
// EXPLAIN-NOT: explain.json
