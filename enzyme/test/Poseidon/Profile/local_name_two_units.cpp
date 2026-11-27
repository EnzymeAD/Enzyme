// REQUIRES: enzyme
// DEFINE: %{gen} = %poseidon_clangxx -O2 -poseidon-profile-generate -poseidon-cache=%t/c
// DEFINE: %{use} = %poseidon_clangxx -O2 -poseidon-profile-use=%t/prof -poseidon-early-prune -poseidon-tau=1e-6 -poseidon-cost-model=%host_cost_model -poseidon-enable-algebraic=0 -poseidon-cache=%t/c -poseidon-explain=%t/explain -poseidon-print
// RUN: rm -rf %t && split-file %s %t
// RUN: %{gen} -c %t/a.cpp -o %t/a.o
// RUN: %{gen} -c %t/b.cpp -o %t/b.o
// RUN: %{gen} %t/main.cpp %t/a.o %t/b.o -o %t/gen
// RUN: env POSEIDON_PROFILE_DIR=%t/prof %t/gen | FileCheck --check-prefix=OUT %s
// RUN: ls %t/prof | FileCheck --check-prefix=FILES %s
// RUN: %{use} -c %t/a.cpp -o %t/ua.o 2>&1 | FileCheck --check-prefix=USE --implicit-check-not=error: %s
// RUN: %{use} -c %t/b.cpp -o %t/ub.o 2>&1 | FileCheck --check-prefix=USE --implicit-check-not=error: %s
// RUN: ls %t/explain | FileCheck --check-prefix=EXPLAIN %s

// Two units each define a different static site function named step: each is
// known by its file and name (as LLVM's PGO knows it), so each has a profile
// and an explanation of its own, named with the GUID of that identity.

// OUT: 2.523682 0.381643
// FILES-COUNT-2: _ZL4stepPdi~{{[0-9a-f]+}}.fpprofile
// FILES-NOT: fpprofile
// USE: [poseidon] Optimizing preprocess__ZL4stepPdi_poseidon_body with relative error tolerance: 1.000000e-06
// EXPLAIN-COUNT-2: _ZL4stepPdi~{{[0-9a-f]+}}.explain.json
// EXPLAIN-NOT: explain.json

//--- main.cpp
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

//--- a.cpp
#include "poseidon/poseidon.h"

POSEIDON_OPTIMIZE static void step(double *x, int n) {
  for (int i = 0; i < n; ++i)
    x[i] = x[i] * x[i] + 1.0;
}

void run_a(double *x, int n) { step(x, n); }

//--- b.cpp
#include "poseidon/poseidon.h"

POSEIDON_OPTIMIZE static void step(double *x, int n) {
  for (int i = 0; i < n; ++i)
    x[i] = x[i] / (x[i] + 2.0);
}

void run_b(double *x, int n) { step(x, n); }
