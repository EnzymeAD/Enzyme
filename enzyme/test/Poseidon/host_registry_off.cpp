// RUN: rm -rf %t && mkdir -p %t
// RUN: %poseidon_clangxx -O2 -poseidon-profile-generate -poseidon-cache=%t/c %s -o %t/gen 2>/dev/null
// RUN: llvm-nm %t/gen | FileCheck %s
// RUN: env POSEIDON_PROFILE_DIR=%t/prof %t/gen
// RUN: ls %t/prof/preprocess__Z5scalePKdPdi.fpprofile
// REQUIRES: poseidon, enzyme

// A profiling build without a POSEIDON_OPTIMIZE function on the host links the
// allocator the host sites need, but without the definition a host site emits
// it only passes allocations through: nothing is recorded.

#include <cstdlib>

template <typename R, typename... T> R __poseidon_fp_optimize(void *, T...);
extern int enzyme_dup;

__attribute__((noinline)) void scale(const double *x, double *y, int n) {
  for (int i = 0; i < n; ++i)
    y[i] = x[i] * x[i] + x[i];
}

int main() {
  const int n = 64;
  double *x = (double *)malloc(n * sizeof(double));
  double *y = (double *)malloc(n * sizeof(double));
  double *dx = (double *)calloc(n, sizeof(double));
  double *dy = (double *)malloc(n * sizeof(double));
  for (int i = 0; i < n; ++i) {
    x[i] = 0.01 * i;
    dy[i] = 1.0;
  }
  __poseidon_fp_optimize<void>((void *)scale, enzyme_dup, x, dx, enzyme_dup, y, dy, n);
  return y[3] > 1.0;
}

// CHECK-DAG: {{^ +}}w __poseidon_host_sites_present
// CHECK-DAG: T malloc
