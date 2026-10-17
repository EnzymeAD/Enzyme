// LULESH's EOS compression 1/v - 1 cancels for a relative volume v near 1.
#include "poseidon/poseidon.h"
#include <cmath>
#include <cstdio>

template <typename R, typename... T> R __poseidon_fp_optimize(void *, T...);
extern int enzyme_dup;

__attribute__((noinline)) void compression(const double *v, double *comp, int n) {
  for (int i = 0; i < n; ++i)
    comp[i] = 1.0 / v[i] - 1.0;
}

int main() {
  const int n = 1000;
  static double v[n], comp[n], dv[n], dcomp[n], ref[n];
  for (int i = 0; i < n; ++i) {
    v[i] = 1.0 - (i + 1) * 1e-12;
    dcomp[i] = 1.0;
  }
  compression(v, ref, n);
  __poseidon_fp_optimize<void>((void *)compression, enzyme_dup, v, dv, enzyme_dup, comp, dcomp, n);
  printf("%-10s %-24s %-24s %s\n", "1 - v", "as written", "Poseidon", "exact");
  const int show[] = {0, 9, 99, 999};
  for (int i : show)
    printf("%-10.3g %-24.17g %-24.17g %.17Lg\n", 1.0 - v[i], ref[i], comp[i], (1.0L - v[i]) / v[i]);
}
