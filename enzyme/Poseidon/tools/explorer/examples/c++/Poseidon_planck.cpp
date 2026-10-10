// Planck's x / (e^x - 1) cancels at low frequency, where x = h nu / kT is small.
#include "poseidon/poseidon.h"
#include <cmath>
#include <cstdio>

template <typename R, typename... T> R __poseidon_fp_optimize(void *, T...);
extern int enzyme_dup;

__attribute__((noinline)) void planck(const double *x, double *b, int n) {
  for (int i = 0; i < n; ++i)
    b[i] = x[i] / (std::exp(x[i]) - 1.0);
}

int main() {
  const int n = 1001;
  static double x[n], b[n], dx[n], db[n], ref[n];
  for (int i = 0; i < n; ++i) {
    x[i] = std::pow(10.0, -9.0 + 0.01 * i);
    db[i] = 1.0;
  }
  planck(x, ref, n);
  __poseidon_fp_optimize<void>((void *)planck, enzyme_dup, x, dx, enzyme_dup, b, db, n);
  printf("%-8s %-24s %s\n", "x", "as written", "Poseidon");
  const int show[] = {0, 300, 600, 900, 1000};
  for (int i : show)
    printf("%-8.0e %-24.17g %.17g\n", x[i], ref[i], b[i]);
}
