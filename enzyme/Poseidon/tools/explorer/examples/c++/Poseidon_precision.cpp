// A Gaussian wave packet: at the default -poseidon-tau=1e-6 single precision is
// accurate enough.
#include "poseidon/poseidon.h"
#include <cmath>
#include <cstdio>

POSEIDON_OPTIMIZE void wave(const double *x, double *y, int n) {
  for (int i = 0; i < n; ++i)
    y[i] = std::exp(-0.5 * x[i] * x[i]) * std::cos(4.0 * x[i]);
}

void as_written(const double *x, double *y, int n) {
  for (int i = 0; i < n; ++i)
    y[i] = std::exp(-0.5 * x[i] * x[i]) * std::cos(4.0 * x[i]);
}

int main() {
  const int n = 1024;
  static double x[n], y[n], ref[n];
  for (int i = 0; i < n; ++i)
    x[i] = -2.0 + 4.0 * i / (n - 1);
  as_written(x, ref, n);
  wave(x, y, n);
  printf("%-8s %-24s %-24s %s\n", "x", "as written", "Poseidon", "rel. diff");
  const int show[] = {0, 300, 512, 900};
  for (int i : show)
    printf("%-8.3f %-24.17g %-24.17g %.1e\n", x[i], ref[i], y[i],
           std::fabs(y[i] - ref[i]) / std::fabs(ref[i]));
}
