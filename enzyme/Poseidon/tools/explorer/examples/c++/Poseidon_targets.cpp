// Each function carries its own accuracy target with POSEIDON_OPTIMIZE_TAU. The
// same wave packet is computed for a plot, where 1e-4 is plenty and single
// precision is chosen, and for a checksum, where 1e-14 keeps it as written.
#include "poseidon/poseidon.h"
#include <cmath>
#include <cstdio>

POSEIDON_OPTIMIZE_TAU(1e-4) void plot(const double *x, double *y, int n) {
  for (int i = 0; i < n; ++i)
    y[i] = std::exp(-0.5 * x[i] * x[i]) * std::cos(4.0 * x[i]);
}

POSEIDON_OPTIMIZE_TAU(1e-14) void checksum(const double *x, double *y, int n) {
  for (int i = 0; i < n; ++i)
    y[i] = std::exp(-0.5 * x[i] * x[i]) * std::cos(4.0 * x[i]);
}

void as_written(const double *x, double *y, int n) {
  for (int i = 0; i < n; ++i)
    y[i] = std::exp(-0.5 * x[i] * x[i]) * std::cos(4.0 * x[i]);
}

int main() {
  const int n = 1024;
  static double x[n], p[n], c[n], ref[n];
  for (int i = 0; i < n; ++i)
    x[i] = -2.0 + 4.0 * i / (n - 1);
  as_written(x, ref, n);
  plot(x, p, n);
  checksum(x, c, n);
  printf("%-8s %-24s %-14s %s\n", "x", "as written", "Poseidon plot",
         "Poseidon checksum");
  const int show[] = {0, 300, 512, 900};
  for (int i : show)
    printf("%-8.3f %-24.17g %-14.1e %.1e\n", x[i], ref[i],
           std::fabs(p[i] - ref[i]) / std::fabs(ref[i]),
           std::fabs(c[i] - ref[i]) / std::fabs(ref[i]));
}
