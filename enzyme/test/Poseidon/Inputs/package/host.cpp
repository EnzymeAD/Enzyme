#include "poseidon/poseidon.h"
#include <cstdio>

POSEIDON_OPTIMIZE void wave(const double *x, double *y, int n) {
  for (int i = 0; i < n; ++i)
    y[i] = 0.5 * x[i] * x[i] + x[i] - 1.0;
}

int main() {
  const int n = 256;
  double x[n], y[n];
  for (int i = 0; i < n; ++i)
    x[i] = 0.01 * i;
  wave(x, y, n);
  double s = 0;
  for (int i = 0; i < n; ++i)
    s += y[i];
  printf("wave %.6g\n", s);
  return 0;
}
