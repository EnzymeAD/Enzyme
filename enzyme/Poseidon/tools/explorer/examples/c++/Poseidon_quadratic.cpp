// The root (-b + sqrt(b^2 - 4)) / 2 of x^2 + b x + 1 cancels for large positive
// b. Herbie answers with a regime split, one series per sign of b that meets
// the 1e-6 target on the profiled range |b| >= 1e3: a branch in the assembly,
// a ternary in the equivalent C (the Poseidon source tool).
#include "poseidon/poseidon.h"
#include <cmath>
#include <cstdio>

POSEIDON_OPTIMIZE void root(const double *b, double *x, int n) {
  for (int i = 0; i < n; ++i)
    x[i] = (-b[i] + std::sqrt(b[i] * b[i] - 4.0)) / 2.0;
}

void as_written(const double *b, double *x, int n) {
  for (int i = 0; i < n; ++i)
    x[i] = (-b[i] + std::sqrt(b[i] * b[i] - 4.0)) / 2.0;
}

int main() {
  const int n = 1024;
  static double b[n], x[n], ref[n];
  for (int i = 0; i < n; ++i)
    b[i] = (i % 2 ? 1.0 : -1.0) * std::pow(10.0, 3.0 + 5.0 * i / (n - 1));
  as_written(b, ref, n);
  root(b, x, n);
  printf("%-10s %-24s %-24s %s\n", "b", "as written", "Poseidon", "exact");
  const int show[] = {0, 1, n / 2, n / 2 + 1, n - 2, n - 1};
  for (int i : show) {
    long double bl = b[i], s = std::sqrt(bl * bl - 4.0L);
    printf("%-10.3g %-24.17g %-24.17g %.17Lg\n", b[i], ref[i], x[i],
           bl > 0 ? -2.0L / (bl + s) : (s - bl) / 2.0L);
  }
}
