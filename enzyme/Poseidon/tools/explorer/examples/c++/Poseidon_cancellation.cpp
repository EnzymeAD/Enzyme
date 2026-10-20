// Poseidon profiles every call of a function marked POSEIDON_OPTIMIZE, then
// rewrites it to the cheapest form whose modelled error meets -poseidon-tau. In
// the assembly, kernel is the rewritten function and as_written the same loop
// without the marker; the solver's decisions are in the compiler output.
#include "poseidon/poseidon.h"
#include <cmath>
#include <cstdio>

// (1 - cos x) / x^2 loses its digits to cancellation as x -> 0.
POSEIDON_OPTIMIZE void kernel(const double *x, double *out, int n) {
  for (int i = 0; i < n; ++i)
    out[i] = (1.0 - std::cos(x[i])) / (x[i] * x[i]);
}

void as_written(const double *x, double *out, int n) {
  for (int i = 0; i < n; ++i)
    out[i] = (1.0 - std::cos(x[i])) / (x[i] * x[i]);
}

int main() {
  const int n = 1024;
  static double x[n], out[n], ref[n];
  for (int i = 0; i < n; ++i)
    x[i] = std::pow(10.0, -8.0 + 4.0 * i / (n - 1));
  as_written(x, ref, n);
  kernel(x, out, n);
  printf("%-10s %-24s %s\n", "x", "as written", "Poseidon");
  const int show[] = {0, n / 4, n / 2, n - 1};
  for (int i : show)
    printf("%-10.3g %-24.17g %.17g\n", x[i], ref[i], out[i]);
}
