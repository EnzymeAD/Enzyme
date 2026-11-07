// __poseidon_optimize makes one call the site: each call has its own profile,
// target and rewritten copy. The two calls below see different ranges of x and
// get different rewrites; the plain call runs the function as written.
#include "poseidon/poseidon.h"
#include <cmath>
#include <cstdio>

// (1 - cos x) / x^2: cancellation for small x, harmless for large x.
void kernel(const double *x, double *out, int n) {
  for (int i = 0; i < n; ++i)
    out[i] = (1.0 - std::cos(x[i])) / (x[i] * x[i]);
}

int main() {
  const int n = 1024;
  static double small[n], large[n], a[n], b[n], ref[n];
  for (int i = 0; i < n; ++i) {
    small[i] = std::pow(10.0, -8.0 + 4.0 * i / (n - 1));
    large[i] = 0.5 + 2.5 * i / (n - 1);
  }
  __poseidon_optimize(kernel, small, a, n);
  __poseidon_optimize(kernel, large, b, n);
  kernel(large, ref, n);
  printf("%-10s %-24s %-10s %-24s %s\n", "small x", "Poseidon 1", "large x",
         "Poseidon 2", "as written");
  const int show[] = {0, n / 2, n - 1};
  for (int i : show)
    printf("%-10.3g %-24.17g %-10.3g %-24.17g %.17g\n", small[i], a[i],
           large[i], b[i], ref[i]);
}
