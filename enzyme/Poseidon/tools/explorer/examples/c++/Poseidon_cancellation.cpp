// Poseidon profiles the call wrapped in __poseidon_fp_optimize, then rewrites
// it to the cheapest form whose modelled error meets -poseidon-tau. In the
// assembly, kernel is the function as written and the rewritten call is inlined
// into main; the solver's decisions are in the compiler output.
#include "poseidon/poseidon.h"
#include <cmath>
#include <cstdio>

template <typename R, typename... T> R __poseidon_fp_optimize(void *, T...);
extern int enzyme_dup;

// (1 - cos x) / x^2 loses its digits to cancellation as x -> 0.
__attribute__((noinline)) void kernel(const double *x, double *out, int n) {
  for (int i = 0; i < n; ++i)
    out[i] = (1.0 - std::cos(x[i])) / (x[i] * x[i]);
}

int main() {
  const int n = 1024;
  static double x[n], out[n], dx[n], dout[n], ref[n];
  for (int i = 0; i < n; ++i) {
    x[i] = std::pow(10.0, -8.0 + 4.0 * i / (n - 1));
    dout[i] = 1.0;
  }
  kernel(x, ref, n);
  __poseidon_fp_optimize<void>((void *)kernel, enzyme_dup, x, dx, enzyme_dup, out, dout, n);
  printf("%-10s %-24s %s\n", "x", "as written", "Poseidon");
  const int show[] = {0, n / 4, n / 2, n - 1};
  for (int i : show)
    printf("%-10.3g %-24.17g %.17g\n", x[i], ref[i], out[i]);
}
