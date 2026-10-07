// The same wave packet with a 1e-10 accuracy target at the call: single precision is refused.
#include "poseidon/poseidon.h"
#include <cmath>
#include <cstdio>

template <typename R, typename... T> R __poseidon_fp_optimize(void *, T...);
extern int enzyme_dup, poseidon_tau;

__attribute__((noinline)) void wave(const double *x, double *y, int n) {
  for (int i = 0; i < n; ++i)
    y[i] = std::exp(-0.5 * x[i] * x[i]) * std::cos(4.0 * x[i]);
}

int main() {
  const int n = 1024;
  static double x[n], y[n], dx[n], dy[n], ref[n];
  for (int i = 0; i < n; ++i) {
    x[i] = -2.0 + 4.0 * i / (n - 1);
    dy[i] = 1.0;
  }
  wave(x, ref, n);
  __poseidon_fp_optimize<void>((void *)wave, poseidon_tau, 1e-10, enzyme_dup, x, dx, enzyme_dup, y, dy, n);
  printf("%-8s %-24s %-24s %s\n", "x", "as written", "Poseidon", "rel. diff");
  const int show[] = {0, 300, 512, 900};
  for (int i : show)
    printf("%-8.3f %-24.17g %-24.17g %.1e\n", x[i], ref[i], y[i], std::fabs(y[i] - ref[i]) / std::fabs(ref[i]));
}
