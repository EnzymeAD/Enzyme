// One step of attitude integration: four quaternion components solved as one FP subgraph.
#include "poseidon/poseidon.h"
#include <cmath>
#include <cstdio>

template <typename R, typename... T> R __poseidon_fp_optimize(void *, T...);
extern int enzyme_dup;

__attribute__((noinline)) void rotation(const double *w, double *q, int n) {
  for (int i = 0; i < n; ++i) {
    double x = w[3 * i], y = w[3 * i + 1], z = w[3 * i + 2];
    double angle = std::sqrt(x * x + y * y + z * z);
    double s = std::sin(0.5 * angle) / angle;
    q[4 * i] = std::cos(0.5 * angle);
    q[4 * i + 1] = s * x;
    q[4 * i + 2] = s * y;
    q[4 * i + 3] = s * z;
  }
}

int main() {
  const int n = 1000;
  static double w[3 * n], q[4 * n], dw[3 * n], dq[4 * n], ref[4 * n];
  for (int i = 0; i < n; ++i) {
    w[3 * i] = 1e-3 * (i + 1);
    w[3 * i + 1] = -0.05 * (i % 10 + 1);
    w[3 * i + 2] = 0.02 * (i % 3 + 1);
  }
  for (int i = 0; i < 4 * n; ++i)
    dq[i] = 1.0;
  rotation(w, ref, n);
  __poseidon_fp_optimize<void>((void *)rotation, enzyme_dup, w, dw, enzyme_dup, q, dq, n);
  printf("%-4s %-24s %s\n", "q", "as written", "Poseidon");
  const char *name[] = {"w", "x", "y", "z"};
  for (int k = 0; k < 4; ++k)
    printf("%-4s %-24.17g %.17g\n", name[k], ref[4 * 999 + k], q[4 * 999 + k]);
}
