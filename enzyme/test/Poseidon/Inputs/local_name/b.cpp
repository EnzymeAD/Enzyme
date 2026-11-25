// One static site name in two units (a.cpp, b.cpp), with different bodies.
#include "poseidon/poseidon.h"

POSEIDON_OPTIMIZE static void step(double *x, int n) {
  for (int i = 0; i < n; ++i)
    x[i] = x[i] / (x[i] + 2.0);
}

void run_b(double *x, int n) { step(x, n); }
