// One static site name in two units (a.cpp, b.cpp), with different bodies.
#include "poseidon/poseidon.h"

POSEIDON_OPTIMIZE static void step(double *x, int n) {
  for (int i = 0; i < n; ++i)
    x[i] = x[i] * x[i] + 1.0;
}

void run_a(double *x, int n) { step(x, n); }
