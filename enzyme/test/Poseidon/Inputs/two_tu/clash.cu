// One static kernel name in two units (compiled with -DVARIANT=1 and 2), with
// different bodies.
#include "poseidon/poseidon.h"
#include <cuda_runtime.h>

POSEIDON_OPTIMIZE __global__ static void step(double *x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
#if VARIANT == 1
    x[i] = x[i] * x[i] + 1.0;
#else
    x[i] = x[i] / (x[i] + 2.0);
#endif
}

#if VARIANT == 1
void run_1(double *x, int n) { step<<<1, 128>>>(x, n); }
#else
void run_2(double *x, int n) { step<<<1, 128>>>(x, n); }
#endif
