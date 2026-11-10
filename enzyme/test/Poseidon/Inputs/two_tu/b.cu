#include "common.h"
#include <cmath>

POSEIDON_OPTIMIZE __global__ void lift(double *o, const double *x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double t = x[i];
    o[i] = sqrt(t * t + 1.0) - t;
  }
}

__global__ void vol_b(const float *g, float D1, float *res, int m) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < m)
    res[i] = vol_at(g, D1, m - 1 - i);
}

void run_b(double *o, double *w, const double *x, const float *g, float *v,
           int n, int m) {
  lift<<<(n + 255) / 256, 256>>>(o, x, n);
  vol_b<<<(m + 63) / 64, 64>>>(g, 0.5f, v, m);
  forall(n, [=] __host__ __device__ POSEIDON_OPTIMIZE_TAU(1e-12)(int i) {
    double t = x[i];
    w[i] = t / (1.0 + t * t) - 0.5 * t;
  });
}
