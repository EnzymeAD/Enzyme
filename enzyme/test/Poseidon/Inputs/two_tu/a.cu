#include "common.h"

POSEIDON_OPTIMIZE __global__ void axpb(double *o, const double *a,
                                       const double *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double x = a[i], y = b[i];
    o[i] = x * y + x;
  }
}

__global__ void vol_a(const float *g, float D1, float *res, int m) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < m)
    res[i] = vol_at(g, D1, i);
}

void run_a(double *o, double *w, const double *x, const double *y,
           const float *g, float *v, int n, int m) {
  axpb<<<(n + 127) / 128, 128>>>(o, x, y, n);
  vol_a<<<(m + 127) / 128, 128>>>(g, 1.25f, v, m);
  const double c = 0.75;
  forall(n, [=] __host__ __device__ POSEIDON_OPTIMIZE(int i) {
    double t = x[i] - y[i];
    w[i] = c * t * t + x[i];
  });
}
