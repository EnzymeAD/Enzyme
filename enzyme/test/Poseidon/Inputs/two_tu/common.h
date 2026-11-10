#pragma once
#include "poseidon/poseidon.h"
#include <cuda_runtime.h>

struct mat3 {
  float a[3][3];
};

__device__ inline float det3(const mat3 &m) {
  return m.a[0][0] * (m.a[1][1] * m.a[2][2] - m.a[1][2] * m.a[2][1]) -
         m.a[0][1] * (m.a[1][0] * m.a[2][2] - m.a[1][2] * m.a[2][0]) +
         m.a[0][2] * (m.a[1][0] * m.a[2][1] - m.a[1][1] * m.a[2][0]);
}

__device__ inline void hooke_vol(const mat3 &dudx, float D1, float *out) {
  mat3 F;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      F.a[i][j] = (i == j ? 1.0f : 0.0f) + dudx.a[i][j];
  float J = det3(F);
  float p = -2.0f * D1 * J * (J - 1.0f);
  *out = -(p / J);
}

// A device call site in an inline function both translation units use: each
// device module has its own copy of the site.
__device__ inline float vol_at(const float *g, float D1, int i) {
  mat3 dudx;
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      dudx.a[r][c] = g[9 * i + 3 * r + c];
  float out;
  __poseidon_optimize_tau(1e-6, hooke_vol, dudx, D1, &out);
  return out;
}

// mfem::forall's shape: one kernel template, instantiated in every translation
// unit with the by-value lambda it runs.
template <typename BODY>
__global__ static void CuKernel1D(const int N, BODY body) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= N)
    return;
  body(k);
}

template <typename BODY> void forall(int N, BODY &&body) {
  CuKernel1D<<<(N + 127) / 128, 128>>>(N, body);
}

void run_a(double *o, double *w, const double *x, const double *y,
           const float *g, float *v, int n, int m);
void run_b(double *o, double *w, const double *x, const float *g, float *v,
           int n, int m);
