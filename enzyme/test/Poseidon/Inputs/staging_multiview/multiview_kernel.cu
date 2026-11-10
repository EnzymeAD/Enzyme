// One __shared__ array used through views of two shapes, one at a constant
// offset into it, by fully unrolled contractions that read each staged value
// twice: the shape of MFEM's SmemPADiffusionApply3D.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "poseidon/poseidon.h"

__global__ POSEIDON_OPTIMIZE void multiview(const double *__restrict__ x,
                                            const double *__restrict__ b,
                                            double *__restrict__ y) {
  __shared__ double sm[2][64];
  double(*X)[4] = (double(*)[4])(sm + 1);
  double(*T)[8] = (double(*)[8])(sm + 0);
  const int tx = threadIdx.x, ty = threadIdx.y, e = blockIdx.x;
  X[ty][tx] = x[e * 16 + ty * 4 + tx];
  __syncthreads();
  double u = 0.0, v = 0.0;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    u += X[ty][k] * b[k * 4 + tx];
    v += X[ty][k] * b[16 + k * 4 + tx];
  }
  T[ty][tx] = u;
  T[ty][4 + tx] = v;
  __syncthreads();
  double w = 0.0;
#pragma unroll
  for (int k = 0; k < 4; ++k)
    w += T[k][tx] * b[ty * 4 + k] + T[k][4 + tx] * b[16 + ty * 4 + k];
  y[e * 16 + ty * 4 + tx] += w;
}

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t err = (x);                                                     \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "cuda: %s\n", cudaGetErrorString(err));                  \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

int main() {
  constexpr int NE = 8;
  double hx[NE * 16], hb[32], hy[NE * 16];
  for (int i = 0; i < NE * 16; ++i) {
    hx[i] = 1.0 + 0.01 * i;
    hy[i] = 0.0;
  }
  for (int i = 0; i < 32; ++i)
    hb[i] = 0.5 + 0.03 * i;
  double *dx, *db, *dy;
  CK(cudaMalloc(&dx, sizeof hx));
  CK(cudaMalloc(&db, sizeof hb));
  CK(cudaMalloc(&dy, sizeof hy));
  CK(cudaMemcpy(dx, hx, sizeof hx, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(db, hb, sizeof hb, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dy, hy, sizeof hy, cudaMemcpyHostToDevice));
  multiview<<<NE, dim3(4, 4)>>>(dx, db, dy);
  CK(cudaGetLastError());
  CK(cudaMemcpy(hy, dy, sizeof hy, cudaMemcpyDeviceToHost));
  double s = 0.0;
  for (int i = 0; i < NE * 16; ++i)
    s += hy[i];
  printf("sum %.17g\n", s);
  return 0;
}
