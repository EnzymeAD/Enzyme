// Per (element, quadrature point) in thread-strided loops whose bound is an
// argument, the inverse determinant of a 2x2 Jacobian scaled by a weight.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <poseidon/poseidon.h>

POSEIDON_OPTIMIZE __global__ void setup_kernel(int Q, const double *W,
                                               const double *J, double *D) {
  const int e = blockIdx.x;
  for (int qy = threadIdx.y; qy < Q; qy += blockDim.y)
    for (int qx = threadIdx.x; qx < Q; qx += blockDim.x) {
      const int q = qx + Q * qy;
      const double *j = J + 4 * (q + Q * Q * e);
      const double det = j[0] * j[3] - j[1] * j[2];
      D[q + Q * Q * e] = W[q] * (j[0] * j[0] + j[1] * j[1]) / det;
    }
}

int main(int argc, char **argv) {
  const int Q = argc > 1 ? std::atoi(argv[1]) : 4, NE = 64, NQ = Q * Q;
  double *hW = (double *)malloc(sizeof(double) * NQ);
  double *hJ = (double *)malloc(sizeof(double) * 4 * NQ * NE);
  double *hD = (double *)malloc(sizeof(double) * NQ * NE);
  for (int q = 0; q < NQ; ++q)
    hW[q] = 0.1 + 0.01 * q;
  for (int i = 0; i < NQ * NE; ++i)
    for (int k = 0; k < 4; ++k)
      hJ[4 * i + k] = (k % 3 == 0 ? 1.0 : 0.0) + 0.001 * ((7 * i + k) % 13);
  double *W, *J, *D;
  cudaMalloc(&W, sizeof(double) * NQ);
  cudaMalloc(&J, sizeof(double) * 4 * NQ * NE);
  cudaMalloc(&D, sizeof(double) * NQ * NE);
  cudaMemcpy(W, hW, sizeof(double) * NQ, cudaMemcpyHostToDevice);
  cudaMemcpy(J, hJ, sizeof(double) * 4 * NQ * NE, cudaMemcpyHostToDevice);
  setup_kernel<<<NE, dim3(Q, Q)>>>(Q, W, J, D);
  if (cudaMemcpy(hD, D, sizeof(double) * NQ * NE, cudaMemcpyDeviceToHost) !=
      cudaSuccess)
    return 1;
  double s = 0;
  for (int i = 0; i < NQ * NE; ++i)
    s += hD[i];
  printf("checksum %.17g\n", s);
  return 0;
}
