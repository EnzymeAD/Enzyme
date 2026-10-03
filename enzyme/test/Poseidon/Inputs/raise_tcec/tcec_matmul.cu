#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

template <typename R, typename... T>
__device__ R __poseidon_fp_optimize(void *, T...);
__device__ int enzyme_const;
__device__ int enzyme_dup;

extern "C" __device__ __attribute__((noinline)) void
matmul_body(const double *A, const double *B, double *D) {
  unsigned tx = threadIdx.x;
  unsigned ty = threadIdx.y;
  double acc = 0.0;
#pragma unroll 1
  for (int k = 0; k < 16; ++k)
    acc += A[ty * 16 + k] * B[k * 16 + tx];
  D[ty * 16 + tx] = acc;
}

__global__ void matmul_kernel(const double *A, const double *B, double *D,
                              double *dD) {
  __poseidon_fp_optimize<void>((void *)matmul_body, enzyme_const, A,
                             enzyme_const, B, enzyme_dup, D, dD);
}

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t e = (x);                                                       \
    if (e != cudaSuccess) {                                                    \
      fprintf(stderr, "cuda: %s\n", cudaGetErrorString(e));                    \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

int main(void) {
  constexpr int M = 16, N = 16, K = 16;
  double hA[M * K], hB[K * N], hD[M * N], hone[M * N];
  for (int i = 0; i < M * K; ++i)
    hA[i] = 1.0 + 0.001 * i;
  for (int i = 0; i < K * N; ++i)
    hB[i] = 0.5 + 0.003 * (i % 97);
  for (int i = 0; i < M * N; ++i)
    hone[i] = 1.0;
  double *dA, *dB, *dD, *ddD;
  CK(cudaMalloc(&dA, sizeof(hA)));
  CK(cudaMalloc(&dB, sizeof(hB)));
  CK(cudaMalloc(&dD, sizeof(hD)));
  CK(cudaMalloc(&ddD, sizeof(hone)));
  CK(cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice));
  for (int rep = 0; rep < 2; ++rep) {
    CK(cudaMemcpy(ddD, hone, sizeof(hone), cudaMemcpyHostToDevice));
    matmul_kernel<<<1, dim3(N, M)>>>(dA, dB, dD, ddD);
    CK(cudaDeviceSynchronize());
  }
  CK(cudaMemcpy(hD, dD, sizeof(hD), cudaMemcpyDeviceToHost));
  int nbad = 0;
  for (int ty = 0; ty < M; ++ty)
    for (int tx = 0; tx < N; ++tx) {
      double e = 0.0;
      for (int k = 0; k < K; ++k)
        e += hA[ty * K + k] * hB[k * N + tx];
      if (std::fabs(hD[ty * N + tx] - e) > 1e-12)
        ++nbad;
    }
  printf(nbad ? "MATMUL-FAIL %d\n" : "MATMUL-PASS\n", nbad);
  return nbad != 0;
}
