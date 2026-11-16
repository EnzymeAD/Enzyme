// Two contractions staged through one __shared__ buffer that reaches the body
// as a pointer parameter, then a once-per-thread epilogue: the shape of MFEM's
// partial-assembly diffusion apply.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

template <typename R, typename... T>
__device__ R __poseidon_fp_optimize(void *, T...);
__device__ int enzyme_const;
__device__ int enzyme_dup;

extern "C" __device__ __attribute__((noinline)) void
staged_body(double *__restrict__ S, const double *__restrict__ A,
            const double *__restrict__ B, double *__restrict__ D) {
  unsigned tx = threadIdx.x;
  unsigned ty = threadIdx.y;
  double acc = 0.0;
#pragma unroll 1
  for (int k = 0; k < 16; ++k)
    acc += A[ty * 16 + k] * B[k * 16 + tx];
  S[ty * 16 + tx] = acc;
  __syncthreads();
  double acc2 = 0.0;
#pragma unroll 1
  for (int k = 0; k < 16; ++k)
    acc2 += A[ty * 16 + k] * S[k * 16 + tx];
  D[ty * 16 + tx] += acc2;
}

__global__ void staged_kernel(const double *A, const double *B, double *D,
                              double *dD) {
  __shared__ double s[256];
  __shared__ double ds[256];
  ds[threadIdx.y * 16 + threadIdx.x] = 0.0;
  __syncthreads();
  __poseidon_fp_optimize<void>((void *)staged_body, enzyme_dup, s, ds,
                               enzyme_const, A, enzyme_const, B, enzyme_dup, D,
                               dD);
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
  constexpr int N = 16;
  double hA[N * N], hB[N * N], hD[N * N], hone[N * N];
  for (int i = 0; i < N * N; ++i) {
    hA[i] = 1.0 + 0.001 * i;
    hB[i] = 0.5 + 0.003 * (i % 97);
    hone[i] = 1.0 + 0.01 * (i % 7);
  }
  double *dA, *dB, *dD, *ddD;
  CK(cudaMalloc(&dA, sizeof(hA)));
  CK(cudaMalloc(&dB, sizeof(hB)));
  CK(cudaMalloc(&dD, sizeof(hD)));
  CK(cudaMalloc(&ddD, sizeof(hone)));
  CK(cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice));
  CK(cudaMemset(dD, 0, sizeof(hD)));
  for (int rep = 0; rep < 2; ++rep) {
    CK(cudaMemcpy(ddD, hone, sizeof(hone), cudaMemcpyHostToDevice));
    staged_kernel<<<1, dim3(N, N)>>>(dA, dB, dD, ddD);
    CK(cudaDeviceSynchronize());
  }
  CK(cudaMemcpy(hD, dD, sizeof(hD), cudaMemcpyDeviceToHost));
  int nbad = 0;
  for (int ty = 0; ty < N; ++ty)
    for (int tx = 0; tx < N; ++tx) {
      double e = 0.0;
      for (int k = 0; k < N; ++k) {
        double s = 0.0;
        for (int j = 0; j < N; ++j)
          s += hA[k * N + j] * hB[j * N + tx];
        e += hA[ty * N + k] * s;
      }
      if (std::fabs(hD[ty * N + tx] - 2.0 * e) > 1e-9 * std::fabs(e))
        ++nbad;
    }
  printf(nbad ? "STAGED-FAIL %d\n" : "STAGED-PASS\n", nbad);
  return nbad != 0;
}
