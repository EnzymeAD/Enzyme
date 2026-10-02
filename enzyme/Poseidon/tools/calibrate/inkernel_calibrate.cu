// inkernel_calibrate: measure every in-kernel tensor-core raise class relative
// to the naive scalar-FP64 GEMM, for the cost model:
//     rel = (candidate wall clock) / (scalar-FP64 wall clock)
// on the same kernel at the same shape. poseidon-calibrate compiles this file
// once per candidate class through the plugin with -poseidon-apply-rewrites and
// times the result, so what is measured is the code a real solve emits. The
// GEMM is byte for byte the shape the dense-GEMM benchmark uses (row-major A,
// K-strided B, one thread per output element, 16x16 blocks, `#pragma unroll 1`
// over the reduction), and without the plugin the annotation is inert, so the
// same kernel is the scalar-FP64 denominator.
//
// Run:  ./bin [reps=5]        ("time_ms = <ms>" on stdout)
//       ./bin --profiling     (a launch of the site, for a profile run)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#include "poseidon/poseidon.h"

#ifndef CAL_N
#define CAL_N 2048
#endif
#define N CAL_N
#define NN ((size_t)N * N)

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t e = (x);                                                       \
    if (e != cudaSuccess) {                                                    \
      printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,                     \
             cudaGetErrorString(e));                                           \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// The name and signature are the dense-GEMM benchmark's annotated kernel, so
// its surrogate profile is the profile this harness enumerates candidates
// from: reverse-mode AD of a K=N reduction does not survive on this
// translation unit, and the profile only decides which candidates exist, never
// what they measure.
POSEIDON_OPTIMIZE __global__ void matmul_opt(double *C, const double *A,
                                             const double *B) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= N || col >= N)
    return;
  const double *arow = A + (size_t)row * N;
  double val = 0.0;
#pragma unroll 1
  for (int k = 0; k < N; ++k)
    val += arow[k] * B[(size_t)k * N + col];
  C[(size_t)row * N + col] = val;
}

static dim3 GG((N + 15) / 16, (N + 15) / 16), GB(16, 16);

int main(int argc, char **argv) {
  bool profiling = false;
  int reps = 5;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--profiling"))
      profiling = true;
    else
      reps = atoi(argv[i]);
  }
  if (reps < 1)
    reps = 1;

  double *A, *B, *C;
  CK(cudaMalloc(&A, NN * 8));
  CK(cudaMalloc(&B, NN * 8));
  CK(cudaMalloc(&C, NN * 8));

  // Operands in [-1,1), the same range the dense-GEMM benchmark profiles over.
  double *h = (double *)malloc(NN * 8);
  srand(1);
  for (size_t i = 0; i < NN; ++i)
    h[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
  CK(cudaMemcpy(A, h, NN * 8, cudaMemcpyHostToDevice));
  for (size_t i = 0; i < NN; ++i)
    h[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
  CK(cudaMemcpy(B, h, NN * 8, cudaMemcpyHostToDevice));
  free(h);
  CK(cudaDeviceSynchronize());

  if (profiling) {
    matmul_opt<<<GG, GB>>>(C, A, B);
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());
    printf("profiling run done (N=%d)\n", N);
    return 0;
  }

  matmul_opt<<<GG, GB>>>(C, A, B);
  CK(cudaDeviceSynchronize());
  CK(cudaGetLastError());

  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  cudaEventRecord(t0);
  for (int r = 0; r < reps; ++r)
    matmul_opt<<<GG, GB>>>(C, A, B);
  cudaEventRecord(t1);
  CK(cudaEventSynchronize(t1));
  float ms;
  cudaEventElapsedTime(&ms, t0, t1);
  CK(cudaGetLastError());

  // Checksum so a silently-wrong raise is visible in the log; the rel is a
  // time, and accuracy is the accuracy model's job.
  double *hc = (double *)malloc(NN * 8);
  CK(cudaMemcpy(hc, C, NN * 8, cudaMemcpyDeviceToHost));
  double s = 0.0;
  for (size_t i = 0; i < NN; i += 4099)
    s += hc[i];
  free(hc);

  printf("N = %d\nreps = %d\nchecksum = %.10e\ntime_ms = %.6f\n", N, reps, s,
         ms / reps);
  return 0;
}
