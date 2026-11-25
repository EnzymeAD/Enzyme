//===- inkernel_calibrate.cu - price the in-kernel raises -----------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// inkernel_calibrate: measure every in-kernel tensor-core raise class relative
// to the naive scalar-FP64 GEMM, for the cost model:
//     rel = (candidate wall clock) / (scalar-FP64 wall clock)
// on the same kernel at the same shape. poseidon-calibrate compiles this file
// once per candidate class through the plugin with -poseidon-apply-rewrites and
// times the result, so what is measured is the code a real solve emits. The
// GEMM is a naive dense one (A contiguous along K, one thread per output
// element, 16x16 blocks, `#pragma unroll 1` over the reduction), and without
// the plugin the annotation is inert, so the same kernel is the scalar-FP64
// denominator. The product is C = A * B, C of CAL_M x CAL_N, K = CAL_K
// (each 2048 by default); B is strided along K (row-major), or with
// -DCAL_B_COL_MAJOR=1 contiguous along K (column-major).
//
// Run:  ./bin [reps=5]        ("time_ms = <ms>" on stdout)
//       ./bin --profiling     (a launch of the site, for a profile run)
//
//===----------------------------------------------------------------------===//

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#include "poseidon/poseidon.h"

#ifndef CAL_M
#define CAL_M 2048
#endif
#ifndef CAL_N
#define CAL_N 2048
#endif
#ifndef CAL_K
#define CAL_K 2048
#endif
#ifndef CAL_B_COL_MAJOR
#define CAL_B_COL_MAJOR 0
#endif

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t e = (x);                                                       \
    if (e != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(e));                                          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

/// The GEMM site. Its profile is recorded at small extents, since
/// reverse-mode AD of a long reduction does not survive at the timed size, and
/// decides only which candidates exist, never what they measure.
POSEIDON_OPTIMIZE __global__ void matmul_opt(double *C, const double *A,
                                             const double *B) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= CAL_M || col >= CAL_N)
    return;
  const double *arow = A + (size_t)row * CAL_K;
  double val = 0.0;
#pragma unroll 1
  for (int k = 0; k < CAL_K; ++k)
#if CAL_B_COL_MAJOR
    val += arow[k] * B[(size_t)col * CAL_K + k];
#else
    val += arow[k] * B[(size_t)k * CAL_N + col];
#endif
  C[(size_t)row * CAL_N + col] = val;
}

static dim3 GG((CAL_M + 15) / 16, (CAL_N + 15) / 16), GB(16, 16);

int main(int argc, char **argv) {
  bool profiling = false;
  int reps = 5;
  for (int i = 1; i < argc; ++i) {
    char *end;
    long v = strtol(argv[i], &end, 10);
    if (!strcmp(argv[i], "--profiling")) {
      profiling = true;
    } else if (*argv[i] && !*end && v > 0 && v <= INT_MAX) {
      reps = (int)v;
    } else {
      fprintf(stderr, "usage: %s [reps] | --profiling\n", argv[0]);
      return 1;
    }
  }

  const size_t aElts = (size_t)CAL_M * CAL_K, bElts = (size_t)CAL_K * CAL_N,
               cElts = (size_t)CAL_M * CAL_N;
  double *A, *B, *C;
  CK(cudaMalloc(&A, aElts * 8));
  CK(cudaMalloc(&B, bElts * 8));
  CK(cudaMalloc(&C, cElts * 8));

  double *h = (double *)malloc((aElts > bElts ? aElts : bElts) * 8);
  srand(1);
  for (size_t i = 0; i < aElts; ++i)
    h[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
  CK(cudaMemcpy(A, h, aElts * 8, cudaMemcpyHostToDevice));
  for (size_t i = 0; i < bElts; ++i)
    h[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
  CK(cudaMemcpy(B, h, bElts * 8, cudaMemcpyHostToDevice));
  free(h);
  CK(cudaDeviceSynchronize());

  // A profiling run launches the site once; a timed run warms it up so.
  matmul_opt<<<GG, GB>>>(C, A, B);
  CK(cudaDeviceSynchronize());
  CK(cudaGetLastError());
  if (profiling) {
    printf("profiling run done (%dx%dx%d)\n", CAL_M, CAL_N, CAL_K);
    return 0;
  }

  cudaEvent_t t0, t1;
  CK(cudaEventCreate(&t0));
  CK(cudaEventCreate(&t1));
  CK(cudaEventRecord(t0));
  for (int r = 0; r < reps; ++r)
    matmul_opt<<<GG, GB>>>(C, A, B);
  CK(cudaEventRecord(t1));
  CK(cudaEventSynchronize(t1));
  float ms;
  CK(cudaEventElapsedTime(&ms, t0, t1));
  CK(cudaGetLastError());

  // Checksum so a silently-wrong raise is visible in the log; the rel is a
  // time, and accuracy is the accuracy model's job.
  double *hc = (double *)malloc(cElts * 8);
  CK(cudaMemcpy(hc, C, cElts * 8, cudaMemcpyDeviceToHost));
  double s = 0.0;
  for (size_t i = 0; i < cElts; i += 4099)
    s += hc[i];
  free(hc);

  printf("shape = %dx%dx%d\nreps = %d\nchecksum = %.10e\ntime_ms = %.6f\n",
         CAL_M, CAL_N, CAL_K, reps, s, ms / reps);
  return 0;
}
