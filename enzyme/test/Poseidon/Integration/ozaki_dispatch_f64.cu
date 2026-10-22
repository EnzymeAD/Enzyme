// Ozaki-II host dispatch end to end through poseidon-clang++: a scalar FP64
// GEMM kernel is profiled, solved under -poseidon-tau=1e-12 against the
// device cost model, the error-budget selector picks an Ozaki-II moduli count,
// the launch stub becomes a __poseidon_ozaki_dgemm call, and the result is
// compared elementwise with cublasDgemm on the same operands. The bound the
// optimized binary is held to is tau itself.
//
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O2 \
// RUN:     -poseidon-profile-generate %s -o %t.exe -lcublas
// RUN: rm -rf %t.profile && POSEIDON_PROFILE_DIR=%t.profile %t.exe 1e-13 \
// RUN:   | FileCheck --check-prefix=PRIMAL %s
//
// RUN: rm -rf %t.cache && %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O2 \
// RUN:     -poseidon-profile-use=%t.profile -poseidon-cache=%t.cache \
// RUN:     -poseidon-cost-model=%gpu_cost_model -poseidon-print \
// RUN:     -poseidon-enable-herbie=0 -poseidon-enable-pt=0 -poseidon-tau=1e-12 \
// RUN:     %s -o %t.opt.exe -lcublas 2>&1 | FileCheck %s
// RUN: %t.opt.exe 1e-12 | FileCheck --check-prefix=OPT %s
//
// REQUIRES: poseidon, enzyme, cuda-runtime

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <poseidon/poseidon.h>

constexpr int N = 256;

POSEIDON_OPTIMIZE __global__ void gemm_kernel(double *__restrict__ C,
                                              const double *__restrict__ A,
                                              const double *__restrict__ B) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= N || col >= N)
    return;
  double acc = 0.0;
#pragma unroll 1
  for (int k = 0; k < N; ++k)
    acc += A[(size_t)row * N + k] * B[(size_t)k * N + col];
  C[(size_t)row * N + col] = acc;
}

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t e = (x);                                                       \
    if (e != cudaSuccess) {                                                    \
      fprintf(stderr, "cuda: %s\n", cudaGetErrorString(e));                    \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <max relative error>\n", argv[0]);
    return 2;
  }
  const double tol = std::atof(argv[1]);
  CK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1ULL << 30));
  const size_t n2 = (size_t)N * N;
  double *hA = (double *)std::malloc(n2 * sizeof(double));
  double *hB = (double *)std::malloc(n2 * sizeof(double));
  double *hC = (double *)std::malloc(n2 * sizeof(double));
  double *hR = (double *)std::malloc(n2 * sizeof(double));
  unsigned s = 12345u;
  auto next = [&s]() {
    s = s * 1664525u + 1013904223u;
    return (double)(s >> 8) / (double)(1u << 24);
  };
  for (size_t i = 0; i < n2; ++i)
    hA[i] = 0.5 + next();
  for (size_t i = 0; i < n2; ++i)
    hB[i] = 0.5 + next();

  double *dA, *dB, *dC, *dR;
  CK(cudaMalloc(&dA, n2 * sizeof(double)));
  CK(cudaMalloc(&dB, n2 * sizeof(double)));
  CK(cudaMalloc(&dC, n2 * sizeof(double)));
  CK(cudaMalloc(&dR, n2 * sizeof(double)));
  CK(cudaMemcpy(dA, hA, n2 * sizeof(double), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, hB, n2 * sizeof(double), cudaMemcpyHostToDevice));

  gemm_kernel<<<dim3(N / 16, N / 16), dim3(16, 16)>>>(dC, dA, dB);
  CK(cudaDeviceSynchronize());

  cublasHandle_t h;
  if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS)
    return 1;
  const double one = 1.0, zero = 0.0;
  if (cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one, dB, N, dA, N,
                  &zero, dR, N) != CUBLAS_STATUS_SUCCESS)
    return 1;
  CK(cudaDeviceSynchronize());
  cublasDestroy(h);

  CK(cudaMemcpy(hC, dC, n2 * sizeof(double), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hR, dR, n2 * sizeof(double), cudaMemcpyDeviceToHost));
  double maxRel = 0.0;
  for (size_t i = 0; i < n2; ++i)
    maxRel = std::fmax(maxRel, std::fabs(hC[i] - hR[i]) / std::fabs(hR[i]));
  printf("max relative error vs cublasDgemm %.3e\n", maxRel);
  printf(maxRel <= tol ? "GEMM-PASS\n" : "GEMM-FAIL\n");
  return maxRel <= tol ? 0 : 1;
}

// PRIMAL: GEMM-PASS

// CHECK: Matmul[0]: 16x16x256 a=f64 b=f64 acc=f64 d=f64 origin=ScalarLoopReduction
// CHECK: ozaki-ii nm=12 wmma m16n16k16 s8/s32  compCost/MAC={{.*}}  rel=4.443700e-02  domainError=
// CHECK: Poseidon error-budget: matmul -> ozaki-ii nm=12 wmma m16n16k16 s8/s32 (domain err {{.*}} <= 1.000000e-12 at confidence 9.500000e-01)
// CHECK: Applying solution for matmul[0] -> ozaki-ii nm=12 wmma m16n16k16 s8/s32
// CHECK: [ozaki-host-dispatch] wrote descriptor for _Z11gemm_kernelPdPKdS1_: C=arg0 A=arg1 B=arg2 256x256x256 nm=12 (standalone)
// CHECK: [ozaki-host-dispatch] replaced stub body _Z26__device_stub__gemm_kernelPdPKdS1_ -> __poseidon_ozaki_dgemm (C=arg0 A=arg1 B=arg2 256x256x256)

// OPT: max relative error vs cublasDgemm
// OPT-NEXT: GEMM-PASS
