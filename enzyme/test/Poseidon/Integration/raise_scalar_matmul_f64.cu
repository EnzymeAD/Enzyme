// RUN: %clang -x cuda --cuda-gpu-arch=%gpu_arch -fcuda-rdc -O2 \
// RUN:     %clangLoadPoseidonEnzyme -mllvm --poseidon-profile-generate \
// RUN:     -I%FPProfileInc %s %FPProfileCUDASrc -x none %FPProfileLib \
// RUN:     -L/usr/local/cuda/lib64 -lcudart -lstdc++ -lm -o %t.exe
// RUN: rm -rf %t.profiles && POSEIDON_PROFILE_DIR=%t.profiles %t.exe
//
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -O2 \
// RUN:     -S -emit-llvm %s -o %t.dev.ll
// RUN: %opt %t.dev.ll %loadPoseidon \
// RUN:     -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" \
// RUN:     -poseidon-profile-use=%t.profiles \
// RUN:     -poseidon-enable-herbie=false -poseidon-enable-pt=false \
// RUN:     -poseidon-cache= -poseidon-raise-wmma -poseidon-print \
// RUN:     -poseidon-cost-model=%gpu_cost_model \
// RUN:     -S 2>&1 | FileCheck %s
//
// REQUIRES: poseidon, enzyme, cuda-runtime

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

template <typename R, typename... T>
__device__ R __poseidon_fp_optimize(void *, T...);
__device__ int enzyme_const;

extern "C" __device__ __attribute__((noinline)) void
scalar_matmul_body_f64(const double *A, const double *B, double *D) {
  unsigned tx = threadIdx.x;
  unsigned ty = threadIdx.y;
  double acc = 0.0;
#pragma unroll 1
  for (int k = 0; k < 8; ++k)
    acc += A[ty * 8 + k] * B[k * 8 + tx];
  D[ty * 8 + tx] = acc;
}

__global__ void scalar_matmul_kernel(const double *A, const double *B,
                                     double *D) {
  __poseidon_fp_optimize<void>((void *)scalar_matmul_body_f64, enzyme_const, A,
                             enzyme_const, B, enzyme_const, D);
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
  constexpr int M = 8, N = 8, K = 8;
  double hA[M * K], hB[K * N], hD[M * N];
  for (int i = 0; i < M * K; ++i)
    hA[i] = 1.0 + 0.25 * (i % 5);
  for (int i = 0; i < K * N; ++i)
    hB[i] = 0.5 + 0.125 * (i % 7);

  double *dA = nullptr, *dB = nullptr, *dD = nullptr;
  CK(cudaMalloc(&dA, sizeof(hA)));
  CK(cudaMalloc(&dB, sizeof(hB)));
  CK(cudaMalloc(&dD, sizeof(hD)));
  CK(cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice));
  scalar_matmul_kernel<<<1, dim3(N, M)>>>(dA, dB, dD);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hD, dD, sizeof(hD), cudaMemcpyDeviceToHost));

  int nbad = 0;
  for (int ty = 0; ty < M; ++ty) {
    for (int tx = 0; tx < N; ++tx) {
      double expected = 0.0;
      for (int k = 0; k < K; ++k)
        expected += hA[ty * K + k] * hB[k * N + tx];
      if (std::fabs(hD[ty * N + tx] - expected) > 1e-12) {
        if (nbad < 4)
          fprintf(stderr, "D[%d,%d] = %.17g (expected %.17g)\n", ty, tx,
                  hD[ty * N + tx], expected);
        ++nbad;
      }
    }
  }
  CK(cudaFree(dA));
  CK(cudaFree(dB));
  CK(cudaFree(dD));
  if (nbad) {
    fprintf(stderr, "MATMUL-FAIL: %d mismatches\n", nbad);
    return 1;
  }
  printf("MATMUL-PASS\n");
  return 0;
}

// CHECK: [poseidon] Found 1 AbstractMatmul(s) for preprocess_scalar_matmul_body_f64
// CHECK-NEXT: Matmul[0]: 8x8x8 a=f64 b=f64 acc=f64 d=f64 origin=ScalarLoopReduction
// CHECK-NOT: Matmul[1]:
