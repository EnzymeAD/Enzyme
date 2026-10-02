// The confidence level of a site's accuracy target, on one real product site.
// The 16x16x16 scalar reduction below is profiled once on the GPU and then
// solved three times against ONE accuracy cache, at 0.95, at 0.5 and at 0.95
// again.
//
// Two properties are checked on the modelled domain error the pass reports per
// candidate. It is monotone non-decreasing in the confidence, because the level
// is the percentile of the sampled relative errors the number is read off. And
// the third run reproduces the first exactly although the second wrote the same
// candidates into the same cache, because the cache key carries the level.
//
// RUN: rm -rf %t.profiles %t.cache
// RUN: %clang -x cuda --cuda-gpu-arch=%gpu_arch -fcuda-rdc -O2 -ffp-contract=on \
// RUN:     %clangLoadPoseidonEnzyme -mllvm --poseidon-profile-generate \
// RUN:     -I%FPProfileInc %s %FPProfileCUDASrc -x none %FPProfileLib \
// RUN:     -L/usr/local/cuda/lib64 -lcudart -lstdc++ -lm -o %t.exe
// RUN: POSEIDON_PROFILE_DIR=%t.profiles %t.exe
//
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -O2 \
// RUN:     -ffp-contract=on -I%FPProfileInc -S -emit-llvm %s -o %t.dev.ll
//
// RUN: %opt %t.dev.ll %loadPoseidon -passes="poseidon,poseidon-finalize" \
// RUN:     -poseidon-profile-use=%t.profiles -poseidon-enable-herbie=false \
// RUN:     -poseidon-enable-pt=false -poseidon-cache=%t.cache \
// RUN:     -poseidon-raise-wmma -poseidon-print -poseidon-confidence=0.95 \
// RUN:     -poseidon-cost-model=%gpu_cost_model -S -o /dev/null 2>%t.p95.log
// RUN: %opt %t.dev.ll %loadPoseidon -passes="poseidon,poseidon-finalize" \
// RUN:     -poseidon-profile-use=%t.profiles -poseidon-enable-herbie=false \
// RUN:     -poseidon-enable-pt=false -poseidon-cache=%t.cache \
// RUN:     -poseidon-raise-wmma -poseidon-print -poseidon-confidence=0.5 \
// RUN:     -poseidon-cost-model=%gpu_cost_model -S -o /dev/null 2>%t.p50.log
// RUN: %opt %t.dev.ll %loadPoseidon -passes="poseidon,poseidon-finalize" \
// RUN:     -poseidon-profile-use=%t.profiles -poseidon-enable-herbie=false \
// RUN:     -poseidon-enable-pt=false -poseidon-cache=%t.cache \
// RUN:     -poseidon-raise-wmma -poseidon-print -poseidon-confidence=0.95 \
// RUN:     -poseidon-cost-model=%gpu_cost_model -S -o /dev/null 2>%t.p95b.log
//
// Both levels are reported, each on its own percentile label.
// RUN: FileCheck --check-prefix=LEVEL95 %s < %t.p95.log
// RUN: FileCheck --check-prefix=LEVEL50 %s < %t.p50.log
//
// RUN: grep -o 'domainError=[0-9.e+-]*' %t.p95.log > %t.p95.err
// RUN: grep -o 'domainError=[0-9.e+-]*' %t.p50.log > %t.p50.err
// RUN: grep -o 'domainError=[0-9.e+-]*' %t.p95b.log > %t.p95b.err
// RUN: paste %t.p50.err %t.p95.err | awk -F'[=\t]' '{ n++; if ($4+0 < $2+0) v++; if ($4+0 > $2+0) s++ } END { printf "CONFIDENCE-MONOTONE candidates=%d violations=%d strict=%d\n", n, v+0, s+0 }' | tee %t.mono.txt | FileCheck --check-prefix=MONO %s
//
// The cache key carries the level: the third run reads back what the first
// wrote, not what the second wrote over the same candidates.
// RUN: diff -u %t.p95.err %t.p95b.err
//
// REQUIRES: poseidon, enzyme, cuda-runtime

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "poseidon/poseidon.h"

#define TILE 16

POSEIDON_OPTIMIZE __global__ void conf_matmul(const double *A, const double *B,
                                              double *D) {
  unsigned tx = threadIdx.x;
  unsigned ty = threadIdx.y;
  double acc = 0.0;
#pragma unroll 1
  for (int k = 0; k < TILE; ++k)
    acc += A[ty * TILE + k] * B[k * TILE + tx];
  D[ty * TILE + tx] = acc;
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
  constexpr int M = TILE, N = TILE, K = TILE;
  constexpr int ROUNDS = 8;
  double hA[M * K], hB[K * N], hD[M * N];

  double *dA = nullptr, *dB = nullptr, *dD = nullptr;
  CK(cudaMalloc(&dA, sizeof(hA)));
  CK(cudaMalloc(&dB, sizeof(hB)));
  CK(cudaMalloc(&dD, sizeof(hD)));

  int nbad = 0;
  // Several launches with different operands, so every profiled cell carries a
  // RANGE. With one launch min == max in every cell, the accuracy model draws
  // the same point every time, and a p50 equal to a p95 would make the
  // monotonicity check vacuous.
  for (int r = 0; r < ROUNDS; ++r) {
    for (int i = 0; i < M * K; ++i)
      hA[i] = (1.0 + 0.25 * ((i + r) % 5)) *
              std::ldexp(1.0, -((i + 3 * r) % 11));
    for (int i = 0; i < K * N; ++i)
      hB[i] = (0.5 + 0.125 * ((i + 2 * r) % 7)) *
              std::ldexp(1.0, -((i + 5 * r) % 13));
    CK(cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice));
    conf_matmul<<<1, dim3(N, M)>>>(dA, dB, dD);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hD, dD, sizeof(hD), cudaMemcpyDeviceToHost));

    for (int ty = 0; ty < M; ++ty)
      for (int tx = 0; tx < N; ++tx) {
        double expected = 0.0;
        for (int k = 0; k < K; ++k)
          expected += hA[ty * K + k] * hB[k * N + tx];
        if (std::fabs(hD[ty * N + tx] - expected) > 1e-12 * std::fabs(expected))
          ++nbad;
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

// LEVEL95: accuracy target confidence 9.500000e-01 (-poseidon-confidence)
// LEVEL95: p95_relErr=
// LEVEL50: accuracy target confidence 5.000000e-01 (-poseidon-confidence)
// LEVEL50: p50_relErr=

// MONO: CONFIDENCE-MONOTONE candidates={{[1-9][0-9]*}} violations=0 strict={{[1-9][0-9]*}}
