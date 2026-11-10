// The Ozaki-II runtime scales operands by the published fast mode
// (arXiv:2504.08009, constants and rounding of GEMMul8 v3.1.1): a row a of A
// is multiplied by 2^floor(log2P - log2||a||_2) and truncated toward zero.
// With B = I every column of B has norm 1, so C = trunc(2^sft a) / 2^sft
// shows each quantized row. nm = 10, log2P = 39.1204758:
//   row 0 = (1 + 2^-38 + 2^-40, 0, ...): shift 39, truncated, so C = 1 + 2^-38
//     (round to nearest would give 1 + 2^-38 + 2^-39);
//   row 1 = 16 x (1 + 2^-36): the 2-norm 4(1 + 2^-36) gives shift 37, so every
//     C entry is 1 + 2^-36 exactly (a max-based shift of 35 drops that bit);
//   row 2 = 0: C = 0.
//
// REQUIRES: cuda-runtime
// RUN: %clang -x cuda --cuda-gpu-arch=%gpu_arch -O2 %s -x none \
// RUN:     %poseidon_rt_lib -L%cuda_lib_dir -lcublas -lcudart \
// RUN:     -lstdc++ -lm -o %t.exe
// RUN: %t.exe | FileCheck %s

#include <cmath>
#include <cstdio>
#ifdef __HIP__
#include "hip_as_cuda.h"
#else
#include <cuda_runtime.h>
#endif

extern "C" void __poseidon_ozaki_dgemm(double *C, const double *A,
                                       const double *B, int N, int lda, int ldb,
                                       int ldc, int transA, int transB,
                                       double alpha, double beta,
                                       cudaStream_t stream, int num_moduli);

int main() {
  constexpr int N = 16;
  double A[N * N] = {}, B[N * N] = {}, C[N * N];
  A[0] = 1.0 + std::ldexp(1.0, -38) + std::ldexp(1.0, -40);
  for (int c = 0; c < N; ++c)
    A[N + c] = 1.0 + std::ldexp(1.0, -36);
  for (int i = 0; i < N; ++i)
    B[i * N + i] = 1.0;
  double *dA, *dB, *dC;
  cudaMalloc(&dA, sizeof A);
  cudaMalloc(&dB, sizeof B);
  cudaMalloc(&dC, sizeof C);
  cudaMemcpy(dA, A, sizeof A, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, sizeof B, cudaMemcpyHostToDevice);
  __poseidon_ozaki_dgemm(dC, dA, dB, N, N, N, N, 0, 0, 1.0, 0.0, 0, 10);
  if (cudaMemcpy(C, dC, sizeof C, cudaMemcpyDeviceToHost) != cudaSuccess)
    return 1;
  bool row1 = true, rest = true;
  for (int c = 0; c < N; ++c)
    row1 = row1 && C[N + c] == C[N];
  for (int i = 0; i < N * N; ++i)
    if (i != 0 && (i < N || i >= 2 * N))
      rest = rest && C[i] == 0.0;
  // CHECK: row0 0x1.0000000004p+0
  printf("row0 %a\n", C[0]);
  // CHECK: row1 0x1.000000001p+0 uniform 1
  printf("row1 %a uniform %d\n", C[N], row1);
  // CHECK: others zero 1
  printf("others zero %d\n", rest);
  return 0;
}
