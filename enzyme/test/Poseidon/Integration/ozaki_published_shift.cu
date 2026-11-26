// The Ozaki-II runtime's shift is GEMMul8 v3.1.1's fast mode: row a of A times
// 2^floor(log2P - log2||a||_2), truncated. With B = I, nm = 10:
//   row 0 = 1 + 2^-38 + 2^-40: C = 1 + 2^-38 (truncated, not rounded);
//   row 1 = 16 x (1 + 2^-36): C = 1 + 2^-36 (a max-based shift drops the bit);
//   row 2 = 0: C = 0.
// REQUIRES: cuda-device
// RUN: %clang -x cuda --cuda-gpu-arch=%gpu_arch -O2 -I%poseidon_runtime_inc %s -x none %poseidon_gemm_cuda_lib -L%cuda_lib_dir -lcublas -lcudart -lstdc++ -lm -o %t.exe
// RUN: %t.exe | FileCheck %s

#include "gemm/LibraryCalls.h"
#include <cmath>
#include <cstdio>

int main() {
  constexpr int N = 16;
  double A[N * N] = {}, B[N * N] = {}, C[N * N];
  A[0] = 1.0 + std::ldexp(1.0, -38) + std::ldexp(1.0, -40);
  for (int c = 0; c < N; ++c)
    A[N + c] = 1.0 + std::ldexp(1.0, -36);
  for (int i = 0; i < N; ++i)
    B[i * N + i] = 1.0;
  double *dA, *dB, *dC;
  gpuMalloc(&dA, sizeof A);
  gpuMalloc(&dB, sizeof B);
  gpuMalloc(&dC, sizeof C);
  gpuMemcpy(dA, A, sizeof A, gpuMemcpyHostToDevice);
  gpuMemcpy(dB, B, sizeof B, gpuMemcpyHostToDevice);
  __poseidon_ozakiii_dgemm(dC, dA, dB, N, N, N, N, N, N, 0, 0, 0, 1.0, 0.0, 0,
                         10);
  if (gpuMemcpy(C, dC, sizeof C, gpuMemcpyDeviceToHost) != gpuSuccess)
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
