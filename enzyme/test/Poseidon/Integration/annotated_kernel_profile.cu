// RUN: rm -rf %t.cache && %clang -x cuda --cuda-gpu-arch=%gpu_arch -fcuda-rdc -O2 -ffp-contract=on \
// RUN:     %clangLoadPoseidonEnzyme -mllvm --poseidon-profile-generate -mllvm --poseidon-cache=%t.cache \
// RUN:     -I%FPProfileInc %s %FPProfileCUDASrc -x none %FPProfileLib \
// RUN:     -L/usr/local/cuda/lib64 -lcudart -lstdc++ -lm -o %t.exe
// RUN: cat %t.cache/_Z4axpbPdPKdS1_i.profgen | FileCheck --check-prefix=DESC %s
// RUN: rm -rf %t.profiles && POSEIDON_PROFILE_DIR=%t.profiles %t.exe | FileCheck --check-prefix=PRIMAL %s
// RUN: cat %t.profiles/*.fpprofile | FileCheck %s
//
// REQUIRES: poseidon, enzyme, cuda-runtime

// A kernel annotated with POSEIDON_OPTIMIZE and nothing else: no marker call,
// no shadow buffers, no profiling mode. The compiler makes the kernel its own
// reverse pass and rewrites its launch stub so the runtime sizes, allocates and
// seeds the shadows; the profile must come back with non-zero gradients and the
// kernel must still compute its primal result.

#include <cstdio>
#include <cuda_runtime.h>

#include "poseidon/poseidon.h"

POSEIDON_OPTIMIZE __global__ void axpb(double *o, const double *a,
                                       const double *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double x = a[i], y = b[i];
    o[i] = x * y + x;
  }
}

int main() {
  const int n = 256;
  double *o, *a, *b;
  cudaMalloc(&o, n * sizeof(double));
  cudaMalloc(&a, n * sizeof(double));
  cudaMalloc(&b, n * sizeof(double));
  double h[n];
  for (int i = 0; i < n; ++i)
    h[i] = 1.0 + i * 0.01;
  cudaMemcpy(a, h, n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(b, h, n * sizeof(double), cudaMemcpyHostToDevice);
  axpb<<<(n + 63) / 64, 64>>>(o, a, b, n);
  cudaDeviceSynchronize();
  cudaMemcpy(h, o, n * sizeof(double), cudaMemcpyDeviceToHost);
  printf("o[0] = %.3f o[255] = %.4f\n", h[0], h[255]);
  return 0;
}

// The site, its four launch arguments, the three pointers among them, and the
// eight-byte seed the one output takes.
// DESC: _Z4axpbPdPKdS1_i 0 4 3 0 1 2 8 0 0

// PRIMAL: o[0] = 2.000 o[255] = 16.1525

// CHECK: SiteId = 0
// CHECK: SumGrad = 2.56000000000000000e+02
// CHECK: Exec = 256
