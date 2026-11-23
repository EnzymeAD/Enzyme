// REQUIRES: enzyme, cuda-runtime
// RUN: rm -rf %t && mkdir %t
// RUN: %clang -x cuda --cuda-gpu-arch=%gpu_arch -fcuda-rdc -O2 -ffp-contract=on \
// RUN:     %loadClangPoseidonEnzyme -mllvm --poseidon-profile-generate -mllvm --poseidon-cache=%t/cache \
// RUN:     -I%poseidon_profile_inc %s -x none %poseidon_profile_cuda_lib %poseidon_profile_lib \
// RUN:     -L%cuda_lib_dir -lcudart -lstdc++ -lm -o %t/exe
// RUN: env POSEIDON_PROFILE_DIR=%t/profiles %t/exe | FileCheck --check-prefix=PRIMAL %s
// RUN: cat %t/profiles/*.fpprofile | FileCheck %s

// The kernel writes its output through a view made in the kernel that holds
// null when it is empty, as MFEM's DeviceTensor does. The output still takes
// the seed, so every profiled value has a nonzero sensitivity.

#include <cstdio>
#include <cuda_runtime.h>

#include "poseidon/poseidon.h"

struct View {
  double *data;
  __device__ View(double *p, int n) { data = n > 0 ? p : nullptr; }
};

POSEIDON_OPTIMIZE __global__ void square(double *y, const double *x, int n) {
  View Y(y, n);
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double a = x[i] * 1.5 + 0.25;
    Y.data[i] += a * a;
  }
}

int main() {
  const int n = 256;
  double *x, *y, h[n];
  cudaMalloc(&x, n * sizeof(double));
  cudaMalloc(&y, n * sizeof(double));
  for (int i = 0; i < n; ++i)
    h[i] = 0.5 + i * 0.01;
  cudaMemcpy(x, h, n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemset(y, 0, n * sizeof(double));
  square<<<(n + 63) / 64, 64>>>(y, x, n);
  cudaDeviceSynchronize();
  cudaMemcpy(h, y, n * sizeof(double), cudaMemcpyDeviceToHost);
  printf("y[0] = %.4f y[255] = %.4f\n", h[0], h[255]);
  return 0;
}

// PRIMAL: y[0] = 1.0000 y[255] = 23.2806

// CHECK: SiteId = 0
// CHECK: {{^0$}}
// CHECK: SumSens = {{[1-9]}}
// CHECK: Exec = 256
// CHECK: {{^1$}}
// CHECK: SumSens = {{[1-9]}}
// CHECK: Exec = 256
