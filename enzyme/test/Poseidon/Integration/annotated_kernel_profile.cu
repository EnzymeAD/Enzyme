// REQUIRES: enzyme, cuda-device
// RUN: rm -rf %t && mkdir %t
// RUN: %clang -x cuda --cuda-gpu-arch=%gpu_arch -fcuda-rdc -O2 -ffp-contract=on %loadClangPoseidonEnzyme -mllvm --poseidon-profile-generate -mllvm --poseidon-cache=%t/cache -I%poseidon_profile_inc %s -x none %poseidon_profile_cuda_lib %poseidon_profile_lib -L%cuda_lib_dir -lcudart -lstdc++ -lm -o %t/exe
// RUN: cat %t/cache/launches_*.json | FileCheck --check-prefix=DESC %s
// RUN: env POSEIDON_PROFILE_DIR=%t/profiles %t/exe | FileCheck --check-prefix=PRIMAL %s
// RUN: cat %t/profiles/*.fpprofile | FileCheck %s
// RUN: %clang -x cuda --cuda-gpu-arch=%gpu_arch -fcuda-rdc -O2 -ffp-contract=on %loadClangPoseidonEnzyme -mllvm --poseidon-profile-generate -mllvm --poseidon-cache=%t/cache2 -DNULL_SPACE -I%poseidon_profile_inc %s -x none %poseidon_profile_cuda_lib %poseidon_profile_lib -L%cuda_lib_dir -lcudart -lstdc++ -lm -o %t/null
// RUN: env POSEIDON_PROFILE_DIR=%t/null_profiles %t/null | FileCheck --check-prefix=NULL-PRIMAL %s
// RUN: cat %t/null_profiles/*.fpprofile | FileCheck --check-prefix=NULL %s
// RUN: cat %t/null_profiles/*.fpprofile | FileCheck --check-prefix=NULL-NONE %s

// A kernel with POSEIDON_OPTIMIZE alone: its launch stub sizes, allocates and
// seeds the shadows (outputSeed of each element's index); the profile has
// non-zero gradients, the result is right. NULL_SPACE: a 1D stiffness apply
// y = G^T W G x through shared memory, G the derivative of a quadratic
// Lagrange basis, whose outputs sum to zero for any x: every value before the
// test-side contraction still has a sensitivity of the outputs' order.

#include <cstdio>
#include <cuda_runtime.h>

#include "poseidon/poseidon.h"

#ifdef NULL_SPACE
constexpr int D = 3, Q = 4, NE = 16;

struct Basis {
  double G[Q][D], W[Q];
};

POSEIDON_OPTIMIZE __global__ void stiffness(double *y, const double *x, Basis b,
                                            int ne) {
  __shared__ double s[Q];
  for (int e = blockIdx.x; e < ne; e += gridDim.x) {
    for (int q = threadIdx.x; q < Q; q += blockDim.x) {
      double u = 0.0;
      for (int j = 0; j < D; ++j)
        u += b.G[q][j] * x[j + D * e];
      s[q] = b.W[q] * u;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
      double v = 0.0;
      for (int q = 0; q < Q; ++q)
        v += b.G[q][i] * s[q];
      y[i + D * e] += v;
    }
    __syncthreads();
  }
}

int main() {
  const double xq[Q] = {0.0694318442029737, 0.3300094782075719,
                        0.6699905217924281, 0.9305681557970263};
  const double wq[Q] = {0.1739274225687269, 0.3260725774312731,
                        0.3260725774312731, 0.1739274225687269};
  Basis b;
  for (int q = 0; q < Q; ++q) {
    b.G[q][0] = 4 * xq[q] - 3;
    b.G[q][1] = -8 * xq[q] + 4;
    b.G[q][2] = 4 * xq[q] - 1;
    b.W[q] = wq[q];
  }
  double h[D * NE], *x, *y;
  for (int i = 0; i < D * NE; ++i)
    h[i] = 0.1 + 0.8 * ((i * 37) % 101) / 101.0;
  cudaMalloc(&x, sizeof h);
  cudaMalloc(&y, sizeof h);
  cudaMemcpy(x, h, sizeof h, cudaMemcpyHostToDevice);
  cudaMemset(y, 0, sizeof h);
  stiffness<<<NE / 4, Q>>>(y, x, b, NE);
  cudaDeviceSynchronize();
  cudaMemcpy(h, y, sizeof h, cudaMemcpyDeviceToHost);
  double sum = 0.0;
  for (double v : h)
    sum += v;
  printf("|sum(y)| < 1e-12: %d, y[0] = %.4f\n", sum < 1e-12 && sum > -1e-12,
         h[0]);
  return 0;
}
#else
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
#endif

// The site, its four launch arguments, the three pointers among them, and the
// eight-byte seed the one output takes.
// DESC: _Z4axpbPdPKdS1_i 0 4 3 0 1 2 8 0 0

// PRIMAL: o[0] = 2.000 o[255] = 16.1525

// The output's 256 elements carry the profiler's seed (outputSeed), by index.
// CHECK: SiteId = 0
// CHECK: SumGrad = -4.8372591333{{[0-9]+}}e+00
// CHECK-NEXT: SumAbsGrad = 2.6151930816{{[0-9]+}}e+02
// CHECK-NEXT: Exec = 256

// NULL-PRIMAL: |sum(y)| < 1e-12: 1, y[0] = {{-?[0-9]+\.[0-9]+}}
// NULL: SiteId = 0
// NULL-COUNT-9: SumSens = {{[1-9]\.[0-9]+}}e+01
// NULL-NONE-NOT: SumSens = {{[0-9.]+}}e-
// NULL-NONE-NOT: SumSens = 0.0
