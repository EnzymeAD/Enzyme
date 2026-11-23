// REQUIRES: enzyme, cuda-runtime
// RUN: rm -rf %t && mkdir %t
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O3 -ffp-contract=on -poseidon-profile-generate -poseidon-cache=%t/cg %s -o %t/gen
// RUN: env POSEIDON_PROFILE_DIR=%t/prof %t/gen | FileCheck --check-prefix=FP64 %s
// RUN: FileCheck --check-prefix=PROF %s < %t/prof/preprocess__Z13forall_kernelIZ5applyiPKdPdEUliE_EviT__poseidon_body.fpprofile
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O3 -ffp-contract=on -poseidon-profile-use=%t/prof -poseidon-tau=1e-6 -poseidon-cost-model=%gpu_cost_model -poseidon-enable-herbie=0 -poseidon-print -poseidon-cache=%t/c %s -o %t/use 2> %t/use.err
// RUN: FileCheck --check-prefix=USE %s < %t/use.err
// RUN: %t/use | FileCheck --check-prefix=OUT %s

// The annotated lambda only calls `element`, which the inliner leaves a call
// (noinline here; in an application, an element body too large to inline).
// Its arithmetic is the site's: it is profiled, and the solve rewrites it.

#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

#include "poseidon/poseidon.h"

template <typename BODY> __global__ void forall_kernel(int n, BODY body) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    body(i);
}

__device__ __noinline__ void element(int i, const double *x, double *y) {
  double v = x[i];
  y[i] = (v * v + 3.0) * v + 1.0 / (v + 2.0);
}

void apply(int n, const double *x, double *y) {
  forall_kernel<<<(n + 127) / 128, 128>>>(
      n, [=] __device__(int i) POSEIDON_OPTIMIZE { element(i, x, y); });
}

int main() {
  const int n = 1024;
  double *x, *y, h[n];
  cudaMalloc(&x, n * sizeof(double));
  cudaMalloc(&y, n * sizeof(double));
  for (int i = 0; i < n; ++i)
    h[i] = 0.5 + i / double(n);
  cudaMemcpy(x, h, n * sizeof(double), cudaMemcpyHostToDevice);
  apply(n, x, y);
  cudaMemcpy(h, y, n * sizeof(double), cudaMemcpyDeviceToHost);
  double ref = 0, err = 0;
  for (int i = 0; i < n; ++i) {
    double v = 0.5 + i / double(n);
    double r = (v * v + 3.0) * v + 1.0 / (v + 2.0);
    ref += r * r;
    err += (h[i] - r) * (h[i] - r);
  }
  double rel = std::sqrt(err / ref);
  printf("%s\n", rel == 0 ? "exact" : rel <= 1e-6 ? "within 1e-6" : "too far");
  return 0;
}

// FP64: exact

// Every launch of the kernel ran the four operations of `element`.
// PROF: Exec = 1024
// PROF: Exec = 1024
// PROF: Exec = 1024
// PROF: Exec = 1024

// USE: [poseidon] pin: _Z13forall_kernelIZ5applyiPKdPdEUliE_EviT_:PT0=
// OUT: within 1e-6
