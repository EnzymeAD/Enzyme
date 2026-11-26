// REQUIRES: enzyme, cuda-device
// RUN: rm -rf %t && mkdir %t
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O2 -poseidon-profile-generate -poseidon-cache=%t/c %s -o %t/gen
// RUN: env POSEIDON_KAPPA_LADDER=1e-6 POSEIDON_PROFILE_DIR=%t/prof %poseidon_kappa -- %t/gen %t/runs > /dev/null 2> %t/kappa.err
// RUN: FileCheck --check-prefix=FOUR %s < %t/runs
// RUN: FileCheck --check-prefix=MEASURE %s < %t/kappa.err
// RUN: cat %t/prof/*_Z5gaussPKdPdi*.fpprofile | FileCheck --check-prefix=MEASURED %s
// RUN: cat %t/prof/*_Z4wavePKdPdi*.fpprofile | FileCheck --check-prefix=MEASURED %s

// poseidon-kappa on two device sites: each run perturbs one kernel on the
// device, and each profile gets a condition number; 1 + 2 x 1 runs after the
// profiling run.

#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

#include "poseidon/poseidon.h"

POSEIDON_OPTIMIZE __global__ void gauss(const double *x, double *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = exp(-0.5 * x[i] * x[i]) * cos(4.0 * x[i]);
}

POSEIDON_OPTIMIZE __global__ void wave(const double *x, double *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = sin(x[i]) * x[i] + 1.0;
}

int main(int argc, char **argv) {
  if (FILE *f = fopen(argv[1], "a")) {
    fputs("run\n", f);
    fclose(f);
  }
  const int n = 256;
  double hx[n], hy[n], hz[n];
  for (int i = 0; i < n; ++i)
    hx[i] = -2.0 + 4.0 * i / (n - 1);
  double *x, *y, *z;
  cudaMalloc(&x, n * sizeof(double));
  cudaMalloc(&y, n * sizeof(double));
  cudaMalloc(&z, n * sizeof(double));
  cudaMemcpy(x, hx, n * sizeof(double), cudaMemcpyHostToDevice);
  gauss<<<(n + 63) / 64, 64>>>(x, y, n);
  wave<<<(n + 63) / 64, 64>>>(x, z, n);
  cudaMemcpy(hy, y, n * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(hz, z, n * sizeof(double), cudaMemcpyDeviceToHost);
  poseidon_metric("y and z", hy[n / 3] + hz[n / 3]);
  printf("%.17g\n", hy[n / 3] + hz[n / 3]);
  cudaFree(x);
  cudaFree(y);
  cudaFree(z);
}

// FOUR-COUNT-4: run
// FOUR-NOT: run

// MEASURE: [poseidon] measuring the condition numbers of metric "y and z" at 2 sites: 3 more runs of this program
// MEASURE: [poseidon] condition-number measurement: 3 runs,

// MEASURED: Kappa = {{[1-9]|0\.0*[1-9]}}
// MEASURED-NEXT: KappaMark = {{ok|nonlinear}}
