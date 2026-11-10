// RUN: rm -rf %t && mkdir -p %t
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch --cuda-device-only -O2 -S %s -o %t/plain.ptx
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O2 -poseidon-profile-generate -poseidon-cache=%t/cg %s -o %t/gen
// RUN: env POSEIDON_PROFILE_DIR=%t/prof %t/gen | FileCheck --check-prefix=OUT %s
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch --cuda-device-only -O2 -S -poseidon-profile-use=%t/prof -poseidon-cost-model=%gpu_cost_model -poseidon-enable-herbie=0 -poseidon-cache=%t/c -poseidon-print %s -o %t/use.ptx 2> %t/use.err
// RUN: awk '/^.visible .entry _Z(5other|10unprofiled)/{p=1} p{print} /^}/{p=0}' %t/plain.ptx > %t/plain.fns
// RUN: awk '/^.visible .entry _Z(5other|10unprofiled)/{p=1} p{print} /^}/{p=0}' %t/use.ptx > %t/use.fns
// RUN: diff %t/plain.fns %t/use.fns
// RUN: FileCheck --check-prefix=FNS %s < %t/use.fns
// RUN: FileCheck --check-prefix=SOLVE %s < %t/use.err
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O2 -poseidon-profile-use=%t/prof -poseidon-cost-model=%gpu_cost_model -poseidon-enable-herbie=0 -poseidon-cache=%t/c %s -o %t/use
// RUN: %t/use | FileCheck --check-prefix=OUT %s
// REQUIRES: poseidon, enzyme, cuda-runtime

// In a module where a site is rewritten, every function no rewrite reached is
// the one as written: a kernel that is no site, and a site the profiling run
// never reached, inline libdevice's square root as the plain build does,
// although the solve kept it out of line in the whole module.

#include "poseidon/poseidon.h"
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

POSEIDON_OPTIMIZE_TAU(1e-6)
__global__ void site(const double *x, double *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double a = x[i];
    y[i] = (3.0 * a + 0.5) * a + 1.0;
  }
}

__global__ void other(const double *x, double *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = sqrt(x[i]) + y[i];
}

POSEIDON_OPTIMIZE __global__ void unprofiled(const double *x, double *y,
                                             int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = sqrt(x[i]) * y[i];
}

int main(int argc, char **) {
  const int n = 1024;
  double *x, *y;
  cudaMallocManaged(&x, n * sizeof(double));
  cudaMallocManaged(&y, n * sizeof(double));
  for (int i = 0; i < n; ++i)
    x[i] = 1.0 + i / (double)n;
  site<<<n / 128, 128>>>(x, y, n);
  other<<<n / 128, 128>>>(x, y, n);
  if (argc > 1)
    unprofiled<<<n / 128, 128>>>(x, y, n);
  cudaDeviceSynchronize();
  double s = 0.0;
  for (int i = 0; i < n; ++i)
    s += y[i];
  std::printf("sum %.6f\n", s);
  cudaFree(x);
  cudaFree(y);
  return 0;
}

// OUT: sum 10203.2{{[0-9]+}}

// SOLVE: [poseidon] {{.*}}: 1 function(s) a rewrite reached; every other function as written

// FNS: .entry _Z5otherPKdPdi
// FNS-NOT: call
// FNS: sqrt.rn.f64
// FNS: .entry _Z10unprofiledPKdPdi
// FNS-NOT: call
// FNS: sqrt.rn.f64
