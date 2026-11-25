// REQUIRES: enzyme, cuda-runtime
// RUN: rm -rf %t && mkdir %t
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch --cuda-device-only -O2 -S %s -o %t/plain.ptx
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O2 -poseidon-profile-generate -poseidon-cache=%t/cg %s -o %t/gen
// RUN: env POSEIDON_PROFILE_DIR=%t/prof %t/gen | FileCheck --check-prefix=OUT %s
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch --cuda-device-only -O2 -S -poseidon-profile-use=%t/prof -poseidon-cost-model=%gpu_cost_model -poseidon-enable-herbie=0 -poseidon-cache=%t/c -poseidon-print %s -o %t/use.ptx 2> %t/use.err
// RUN: awk '/^.visible .entry _Z(5other|10unprofiled)/{p=1} p{print} /^}/{p=0}' %t/plain.ptx > %t/plain.fns
// RUN: awk '/^.visible .entry _Z(5other|10unprofiled)/{p=1} p{print} /^}/{p=0}' %t/use.ptx > %t/use.fns
// RUN: diff %t/plain.fns %t/use.fns
// RUN: FileCheck --check-prefix=FNS %s < %t/use.fns
// RUN: FileCheck --check-prefix=SOLVE --implicit-check-not=warning: %s < %t/use.err
// RUN: awk '/^.visible .entry _Z4site/{p=1} p{print} /^}/{p=0}' %t/use.ptx | FileCheck --check-prefix=USE-SITE --implicit-check-not=fma.rn.f64 %s
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O2 -poseidon-profile-use=%t/prof -poseidon-cost-model=%gpu_cost_model -poseidon-enable-herbie=0 -poseidon-cache=%t/c %s -o %t/use
// RUN: %t/use | FileCheck --check-prefix=OUT %s

// In a module where a site is rewritten, every function no rewrite reached is
// the one as written: a kernel that is no site, and a site the profiling run
// never reached, inline libdevice's square root as the plain build does,
// although the solve kept it out of line in the whole module. The site itself
// is rewritten to FP32.

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

// SOLVE: [poseidon] site site(double const*, double*, int): accuracy target 1e-06
// SOLVE-NEXT: [poseidon] Optimizing [[BODY:preprocess__Z4sitePKdPdi_poseidon_body]] with relative error tolerance: 1.000000e-06
// SOLVE: [poseidon] [[BODY]]: tau=1.000000e-06 S={{[^ ]+}} A0={{[^ ]+}} selected cost={{[^ ]+}} accCost={{[^ ]+}} rel={{[^ ]+}}{{$}}
// SOLVE: Applying solution for CS: All FP64(0%) + FP32(100%) (#0)
// SOLVE: [poseidon] pin: _Z4sitePKdPdi:PT0=All FP64(0%) + FP32(100%){{$}}
// SOLVE: [poseidon] _Z10unprofiledPKdPdi_poseidon_body: the profiling run instrumented this site and never executed it ({{.*}}/preprocess__Z10unprofiledPKdPdi_poseidon_body.fpprofile); left unchanged
// SOLVE-NEXT: [poseidon] {{.*}}rewritten_module_as_written.cu: 1 function(s) a rewrite reached; every other function as written

// Rewritten, y = (3 x + 0.5) x + 1 is two FMAs in FP32 between a narrowing of
// x and a widening of the result, and no FP64 FMA is left.
// USE-SITE: .entry _Z4sitePKdPdi(
// USE-SITE: cvt.rn.f32.f64 [[XF:%[a-z]+[0-9]+]], {{%[a-z]+[0-9]+}};
// USE-SITE-NEXT: fma.rn.f32 [[T:%[a-z]+[0-9]+]], [[XF]], 0f40400000, 0f3F000000;
// USE-SITE-NEXT: fma.rn.f32 [[R:%[a-z]+[0-9]+]], [[T]], [[XF]], 0f3F800000;
// USE-SITE-NEXT: cvt.f64.f32 [[RD:%[a-z]+[0-9]+]], [[R]];
// USE-SITE-NEXT: st.global.{{[bf]}}64 {{\[}}{{%[a-z]+[0-9]+}}{{\]}}, [[RD]];

// FNS: .entry _Z5otherPKdPdi
// FNS-NOT: call
// FNS: sqrt.rn.f64
// FNS: .entry _Z10unprofiledPKdPdi
// FNS-NOT: call
// FNS: sqrt.rn.f64
