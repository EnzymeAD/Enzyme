// A rewrite calls a libdevice function the module already calls elsewhere.
// GlobalOpt has moved that function to fastcc by then, since the site is named
// with -poseidon-kernels and poseidon.h, which keeps it addressed, is not
// included; a call in the C convention would be undefined, and InstCombine
// would delete it and the rewritten site with it.
//
// REQUIRES: enzyme, cuda-device
// RUN: rm -rf %t && mkdir %t
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O2 \
// RUN:     -poseidon-kernels=waved -poseidon-profile-generate -poseidon-cache=%t/pcache %s -o %t/exe
// RUN: env POSEIDON_PROFILE_DIR=%t/profile %t/exe \
// RUN:   | FileCheck --check-prefix=PRIMAL %s
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O2 \
// RUN:     -poseidon-kernels=waved -poseidon-profile-use=%t/profile -poseidon-early-prune \
// RUN:     -poseidon-cache=%t/cache -poseidon-cost-model=%gpu_cost_model \
// RUN:     -poseidon-print -poseidon-enable-rewrite=0 -poseidon-tau=1e-3 \
// RUN:     %s -o %t/opt.exe 2>&1 | FileCheck --check-prefix=SOLVE %s
// RUN: %t/opt.exe | FileCheck --check-prefix=PRIMAL %s

#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

#include "poseidon/poseidon.h"

__global__ void waved(double *o, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double x = a[i];
    o[i] = exp(-x) * x + x;
  }
}

// Not a site: the module calls __nv_expf, directly only.
__global__ void wavef(float *o, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    o[i] = expf(-a[i]);
}

int main() {
  const int n = 4096;
  double *a, *o;
  float *af, *of;
  cudaMallocManaged(&a, n * sizeof(double));
  cudaMallocManaged(&o, n * sizeof(double));
  cudaMallocManaged(&af, n * sizeof(float));
  cudaMallocManaged(&of, n * sizeof(float));
  for (int i = 0; i < n; ++i) {
    a[i] = 0.25 + 2.0 * i / n;
    af[i] = (float)a[i];
  }
  waved<<<n / 128, 128>>>(o, a, n);
  wavef<<<n / 128, 128>>>(of, af, n);
  cudaDeviceSynchronize();
  double err = 0;
  for (int i = 0; i < n; ++i) {
    double x = a[i], ref = std::exp(-x) * x + x;
    err = std::fmax(err, std::fabs(o[i] - ref) / std::fabs(ref));
  }
  printf(err < 1e-3 ? "WAVED-PASS\n" : "WAVED-FAIL %g\n", err);
  return 0;
}

// PRIMAL: WAVED-PASS

// The rewritten body narrows x once and calls the module's __nv_expf on
// its negation.
// SOLVE: [poseidon] Optimizing [[BODY:preprocess__Z5wavedPdPKdi_poseidon_body]] with relative error tolerance: 1.000000e-03
// SOLVE: Applying solution for precision change All FP64(0%) + FP32(100%) (#0)
// SOLVE: [poseidon] pin: _Z5wavedPdPKdi:P0=All FP64(0%) + FP32(100%){{$}}
// SOLVE: define internal void @[[BODY]](
// SOLVE: [[XF:%[0-9]+]] = fptrunc double %{{[0-9]+}} to float
// SOLVE-NEXT: [[NEG:%[0-9]+]] = fneg float [[XF]]
// SOLVE-NEXT: {{%[0-9]+}} = {{(tail )?}}call float @__nv_expf(float [[NEG]])
// SOLVE: [poseidon] {{.*}}rewrite_existing_callee.cu: 1 function(s) a rewrite reached; every other function as written
