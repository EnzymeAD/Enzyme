// A rewrite calls a libdevice function the module already calls elsewhere.
// GlobalOpt has moved that function to fastcc by then, since the site is named
// with -poseidon-kernels and poseidon.h, which keeps it addressed, is not
// included; a call in the C convention would be undefined, and InstCombine
// would delete it and the rewritten site with it.
//
// REQUIRES: enzyme, cuda-runtime
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O2 \
// RUN:     -poseidon-kernels=waved -poseidon-profile-generate -poseidon-cache=%t.pcache %s -o %t.exe
// RUN: rm -rf %t.profile && POSEIDON_PROFILE_DIR=%t.profile %t.exe \
// RUN:   | FileCheck --check-prefix=PRIMAL %s
// RUN: rm -rf %t.cache && %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O2 \
// RUN:     -poseidon-kernels=waved -poseidon-profile-use=%t.profile \
// RUN:     -poseidon-cache=%t.cache -poseidon-cost-model=%gpu_cost_model \
// RUN:     -poseidon-print -poseidon-enable-herbie=0 -poseidon-tau=1e-3 \
// RUN:     %s -o %t.opt.exe 2>&1 | FileCheck --check-prefix=SOLVE %s
// RUN: %t.opt.exe | FileCheck --check-prefix=PRIMAL %s

#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

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

// SOLVE: Applying solution for CS: {{.*}}FP32
