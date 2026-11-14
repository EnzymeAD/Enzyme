// The site is named with -poseidon-kernels and poseidon.h, which links the
// libdevice functions a rewrite may call, is not included. A rewrite that
// needs one the module does not define stops the compile with an error saying
// so, rather than ptxas failing on an unresolved function.
//
// REQUIRES: enzyme, cuda-runtime
//
// Single precision is cheapest at 1e-3, and nothing calls __nv_expf.
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O2 \
// RUN:     -poseidon-kernels=waved -poseidon-profile-generate -poseidon-cache=%t.pcache %s -o %t.exe
// RUN: rm -rf %t.profile && POSEIDON_PROFILE_DIR=%t.profile %t.exe
// RUN: rm -rf %t.cache && not %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O2 \
// RUN:     -poseidon-kernels=waved -poseidon-profile-use=%t.profile \
// RUN:     -poseidon-cache=%t.cache -poseidon-cost-model=%gpu_cost_model \
// RUN:     -poseidon-enable-herbie=0 -poseidon-tau=1e-3 %s -o %t.opt.exe 2>&1 \
// RUN:   | FileCheck --check-prefix=REWRITE --implicit-check-not=Unresolved %s

#include <cmath>
#include <cuda_runtime.h>

__global__ void waved(double *o, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double x = a[i];
    o[i] = exp(-x) * x + x;
  }
}

int main() {
  const int n = 4096;
  double *a, *o;
  cudaMallocManaged(&a, n * sizeof(double));
  cudaMallocManaged(&o, n * sizeof(double));
  for (int i = 0; i < n; ++i)
    a[i] = 0.25 + 2.0 * i / n;
  waved<<<n / 128, 128>>>(o, a, n);
  cudaDeviceSynchronize();
  return 0;
}

// REWRITE: Poseidon: a rewrite in waved(double*, double const*, int) calls __nv_expf, which this device module does not define.
