// A device site whose math is libdevice calls, through poseidon-clang++: the
// profile is generated with Enzyme's plugin and used without it, so the form
// the profile names must be the form the use compile sees.
//
// REQUIRES: enzyme, cuda-runtime
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O2 \
// RUN:     -poseidon-profile-generate -poseidon-cache=%t.pcache %s -o %t.exe
// RUN: rm -rf %t.profile && POSEIDON_PROFILE_DIR=%t.profile %t.exe \
// RUN:   | FileCheck --check-prefix=PRIMAL %s
// RUN: rm -rf %t.cache && %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O2 \
// RUN:     -poseidon-profile-use=%t.profile -poseidon-cache=%t.cache \
// RUN:     -poseidon-cost-model=%gpu_cost_model -poseidon-print \
// RUN:     -poseidon-enable-herbie=0 -poseidon-tau=1e-6 %s -o %t.opt.exe 2> %t.use.log
// RUN: cat %t.profile/*.fpprofile %t.use.log | FileCheck %s
// RUN: %t.opt.exe | FileCheck --check-prefix=PRIMAL %s

#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

#include "poseidon/poseidon.h"

POSEIDON_OPTIMIZE __global__ void wave(float *o, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = a[i];
    o[i] = sqrtf(x) * expf(-x) + (1.0f - cosf(x)) / (x * x) +
           powf(x, 1.5f) * sinf(x);
  }
}

int main() {
  const int n = 4096;
  float *a, *o;
  cudaMallocManaged(&a, n * sizeof(float));
  cudaMallocManaged(&o, n * sizeof(float));
  for (int i = 0; i < n; ++i)
    a[i] = 0.25f + 2.0f * i / n;
  wave<<<n / 128, 128>>>(o, a, n);
  cudaDeviceSynchronize();
  double err = 0;
  for (int i = 0; i < n; ++i) {
    double x = a[i];
    double ref = std::sqrt(x) * std::exp(-x) + (1.0 - std::cos(x)) / (x * x) +
                 std::pow(x, 1.5) * std::sin(x);
    err = std::fmax(err, std::fabs(o[i] - ref) / std::fabs(ref));
  }
  printf(err < 1e-5 ? "WAVE-PASS\n" : "WAVE-FAIL %g\n", err);
  return 0;
}

// PRIMAL: WAVE-PASS

// CHECK: CanonicalHash = [[HASH:[0-9a-f]+]]
// CHECK: canonical form hash of preprocess__Z4wavePfPKfi_poseidon_body is [[HASH]]
