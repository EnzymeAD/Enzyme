// Three-component FP32 expansion on the device: a 24-step logistic map is
// profiled once and solved three ways; each binary's result is compared with
// a binary128 evaluation of the same map on the host. With two components a
// 1e-16 site tolerance keeps the FP64 body; admitting three components
// selects the Expansion3 rewrite, which lands orders of magnitude closer to
// the binary128 result than FP64 does; at 1e-15 the two-component rewrite
// is selected and is the least accurate of the three.
//
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O2 \
// RUN:     -poseidon-profile-generate %s -o %t.exe
// RUN: rm -rf %t.profile && POSEIDON_PROFILE_DIR=%t.profile %t.exe 1e-11 1e-8 \
// RUN:   | FileCheck --check-prefix=EXEC %s
//
// RUN: rm -rf %t.c2 && %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O2 \
// RUN:     -poseidon-profile-use=%t.profile -poseidon-cache=%t.c2 -poseidon-cost-model=%gpu_cost_model -poseidon-print -poseidon-enable-herbie=0 -poseidon-two-tier-step=100 -poseidon-enable-three-tier=false \
// RUN:     -poseidon-expansion-components=2 -poseidon-tau=1e-16 \
// RUN:     %s -o %t.two.exe 2>&1 | FileCheck --check-prefix=TWO %s
// RUN: %t.two.exe 1e-11 1e-8 | FileCheck --check-prefix=EXEC %s
//
// RUN: rm -rf %t.c3 && %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O2 \
// RUN:     -poseidon-profile-use=%t.profile -poseidon-cache=%t.c3 -poseidon-cost-model=%gpu_cost_model -poseidon-print -poseidon-enable-herbie=0 -poseidon-two-tier-step=100 -poseidon-enable-three-tier=false \
// RUN:     -poseidon-expansion-components=3 -poseidon-tau=1e-16 \
// RUN:     %s -o %t.three.exe 2>&1 | FileCheck --check-prefix=THREE %s
// RUN: %t.three.exe 0 1e-13 | FileCheck --check-prefix=EXEC %s
//
// RUN: rm -rf %t.cl && %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O2 \
// RUN:     -poseidon-profile-use=%t.profile -poseidon-cache=%t.cl -poseidon-cost-model=%gpu_cost_model -poseidon-print -poseidon-enable-herbie=0 -poseidon-two-tier-step=100 -poseidon-enable-three-tier=false \
// RUN:     -poseidon-expansion-components=3 -poseidon-tau=1e-15 \
// RUN:     %s -o %t.loose.exe 2>&1 | FileCheck --check-prefix=LOOSE %s
// RUN: %t.loose.exe 1e-9 1e-6 | FileCheck --check-prefix=EXEC %s
//
// REQUIRES: poseidon, enzyme, cuda-runtime

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <poseidon/poseidon.h>

constexpr int kSteps = 24;

POSEIDON_OPTIMIZE __global__ void logistic_kernel(const double *x0,
                                                  double *out) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  double x = x0[i];
#pragma unroll
  for (int k = 0; k < kSteps; ++k)
    x = 3.9 * x * (1.0 - x);
  out[i] = x;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s <min error> <max error>\n", argv[0]);
    return 2;
  }
  const double lo = std::atof(argv[1]), hi = std::atof(argv[2]);
  constexpr int N = 256;
  double hx[N], hout[N];
  for (int i = 0; i < N; ++i)
    hx[i] = 0.2 + 0.6 * i / N;
  double *dx, *dout;
  if (cudaMalloc(&dx, sizeof(hx)) != cudaSuccess ||
      cudaMalloc(&dout, sizeof(hout)) != cudaSuccess)
    return 1;
  cudaMemcpy(dx, hx, sizeof(hx), cudaMemcpyHostToDevice);
  logistic_kernel<<<4, 64>>>(dx, dout);
  if (cudaMemcpy(hout, dout, sizeof(hout), cudaMemcpyDeviceToHost) !=
      cudaSuccess)
    return 1;
  double maxRel = 0.0;
  for (int i = 0; i < N; ++i) {
    __float128 x = hx[i];
    for (int k = 0; k < kSteps; ++k)
      x = (__float128)3.9 * x * (1 - x);
    double ref = (double)x;
    maxRel = std::fmax(maxRel, std::fabs(hout[i] - ref) / std::fabs(ref));
  }
  printf("max relative error vs binary128 %.3e\n", maxRel);
  bool ok = maxRel >= lo && maxRel <= hi;
  printf(ok ? "LOGISTIC-PASS\n" : "LOGISTIC-FAIL\n");
  return ok ? 0 : 1;
}

// EXEC: max relative error vs binary128
// EXEC-NEXT: LOGISTIC-PASS

// TWO: no rewrite applied for preprocess__Z15logistic_kernelPKdPd_poseidon_body
// TWO-NOT: Applying solution

// THREE: Applying solution for CS: All FP64(0%) + Expansion3(100%)

// LOOSE: Applying solution for CS: All FP64(0%) + Expansion2(100%)
