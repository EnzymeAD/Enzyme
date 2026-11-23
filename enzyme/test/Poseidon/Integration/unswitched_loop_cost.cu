// Precision candidates of a kernel whose loops are strided by thread index up
// to a bound the kernel is given, priced and applied end to end. The guard of
// the inner loop is invariant in the outer one, so the simplified clone each
// candidate is priced on unswitches the outer loop and keeps the body in one
// version only; every candidate is priced there, none at zero, and at 1e-6 the
// cheapest candidate meeting the target, FP32, is applied and runs within
// FP32's error of the binary64 result.
//
// REQUIRES: enzyme, cuda-runtime
// RUN: rm -rf %t && mkdir %t
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O2 -ffp-contract=on \
// RUN:     -poseidon-profile-generate -poseidon-cache=%t/pcache %s -o %t/exe
// RUN: env POSEIDON_PROFILE_DIR=%t/profile %t/exe 0 1e-15 \
// RUN:   | FileCheck --check-prefix=EXEC %s
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O2 -ffp-contract=on \
// RUN:     -poseidon-profile-use=%t/profile -poseidon-cache=%t/c \
// RUN:     -poseidon-cost-model=%gpu_cost_model -poseidon-print \
// RUN:     -poseidon-enable-herbie=0 -poseidon-tau=1e-6 %s -o %t/opt.exe 2>&1 \
// RUN:   | FileCheck --check-prefix=SOLVE %s
// RUN: %t/opt.exe 1e-9 1e-5 | FileCheck --check-prefix=EXEC %s

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <poseidon/poseidon.h>

POSEIDON_OPTIMIZE __global__ void setup_kernel(int Q, const double *W,
                                               const double *J, double *D) {
  const int e = blockIdx.x;
  for (int qy = threadIdx.y; qy < Q; qy += blockDim.y)
    for (int qx = threadIdx.x; qx < Q; qx += blockDim.x) {
      const int q = qx + Q * qy;
      const double *j = J + 4 * (q + Q * Q * e);
      const double det = j[0] * j[3] - j[1] * j[2];
      D[q + Q * Q * e] = W[q] * (j[0] * j[0] + j[1] * j[1]) / det;
    }
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s <min error> <max error>\n", argv[0]);
    return 2;
  }
  const double lo = std::atof(argv[1]), hi = std::atof(argv[2]);
  constexpr int Q = 4, NE = 64, NQ = Q * Q;
  static double hW[NQ], hJ[4 * NQ * NE], hD[NQ * NE];
  for (int q = 0; q < NQ; ++q)
    hW[q] = 0.1 + 0.01 * q;
  for (int i = 0; i < NQ * NE; ++i)
    for (int k = 0; k < 4; ++k)
      hJ[4 * i + k] = (k % 3 == 0 ? 1.0 : 0.0) + 0.001 * ((7 * i + k) % 13);
  double *W, *J, *D;
  if (cudaMalloc(&W, sizeof(hW)) != cudaSuccess ||
      cudaMalloc(&J, sizeof(hJ)) != cudaSuccess ||
      cudaMalloc(&D, sizeof(hD)) != cudaSuccess)
    return 1;
  cudaMemcpy(W, hW, sizeof(hW), cudaMemcpyHostToDevice);
  cudaMemcpy(J, hJ, sizeof(hJ), cudaMemcpyHostToDevice);
  setup_kernel<<<NE, dim3(Q, Q)>>>(Q, W, J, D);
  if (cudaMemcpy(hD, D, sizeof(hD), cudaMemcpyDeviceToHost) != cudaSuccess)
    return 1;
  double maxRel = 0.0;
  for (int e = 0; e < NE; ++e)
    for (int q = 0; q < NQ; ++q) {
      const double *j = hJ + 4 * (q + NQ * e);
      double ref = hW[q] * (j[0] * j[0] + j[1] * j[1]) /
                   (j[0] * j[3] - j[1] * j[2]);
      maxRel = std::fmax(maxRel, std::fabs(hD[q + NQ * e] - ref) /
                                     std::fabs(ref));
    }
  printf("max relative error vs binary64 %.3e\n", maxRel);
  bool ok = maxRel >= lo && maxRel <= hi;
  printf(ok ? "SETUP-PASS\n" : "SETUP-FAIL\n");
  return ok ? 0 : 1;
}

// EXEC: max relative error vs binary64
// EXEC-NEXT: SETUP-PASS

// SOLVE-NOT: cand=0.000000e+00
// SOLVE: CS candidate 'All FP64(0%) + FP32(100%)'
// SOLVE-NOT: cand=0.000000e+00
// SOLVE: Applying solution for CS: All FP64(0%) + FP32(100%) (#0)
