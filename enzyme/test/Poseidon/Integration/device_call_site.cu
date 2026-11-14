// REQUIRES: enzyme, cuda-runtime
// RUN: rm -rf %t && mkdir -p %t
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O3 -poseidon-profile-generate -poseidon-cache=%t/cg %s -o %t/gen 2> %t/gen.err
// RUN: FileCheck --check-prefix=GEN %s < %t/gen.err
// RUN: env POSEIDON_PROFILE_DIR=%t/prof %t/gen | FileCheck --check-prefix=PRIMAL %s
//
// The call form and the explicit form with the shadows written by hand record
// the same profile: the same ranges, execution counts and gradients. The sums
// are accumulated by atomics and differ between any two runs in their last
// digits, so they are compared to a relative 1e-12.
// RUN: grep -v -e SiteId -e Sum %t/prof/preprocess__Z9hooke_volRK4mat3fPf.fpprofile > %t/explicit.prof
// RUN: grep -v -e SiteId -e Sum %t/prof/preprocess__Z9hooke_volRK4mat3fPf.poseidon._Z8vol_callPKffPfi.0.fpprofile > %t/call.prof
// RUN: diff %t/explicit.prof %t/call.prof
// RUN: FileCheck --check-prefix=PROF %s < %t/call.prof
// RUN: grep Sum %t/prof/preprocess__Z9hooke_volRK4mat3fPf.fpprofile > %t/explicit.sums
// RUN: grep Sum %t/prof/preprocess__Z9hooke_volRK4mat3fPf.poseidon._Z8vol_callPKffPfi.0.fpprofile > %t/call.sums
// RUN: paste %t/explicit.sums %t/call.sums \
// RUN:   | awk '{ d = $3 - $6; if (d * d > 1e-24 * $3 * $3) bad = 1 } END { print bad ? "SUMS DIFFER" : "SUMS SAME" }' \
// RUN:   | FileCheck --check-prefix=SUMS %s
//
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O3 -poseidon-profile-use=%t/prof -poseidon-cost-model=%gpu_cost_model -poseidon-enable-herbie=0 -poseidon-print -poseidon-cache=%t/c %s -o %t/use 2> %t/use.err
// RUN: FileCheck --check-prefix=USE %s < %t/use.err
// RUN: %t/use | FileCheck --check-prefix=OUT %s

// __poseidon_optimize_tau in device code: the call is a site, profiled with
// shadows the pass makes for the local objects it passes, against the same
// call written with __poseidon_fp_optimize and shadows of the caller's own.
// The body is the volumetric response of a Neo-Hookean material near the
// reference configuration, where J - 1 cancels.

#include "poseidon/poseidon.h"
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

template <typename R, typename... T>
__device__ R __poseidon_fp_optimize(void *, T...);
__device__ int enzyme_const, enzyme_dup, poseidon_tau;

struct mat3 {
  float a[3][3];
};

__device__ float det3(const mat3 &m) {
  return m.a[0][0] * (m.a[1][1] * m.a[2][2] - m.a[1][2] * m.a[2][1]) -
         m.a[0][1] * (m.a[1][0] * m.a[2][2] - m.a[1][2] * m.a[2][0]) +
         m.a[0][2] * (m.a[1][0] * m.a[2][1] - m.a[1][1] * m.a[2][0]);
}

__device__ void hooke_vol(const mat3 &dudx, float D1, float *out) {
  mat3 F;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      F.a[i][j] = (i == j ? 1.0f : 0.0f) + dudx.a[i][j];
  float J = det3(F);
  float p = -2.0f * D1 * J * (J - 1.0f);
  *out = -(p / J);
}

__device__ mat3 load(const float *g, int i) {
  mat3 m;
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      m.a[r][c] = g[9 * i + 3 * r + c];
  return m;
}

__global__ void vol_call(const float *g, float D1, float *res, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  mat3 dudx = load(g, i);
  float out;
  __poseidon_optimize_tau(1e-6, hooke_vol, dudx, D1, &out);
  res[i] = out;
}

__global__ void vol_explicit(const float *g, float D1, float *res, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  mat3 dudx = load(g, i);
  float out, out_shadow = 1.0f;
  __poseidon_fp_optimize<void>((void *)hooke_vol, poseidon_tau, 1e-6,
                               enzyme_const, &dudx, D1, enzyme_dup, &out,
                               &out_shadow);
  res[i] = out;
}

int main() {
  const int n = 4096;
  const float D1 = 1.25f;
  float *hg = new float[9 * n];
  unsigned s = 12345u;
  for (int k = 0; k < 9 * n; ++k) {
    s = s * 1664525u + 1013904223u;
    hg[k] = 0.02f * ((float)(s >> 8) / 16777216.0f - 0.5f);
  }
  float *g, *rc, *re;
  cudaMalloc(&g, 9 * n * sizeof(float));
  cudaMalloc(&rc, n * sizeof(float));
  cudaMalloc(&re, n * sizeof(float));
  cudaMemcpy(g, hg, 9 * n * sizeof(float), cudaMemcpyHostToDevice);
  vol_call<<<(n + 127) / 128, 128>>>(g, D1, rc, n);
  vol_explicit<<<(n + 127) / 128, 128>>>(g, D1, re, n);
  float *hc = new float[n], *he = new float[n];
  cudaMemcpy(hc, rc, n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(he, re, n * sizeof(float), cudaMemcpyDeviceToHost);
  // 2 D1 (J - 1), with J - 1 = tr A + (tr(A)^2 - tr(A^2)) / 2 + det A in
  // double: free of the cancellation, so a reference for both kernels.
  double nc = 0, ne = 0, d = 0;
  for (int i = 0; i < n; ++i) {
    double A[3][3];
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c)
        A[r][c] = hg[9 * i + 3 * r + c];
    double tr = A[0][0] + A[1][1] + A[2][2], tr2 = 0;
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c)
        tr2 += A[r][c] * A[c][r];
    double det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
                 A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
                 A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
    double ref = 2.0 * D1 * (tr + 0.5 * (tr * tr - tr2) + det);
    nc += (hc[i] - ref) * (hc[i] - ref);
    ne += (he[i] - ref) * (he[i] - ref);
    d += ref * ref;
  }
  printf("call %.9g explicit %.9g\n", hc[7], he[7]);
  printf("call_rel_l2 = %.3e\nexplicit_rel_l2 = %.3e\n", sqrt(nc / d),
         sqrt(ne / d));
  return 0;
}

// PRIMAL: call [[V:[^ ]+]] explicit [[V]]
// PRIMAL-NEXT: call_rel_l2 = [[E:[^ ]+]]
// PRIMAL-NEXT: explicit_rel_l2 = [[E]]

// GEN: [poseidon-profgen] hooke_vol(mat3 const&, float, float*) at call 0 in vol_call(float const*, float, float*, int): 2 shadow object(s), output(s) 2

// PROF: Exec = 4096

// SUMS: SUMS SAME

// Both sites are solved alike.
// USE: [poseidon] site hooke_vol(mat3 const&, float, float*) at its call in vol_call(float const*, float, float*, int): accuracy target 1e-06
// USE: preprocess__Z9hooke_volRK4mat3fPf.poseidon._Z8vol_callPKffPfi.0: tau=1.000000e-06 [[PICK:.*]]
// USE: preprocess__Z9hooke_volRK4mat3fPf: tau=1.000000e-06 [[PICK]]

// OUT: call [[V:[^ ]+]] explicit [[V]]
// OUT-NEXT: call_rel_l2 = [[E:[^ ]+]]
// OUT-NEXT: explicit_rel_l2 = [[E]]
