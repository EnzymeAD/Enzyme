// REQUIRES: enzyme, cuda-runtime
// RUN: rm -rf %t && mkdir -p %t
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O2 -poseidon-profile-generate -poseidon-cache=%t/cg -poseidon-print %s -o %t/gen 2>&1 \
// RUN:   | FileCheck --check-prefix=DEFER %s
// RUN: env POSEIDON_PROFILE_DIR=%t/prof %t/gen | FileCheck --check-prefix=PRIMAL %s
// RUN: FileCheck --check-prefix=PROFILE %s < %t/prof/preprocess_jvp_body.poseidon._Z8jvp_callPKdS0_PdS1_.0.fpprofile
//
// The call form records what the explicit form with the shadows written by
// hand records. The sums are accumulated by atomics and differ between any two
// runs in their last digits, so they are compared to a relative 1e-12.
// RUN: grep -v -e SiteId -e Sum %t/prof/preprocess_jvp_body.fpprofile > %t/explicit.prof
// RUN: grep -v -e SiteId -e Sum %t/prof/preprocess_jvp_body.poseidon._Z8jvp_callPKdS0_PdS1_.0.fpprofile > %t/call.prof
// RUN: diff %t/explicit.prof %t/call.prof
// RUN: grep Sum %t/prof/preprocess_jvp_body.fpprofile > %t/explicit.sums
// RUN: grep Sum %t/prof/preprocess_jvp_body.poseidon._Z8jvp_callPKdS0_PdS1_.0.fpprofile > %t/call.sums
// RUN: paste %t/explicit.sums %t/call.sums \
// RUN:   | awk '{ d = $3 - $6; if (d * d > 1e-24 * $3 * $3) bad = 1 } END { print bad ? "SUMS DIFFER" : "SUMS SAME" }' \
// RUN:   | FileCheck --check-prefix=SUMS %s
//
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O2 -poseidon-profile-use=%t/prof -poseidon-cost-model=%gpu_cost_model -poseidon-enable-herbie=0 -poseidon-print -poseidon-cache=%t/c %s -o %t/opt 2>&1 \
// RUN:   | FileCheck --check-prefix=SOLVE %s
// RUN: %t/opt | FileCheck --check-prefix=OPT %s

// deferred_jvp_site.cu with the site written as a call: the body is one
// __enzyme_fwddiff request, so the call is a deferred site, profiled and solved
// by the late run on the tangent code Enzyme generated for it. Its pointers
// point into local objects of the kernel, whose shadows the pass makes.

#include "poseidon/poseidon.h"
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

template <typename R, typename... T>
__device__ R __poseidon_fp_optimize(void *, T...);
template <typename R, typename... T> __device__ R __enzyme_fwddiff(void *, T...);
__device__ int enzyme_dup;
__device__ int poseidon_tau;

extern "C" __device__ __attribute__((noinline)) void
qf(const double *x, double *y) {
  double a = x[0], b = x[1], c = x[2];
  double p = a * b + c;
  double q = a * a + b * b + c * c;
  double r = p * q + a * c;
  y[0] = r * q + p * p;
  y[1] = r * p - q * c;
}

extern "C" __device__ void jvp_body(const double *x, const double *dx,
                                    double *y, double *dy) {
  __enzyme_fwddiff<void>((void *)qf, enzyme_dup, x, dx, enzyme_dup, y, dy);
}

#define NPT 4096

__device__ void load(const double *x, const double *dx, int i, double *xl,
                     double *dxl) {
  for (int k = 0; k < 3; ++k) {
    xl[k] = x[3 * i + k];
    dxl[k] = dx[3 * i + k];
  }
}

__global__ void jvp_call(const double *x, const double *dx, double *y,
                         double *dy) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= NPT)
    return;
  double xl[3], dxl[3], yl[2], dyl[2];
  load(x, dx, i, xl, dxl);
  __poseidon_optimize_tau(1e-6, jvp_body, xl, dxl, yl, dyl);
  for (int k = 0; k < 2; ++k) {
    y[2 * i + k] = yl[k];
    dy[2 * i + k] = dyl[k];
  }
}

__global__ void jvp_explicit(const double *x, const double *dx, double *y,
                             double *dy) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= NPT)
    return;
  double xl[3], dxl[3], yl[2], dyl[2];
  load(x, dx, i, xl, dxl);
  double x_sh[3] = {0.0, 0.0, 0.0};
  double dx_sh[3] = {0.0, 0.0, 0.0};
  double y_sh[2] = {1.0, 1.0};
  double dy_sh[2] = {1.0, 1.0};
  __poseidon_fp_optimize<void>((void *)jvp_body, poseidon_tau, 1e-6,
                               enzyme_dup, xl, x_sh, enzyme_dup, dxl, dx_sh,
                               enzyme_dup, yl, y_sh, enzyme_dup, dyl, dy_sh);
  for (int k = 0; k < 2; ++k) {
    y[2 * i + k] = yl[k];
    dy[2 * i + k] = dyl[k];
  }
}

int main() {
  double *hx = new double[3 * NPT], *hdx = new double[3 * NPT];
  for (int i = 0; i < NPT; ++i) {
    double t = 0.25 + 1.5 * (double)(i % 97) / 97.0;
    hx[3 * i + 0] = 1.25 * t;
    hx[3 * i + 1] = 0.75 + 0.5 * t;
    hx[3 * i + 2] = 0.5 * t + 0.125;
    hdx[3 * i + 0] = 0.5 + 0.25 * t;
    hdx[3 * i + 1] = -0.25 - 0.125 * t;
    hdx[3 * i + 2] = 0.125 + 0.0625 * t;
  }
  double *x, *dx, *y, *dy;
  cudaMalloc(&x, 3 * NPT * sizeof(double));
  cudaMalloc(&dx, 3 * NPT * sizeof(double));
  cudaMalloc(&y, 2 * NPT * sizeof(double));
  cudaMalloc(&dy, 2 * NPT * sizeof(double));
  cudaMemcpy(x, hx, 3 * NPT * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dx, hdx, 3 * NPT * sizeof(double), cudaMemcpyHostToDevice);
  double *hy = new double[2 * NPT], *hdy = new double[2 * NPT];
  for (int form = 0; form < 2; ++form) {
    if (form == 0)
      jvp_call<<<(NPT + 63) / 64, 64>>>(x, dx, y, dy);
    else
      jvp_explicit<<<(NPT + 63) / 64, 64>>>(x, dx, y, dy);
    cudaMemcpy(hy, y, 2 * NPT * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(hdy, dy, 2 * NPT * sizeof(double), cudaMemcpyDeviceToHost);
    double num = 0.0, den = 0.0;
    for (int i = 0; i < NPT; ++i) {
      double a = hx[3 * i], b = hx[3 * i + 1], c = hx[3 * i + 2];
      double da = hdx[3 * i], db = hdx[3 * i + 1], dc = hdx[3 * i + 2];
      double p = a * b + c, dp = da * b + a * db + dc;
      double q = a * a + b * b + c * c,
             dq = 2 * a * da + 2 * b * db + 2 * c * dc;
      double r = p * q + a * c, dr = dp * q + p * dq + da * c + a * dc;
      double ref[2] = {dr * q + r * dq + 2 * p * dp,
                       dr * p + r * dp - dq * c - q * dc};
      for (int k = 0; k < 2; ++k) {
        double d = hdy[2 * i + k] - ref[k];
        num += d * d;
        den += ref[k] * ref[k];
      }
    }
    printf("%s: y = %.4f %.4f dy = %.4f %.4f jvp_rel_l2 = %.3e\n",
           form == 0 ? "call" : "explicit", hy[0], hy[1], hdy[0], hdy[1],
           den > 0 ? sqrt(num / den) : 0.0);
  }
  return 0;
}

// DEFER: [poseidon] site jvp_body at its call in jvp_call(double const*, double const*, double*, double*): accuracy target 1e-06
// DEFER-DAG: [poseidon] jvp_body.poseidon._Z8jvp_callPKdS0_PdS1_.0: marked body still holds an __enzyme_* request; deferred to the late run
// DEFER-DAG: [poseidon] jvp_body: marked body still holds an __enzyme_* request; deferred to the late run
// DEFER-DAG: [poseidon] jvp_body.poseidon._Z8jvp_callPKdS0_PdS1_.0: folded 1 Enzyme-generated call(s) into the deferred body
// DEFER-DAG: [poseidon-profgen] jvp_body at call 0 in jvp_call(double const*, double const*, double*, double*): 4 shadow object(s), output(s) 2 3

// PRIMAL: call: y = 0.7949 0.0631 dy = 1.1347 0.5354 jvp_rel_l2 = {{[0-9.]+e-1[5-9]}}
// PRIMAL-NEXT: explicit: y = 0.7949 0.0631 dy = 1.1347 0.5354 jvp_rel_l2 = {{[0-9.]+e-1[5-9]}}

// PROFILE: Exec = 4096

// SUMS: SUMS SAME

// SOLVE-DAG: [poseidon] jvp_body.poseidon._Z8jvp_callPKdS0_PdS1_.0: folded 1 Enzyme-generated call(s) into the deferred body
// SOLVE-DAG: [poseidon] Optimizing preprocess_jvp_body.poseidon._Z8jvp_callPKdS0_PdS1_.0 with relative error tolerance: 1.000000e-06
// SOLVE-DAG: [poseidon] Optimizing preprocess_jvp_body with relative error tolerance: 1.000000e-06

// OPT: call: {{.*}} jvp_rel_l2 = {{[0-9.]+e-0[6789]}}
// OPT-NEXT: explicit: {{.*}} jvp_rel_l2 = {{[0-9.]+e-0[6789]}}
