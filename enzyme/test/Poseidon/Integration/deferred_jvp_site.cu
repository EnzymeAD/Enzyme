// RUN: rm -rf %t.cache && %clang -x cuda --cuda-gpu-arch=%gpu_arch -fcuda-rdc -O2 -ffp-contract=on \
// RUN:     %clangLoadPoseidonEnzyme -mllvm --poseidon-profile-generate \
// RUN:     -mllvm --poseidon-cache=%t.cache -mllvm --poseidon-print \
// RUN:     -I%FPProfileInc %s %FPProfileCUDASrc -x none %FPProfileLib \
// RUN:     -L/usr/local/cuda/lib64 -lcudart -lstdc++ -lm -o %t.exe 2>&1 \
// RUN:   | FileCheck --check-prefix=DEFER %s
// RUN: rm -rf %t.profiles && POSEIDON_PROFILE_DIR=%t.profiles %t.exe \
// RUN:   | FileCheck --check-prefix=PRIMAL %s
// RUN: cat %t.profiles/preprocess_jvp_body.fpprofile \
// RUN:   | FileCheck --check-prefix=PROFILE %s
//
// RUN: rm -rf %t.solve && %clang -x cuda --cuda-gpu-arch=%gpu_arch -O2 -ffp-contract=on \
// RUN:     %clangLoadPoseidonEnzyme -mllvm --poseidon-profile-use=%t.profiles \
// RUN:     -mllvm --poseidon-cache=%t.solve -mllvm --poseidon-print \
// RUN:     -mllvm --poseidon-cost-model=%gpu_cost_model \
// RUN:     -mllvm --poseidon-enable-herbie=0 \
// RUN:     %s -L/usr/local/cuda/lib64 -lcudart -lstdc++ -lm -o %t.opt.exe 2>&1 \
// RUN:   | FileCheck --check-prefix=SOLVE %s
// RUN: %t.opt.exe | FileCheck --check-prefix=OPT %s
//
// REQUIRES: poseidon, enzyme, cuda-runtime

// A site whose marked body is nothing but a forward-mode derivative request.
// Before Enzyme has run, that body holds no floating-point arithmetic at all,
// so the early run has nothing to profile and nothing to rewrite; the site is
// deferred to the late run, which sees the tangent code Enzyme generated for
// it and profiles and solves that.

#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

template <typename R, typename... T>
__device__ R __poseidon_fp_optimize(void *, T...);
template <typename R, typename... T> __device__ R __enzyme_fwddiff(void *, T...);
__device__ int enzyme_dup;
__device__ int poseidon_tau;

// The primal point function. Nothing marks it; it reaches Poseidon only
// through the tangent code Enzyme generates for it.
extern "C" __device__ __attribute__((noinline)) void
qf(const double *x, double *y) {
  double a = x[0], b = x[1], c = x[2];
  double p = a * b + c;
  double q = a * a + b * b + c * c;
  double r = p * q + a * c;
  y[0] = r * q + p * p;
  y[1] = r * p - q * c;
}

// The marked body: one __enzyme_fwddiff call and nothing else.
extern "C" __device__ __attribute__((noinline)) void
jvp_body(const double *x, const double *dx, double *y, double *dy) {
  __enzyme_fwddiff<void>((void *)qf, enzyme_dup, x, dx, enzyme_dup, y, dy);
}

#define NPT 4096

__global__ void jvp_kernel(const double *x, const double *dx, double *y,
                           double *dy) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= NPT)
    return;
  // Reverse-mode shadows for the profiling run: the input slots accumulate the
  // gradient and start at zero, the output slots carry the seed.
  double x_sh[3] = {0.0, 0.0, 0.0};
  double dx_sh[3] = {0.0, 0.0, 0.0};
  double y_sh[2] = {1.0, 1.0};
  double dy_sh[2] = {1.0, 1.0};
  __poseidon_fp_optimize<void>((void *)jvp_body, poseidon_tau, 1e-6,
                               enzyme_dup, x + 3 * i, x_sh, enzyme_dup,
                               dx + 3 * i, dx_sh, enzyme_dup, y + 2 * i, y_sh,
                               enzyme_dup, dy + 2 * i, dy_sh);
}

#define CK(e)                                                                  \
  do {                                                                         \
    cudaError_t err = (e);                                                     \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA error %s at line %d\n", cudaGetErrorString(err), __LINE__); \
      return 1;                                                                \
    }                                                                          \
  } while (0)

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
  CK(cudaMalloc(&x, 3 * NPT * sizeof(double)));
  CK(cudaMalloc(&dx, 3 * NPT * sizeof(double)));
  CK(cudaMalloc(&y, 2 * NPT * sizeof(double)));
  CK(cudaMalloc(&dy, 2 * NPT * sizeof(double)));
  CK(cudaMemcpy(x, hx, 3 * NPT * sizeof(double), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dx, hdx, 3 * NPT * sizeof(double), cudaMemcpyHostToDevice));
  CK(cudaMemset(y, 0, 2 * NPT * sizeof(double)));
  CK(cudaMemset(dy, 0, 2 * NPT * sizeof(double)));
  jvp_kernel<<<(NPT + 63) / 64, 64>>>(x, dx, y, dy);
  CK(cudaDeviceSynchronize());
  double *hy = new double[2 * NPT], *hdy = new double[2 * NPT];
  CK(cudaMemcpy(hy, y, 2 * NPT * sizeof(double), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hdy, dy, 2 * NPT * sizeof(double), cudaMemcpyDeviceToHost));
  printf("y = %.4f %.4f\n", hy[0], hy[1]);
  printf("dy = %.4f %.4f\n", hdy[0], hdy[1]);
  // The tangent is what the rewrite is about, so its accuracy is what the run
  // reports: relative l2 against the same tangent evaluated in double on the
  // host.
  double num = 0.0, den = 0.0;
  for (int i = 0; i < NPT; ++i) {
    double a = hx[3 * i], b = hx[3 * i + 1], c = hx[3 * i + 2];
    double da = hdx[3 * i], db = hdx[3 * i + 1], dc = hdx[3 * i + 2];
    double p = a * b + c, dp = da * b + a * db + dc;
    double q = a * a + b * b + c * c, dq = 2 * a * da + 2 * b * db + 2 * c * dc;
    double r = p * q + a * c, dr = dp * q + p * dq + da * c + a * dc;
    double ref[2] = {dr * q + r * dq + 2 * p * dp,
                     dr * p + r * dp - dq * c - q * dc};
    for (int k = 0; k < 2; ++k) {
      double d = hdy[2 * i + k] - ref[k];
      num += d * d;
      den += ref[k] * ref[k];
    }
  }
  printf("jvp_rel_l2 = %.3e\n", den > 0 ? sqrt(num / den) : 0.0);
  return 0;
}

// The early run leaves the site alone and says so; the late run folds in the
// code Enzyme generated for the request.
// DEFER: [poseidon] jvp_body: marked body still holds an __enzyme_* request; deferred to the late run
// DEFER: [poseidon] jvp_body: folded 1 Enzyme-generated call(s) into the deferred body
// DEFER: [poseidon] block-geometry probe placed in 1 profiled function(s)

// PRIMAL: y = 0.7949 0.0631
// PRIMAL-NEXT: dy = 1.1347 0.5354
// PRIMAL-NEXT: jvp_rel_l2 = {{[0-9.]+e-1[5-9]}}

// The profile is keyed on the marked body and carries the tangent code's slots.
// PROFILE: SiteId = 0
// PROFILE: Exec = 4096

// The late run solves it like any other site, under the marker's own target.
// SOLVE: [poseidon] jvp_body: marked body still holds an __enzyme_* request; deferred to the late run
// SOLVE: [poseidon] jvp_body: folded 1 Enzyme-generated call(s) into the deferred body
// SOLVE: [poseidon] Optimizing preprocess_jvp_body with relative error tolerance: 1.000000e-06
// SOLVE: [poseidon] preprocess_jvp_body: tau=1.000000e-06
// SOLVE: Applying solution

// The rewritten tangent still computes the JVP, to the target it was given.
// OPT: jvp_rel_l2 = {{[0-9.]+e-0[6789]}}
