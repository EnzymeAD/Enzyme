// REQUIRES: enzyme, cuda-runtime
// RUN: rm -rf %t && mkdir -p %t
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch --cuda-device-only -O2 -S -emit-llvm %loadClangEnzyme %s -o %t/plain.ll
//
// Every site left alone (no profile): the module is the one as written.
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch --cuda-device-only -O2 -S -emit-llvm %loadClangEnzyme -poseidon-profile-use=%S/../Inputs/other_sites %s -o %t/none.ll
// RUN: diff %t/plain.ll %t/none.ll
//
// One site rewritten and one refused: the refused one's kernel and tangent are
// the ones as written.
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O2 -poseidon-profile-generate -poseidon-cache=%t/cg %s -o %t/gen
// RUN: env POSEIDON_PROFILE_DIR=%t/prof %t/gen
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch --cuda-device-only -O2 -S -emit-llvm %loadClangEnzyme -poseidon-profile-use=%t/prof -poseidon-cost-model=%gpu_cost_model -poseidon-enable-herbie=0 -poseidon-cache=%t/c -poseidon-print %s -o %t/left.ll 2>&1 | FileCheck --check-prefix=SOLVE %s
// RUN: awk '/^define .*@(_Z5jvp_bPKdS0_PdS1_|fwddiffe_Z2qfPKdPd)\(/{p=1} p{print} /^}/{p=0}' %t/plain.ll | sed -E 's/#[0-9]+/#/g; s/![0-9]+/!/g' > %t/plain.fns
// RUN: awk '/^define .*@(_Z5jvp_bPKdS0_PdS1_|fwddiffe_Z2qfPKdPd)\(/{p=1} p{print} /^}/{p=0}' %t/left.ll | sed -E 's/#[0-9]+/#/g; s/![0-9]+/!/g' > %t/left.fns
// RUN: diff %t/plain.fns %t/left.fns
// RUN: FileCheck --check-prefix=FNS %s < %t/left.fns

// A deferred site the solve leaves alone, in a module where another site is
// rewritten, emits its body and the code Enzyme generated for it as the
// pipeline left them: Enzyme's tangent of 1/sqrt(s) calls libdevice's
// __nv_sqrt once more, nothing inlines after Enzyme without a site, so the two
// square roots stay two calls; the late run must not inline and merge them.
// The program asks Enzyme for a derivative of its own, so it loads Enzyme's
// plugin itself where poseidon-clang++ does not, outside profile generation.

#include "poseidon/poseidon.h"
#include <cuda_runtime.h>

template <typename R, typename... T> __device__ R __enzyme_fwddiff(void *, T...);
__device__ int enzyme_dup;

__device__ __attribute__((noinline)) void qf(const double *x, double *y) {
  y[0] = x[0] / sqrt(1.0 + x[0] * x[0]);
}

__device__ void jvp_body(const double *x, const double *dx, double *y,
                         double *dy) {
  __enzyme_fwddiff<void>((void *)qf, enzyme_dup, x, dx, enzyme_dup, y, dy);
}

__global__ void jvp_a(const double *x, const double *dx, double *y,
                      double *dy) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double xl = x[i], dxl = dx[i], yl, dyl;
  __poseidon_optimize_tau(1e-6, jvp_body, &xl, &dxl, &yl, &dyl);
  y[i] = 2.0 * yl;
  dy[i] = 2.0 * dyl;
}

__global__ void jvp_b(const double *x, const double *dx, double *y,
                      double *dy) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double xl = x[i], dxl = dx[i], yl, dyl;
  __poseidon_optimize_tau(1e-30, jvp_body, &xl, &dxl, &yl, &dyl);
  y[i] = 3.0 * yl;
  dy[i] = 3.0 * dyl;
}

#define N 1024

int main() {
  double h[N];
  for (int i = 0; i < N; ++i)
    h[i] = 0.25 + 1.5 * (double)(i % 97) / 97.0;
  double *x, *dx, *y, *dy;
  cudaMalloc(&x, N * sizeof(double));
  cudaMalloc(&dx, N * sizeof(double));
  cudaMalloc(&y, N * sizeof(double));
  cudaMalloc(&dy, N * sizeof(double));
  cudaMemcpy(x, h, N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dx, h, N * sizeof(double), cudaMemcpyHostToDevice);
  jvp_a<<<N / 64, 64>>>(x, dx, y, dy);
  jvp_b<<<N / 64, 64>>>(x, dx, y, dy);
  cudaDeviceSynchronize();
  return 0;
}

// SOLVE-DAG: Applying solution for
// SOLVE-DAG: No solution found that meets accuracy tolerance 1.000000e-30
// FNS-LABEL: define {{.*}} @_Z5jvp_bPKdS0_PdS1_(
// FNS: call fastcc void @fwddiffe_Z2qfPKdPd(
// FNS-LABEL: define internal fastcc void @fwddiffe_Z2qfPKdPd(
// FNS-COUNT-2: call {{.*}} @__nv_sqrt(
