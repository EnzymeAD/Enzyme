// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -S -emit-llvm -I%FPProfileInc %clangLoadPoseidonEnzyme -mllvm -poseidon-print %s -o %t/plain.ll 2>&1 | FileCheck --check-prefix=PLAIN %s
// RUN: FileCheck --check-prefix=LEFT %s < %t/plain.ll
// RUN: %clang -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -S -emit-llvm -I%FPProfileInc %clangLoadPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c -mllvm -poseidon-unit=ir -mllvm -poseidon-print %s -o %t/gen.ll 2>&1 | FileCheck --check-prefix=GEN %s
// RUN: FileCheck --check-prefix=GENIR %s < %t/gen.ll
// RUN: %clang -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path --cuda-device-only --no-gpu-bundle-output -cuid=0 -O2 -c -I%FPProfileInc %clangLoadPoseidonEnzyme -Xclang -mllvm -Xclang -poseidon-profile-generate -Xclang -mllvm -Xclang -poseidon-cache=%t/c -Xclang -mllvm -Xclang -poseidon-unit=object %s -o %t/gen.co
// RUN: llvm-readelf --dyn-syms %t/gen.co | FileCheck --check-prefix=CO %s
// REQUIRES: poseidon, enzyme, amdgpu-registered-target, rocm

// A site whose marked body is a forward-mode derivative request, in AMDGPU
// device code: the early run defers it, the late run profiles the tangent code
// Enzyme generated, and a site left alone calls the module's own copy of the
// body as the pipeline left it. Its profiled code object links: the derivative
// mallocs its caches, the probes are the module's, and the table is exported.
// (-mllvm would also reach the device linker, which knows no Poseidon flag.)

#include <hip/hip_runtime.h>

#include "poseidon/poseidon.h"

template <typename R, typename... T> __device__ R __enzyme_fwddiff(void *, T...);
template <typename R, typename... T>
__device__ R __poseidon_fp_optimize(void *, T...);
__device__ int enzyme_dup, poseidon_tau;

extern "C" __device__ __attribute__((noinline)) void qf(const double *x,
                                                        double *y, int n) {
  double s = 0.0;
  for (int i = 0; i < n; ++i)
    s += x[i] * x[i];
  y[0] = s + 1.0;
}

extern "C" __device__ void jvp_body(const double *x, const double *dx,
                                    double *y, double *dy, int n) {
  __enzyme_fwddiff<void>((void *)qf, enzyme_dup, x, dx, enzyme_dup, y, dy, n);
}

__global__ void jvp(const double *x, const double *dx, double *y, double *dy,
                    int n) {
  double xl[4] = {x[0], x[1], x[2], x[3]}, dxl[4] = {dx[0], dx[1], dx[2], dx[3]};
  double yl, dyl;
  double x_sh[4] = {}, dx_sh[4] = {}, y_sh = 1.0, dy_sh = 1.0;
  __poseidon_fp_optimize<void>((void *)jvp_body, poseidon_tau, 1e-6,
                               enzyme_dup, xl, x_sh, enzyme_dup, dxl, dx_sh,
                               enzyme_dup, &yl, &y_sh, enzyme_dup, &dyl, &dy_sh,
                               n);
  y[0] = yl;
  dy[0] = dyl;
}

// PLAIN: [poseidon] jvp_body: marked body still holds an __enzyme_* request; deferred to the late run
// LEFT: define {{.*}}amdgpu_kernel void @_Z3jvpPKdS0_PdS1_i(
// LEFT: call {{.*}}@fwddiffeqf.left(
// LEFT: define internal {{.*}}@fwddiffeqf.left(

// GEN: [poseidon] jvp_body: marked body still holds an __enzyme_* request; deferred to the late run
// GEN: [poseidon] jvp_body: folded 1 Enzyme-generated call(s) into the deferred body
// GEN: [poseidon] block-geometry probe placed in 1 profiled function(s)
// GENIR-DAG: define internal void @poseidonLogValueCUDA(
// GENIR-DAG: define internal void @poseidonLogGradCUDA(

// CO-NOT: UND {{[_a-zA-Z]}}
// CO: __poseidon_profile_table
// CO-NOT: UND {{[_a-zA-Z]}}
