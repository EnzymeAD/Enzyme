// REQUIRES: enzyme, cuda-runtime
// From LLVM 22 an sm_120 compile needs the PTX version of a CUDA installation.
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -I%poseidon_profile_inc %loadClangPoseidonEnzyme %s -o - | FileCheck --implicit-check-not=jvp_body.left %s

// The copy of a deferred body that a site the solve leaves alone calls is the
// module's own: PTX can spell its name, and it does not collide with the body.

#include "poseidon/poseidon.h"

#define __device__ __attribute__((device))
#define __global__ __attribute__((global))

template <typename R, typename... T> __device__ R __enzyme_fwddiff(void *, T...);
template <typename R, typename... T>
__device__ R __poseidon_fp_optimize(void *, T...);
__device__ int enzyme_dup, poseidon_tau;

extern "C" __device__ __attribute__((noinline)) void qf(const double *x,
                                                        double *y) {
  y[0] = x[0] * x[0] + 1.0;
}

extern "C" __device__ void jvp_body(const double *x, const double *dx,
                                    double *y, double *dy) {
  __enzyme_fwddiff<void>((void *)qf, enzyme_dup, x, dx, enzyme_dup, y, dy);
}

__global__ void jvp(const double *x, const double *dx, double *y, double *dy) {
  double xl = x[0], dxl = dx[0], yl, dyl;
  double x_sh = 0.0, dx_sh = 0.0, y_sh = 1.0, dy_sh = 1.0;
  __poseidon_fp_optimize<void>((void *)jvp_body, poseidon_tau, 1e-6,
                               enzyme_dup, &xl, &x_sh, enzyme_dup, &dxl,
                               &dx_sh, enzyme_dup, &yl, &y_sh, enzyme_dup, &dyl,
                               &dy_sh);
  y[0] = yl;
  dy[0] = dyl;
}

// CHECK: .visible .func jvp_body(
// CHECK: .entry _Z3jvpPKdS0_PdS1_(
// CHECK: call.uni fwddiffeqf_$_left,
