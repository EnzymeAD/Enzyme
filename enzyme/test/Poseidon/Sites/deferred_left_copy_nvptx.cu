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

// The kernel passes its locals to the copy and stores what the copy wrote.
// CHECK-LABEL: .entry _Z3jvpPKdS0_PdS1_(
// CHECK-DAG: ld.param{{.*}} [[XG:%rd[0-9]+]], [_Z3jvpPKdS0_PdS1__param_0];
// CHECK-DAG: ld.param{{.*}} [[DXG:%rd[0-9]+]], [_Z3jvpPKdS0_PdS1__param_1];
// CHECK-DAG: ld.param{{.*}} [[Y:%rd[0-9]+]], [_Z3jvpPKdS0_PdS1__param_2];
// CHECK-DAG: ld.param{{.*}} [[DYG:%rd[0-9]+]], [_Z3jvpPKdS0_PdS1__param_3];
// CHECK-DAG: cvta.to.global.u64 [[X:%rd[0-9]+]], [[XG]];
// CHECK-DAG: cvta.to.global.u64 [[DX:%rd[0-9]+]], [[DXG]];
// CHECK-DAG: cvta.to.global.u64 [[DY:%rd[0-9]+]], [[DYG]];
// CHECK-DAG: add.u64 [[XO:%rd[0-9]+]], %SPL, 0;
// CHECK-DAG: cvta.local.u64 [[XA:%rd[0-9]+]], [[XO]];
// CHECK-DAG: add.u64 [[DXO:%rd[0-9]+]], %SPL, 8;
// CHECK-DAG: cvta.local.u64 [[DXA:%rd[0-9]+]], [[DXO]];
// CHECK-DAG: add.u64 [[DYO:%rd[0-9]+]], %SPL, 16;
// CHECK-DAG: cvta.local.u64 [[DYA:%rd[0-9]+]], [[DYO]];
// CHECK-DAG: ld.global{{.*}} [[XV:%[a-z]+[0-9]+]], {{\[}}[[X]]{{\]}};
// CHECK-DAG: st.local{{.*}} [%SPL], [[XV]];
// CHECK-DAG: ld.global{{.*}} [[DXV:%[a-z]+[0-9]+]], {{\[}}[[DX]]{{\]}};
// CHECK-DAG: st.local{{.*}} [%SPL+8], [[DXV]];
// CHECK-DAG: st.param{{.*}} [param0], [[XA]];
// CHECK-DAG: st.param{{.*}} [param1], [[DXA]];
// CHECK-DAG: st.param{{.*}} [param2], [[Y]];
// CHECK-DAG: st.param{{.*}} [param3], [[DYA]];
// CHECK: call.uni fwddiffeqf_$_left, (param0, param1, param2, param3);
// CHECK: ld.local{{.*}} [[DYV:%[a-z]+[0-9]+]], [%SPL+16];
// CHECK-NEXT: st.global{{.*}} {{\[}}[[DY]]{{\]}}, [[DYV]];
// CHECK-NEXT: ret;

// The copy is defined in this module and computes y = x * x + 1 and
// dy = 2 * dx * x.
// CHECK-LABEL: .func fwddiffeqf_$_left(
// CHECK-DAG: ld.param{{.*}} [[QX:%rd[0-9]+]], [fwddiffeqf_$_left_param_0];
// CHECK-DAG: ld.param{{.*}} [[QDX:%rd[0-9]+]], [fwddiffeqf_$_left_param_1];
// CHECK-DAG: ld.param{{.*}} [[QY:%rd[0-9]+]], [fwddiffeqf_$_left_param_2];
// CHECK-DAG: ld.param{{.*}} [[QDY:%rd[0-9]+]], [fwddiffeqf_$_left_param_3];
// CHECK-DAG: ld{{(\.[a-z0-9]+)*}} [[QXV:%[a-z]+[0-9]+]], {{\[}}[[QX]]{{\]}};
// CHECK-DAG: ld{{(\.[a-z0-9]+)*}} [[QDXV:%[a-z]+[0-9]+]], {{\[}}[[QDX]]{{\]}};
// CHECK-DAG: add.f64 [[T:%[a-z]+[0-9]+]], [[QXV]], [[QXV]];
// CHECK-DAG: mul.f64 [[QDYV:%[a-z]+[0-9]+]], [[QDXV]], [[T]];
// CHECK-DAG: fma.rn.f64 [[QYV:%[a-z]+[0-9]+]], [[QXV]], [[QXV]], 0d3FF0000000000000;
// CHECK-DAG: st{{(\.[a-z0-9]+)*}} {{\[}}[[QDY]]{{\]}}, [[QDYV]];
// CHECK-DAG: st{{(\.[a-z0-9]+)*}} {{\[}}[[QY]]{{\]}}, [[QYV]];
// CHECK: ret;
