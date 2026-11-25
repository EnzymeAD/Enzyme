// REQUIRES: enzyme, nvptx-device-profiler
// RUN: rm -rf %t && mkdir %t
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -Xclang -disable-llvm-passes -S -emit-llvm -I%poseidon_profile_inc -DPOSEIDON_DEVICE_MATH %s -o %t/dev.ll
// RUN: (env TMPDIR=%t %clang --target=nvptx64-nvidia-cuda -march=%gpu_arch -O2 -flto -x ir %t/dev.ll %loadClangPoseidonEnzyme -S -emit-llvm -o /dev/null 2>&1; echo "exit status $?") | FileCheck --check-prefix=DEFERRED --implicit-check-not=error: --implicit-check-not=REPRODUCER --implicit-check-not="exit code" %s
// RUN: (env TMPDIR=%t %clang --target=nvptx64-nvidia-cuda -march=%gpu_arch -O2 -flto -x ir %t/dev.ll %loadClangPoseidonEnzyme -mllvm -poseidon-profile-use=%S/../Inputs/unrelated_site_profile -mllvm -poseidon-cache=%t/use -S -emit-llvm -o /dev/null 2>&1; echo "exit status $?") | FileCheck --check-prefix=DEFERRED --implicit-check-not=error: --implicit-check-not=REPRODUCER --implicit-check-not="exit code" %s
// RUN: (env TMPDIR=%t %clang --target=nvptx64-nvidia-cuda -march=%gpu_arch -O2 -flto=thin -x ir %t/dev.ll %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/gen -S -emit-llvm -o /dev/null 2>&1; echo "exit status $?") | FileCheck --check-prefix=DEFERRED --implicit-check-not=error: --implicit-check-not=REPRODUCER --implicit-check-not="exit code" %s
// RUN: %clang --target=nvptx64-nvidia-cuda -march=%gpu_arch -O2 -flto -x ir %t/dev.ll %loadClangPoseidonEnzyme -mllvm -enzyme-lto-prelink=1 -S -emit-llvm -o - | FileCheck --check-prefix=LOWERED --implicit-check-not=__enzyme_fwddiff %s
// RUN: %clang --target=nvptx64-nvidia-cuda -march=%gpu_arch -O2 -flto -x ir %t/dev.ll %loadClangPoseidonEnzyme -mllvm -enzyme-lto-prelink=1 -mllvm -poseidon-profile-use=%S/../Inputs/unrelated_site_profile -mllvm -poseidon-cache=%t/use1 -S -emit-llvm -o - | FileCheck --check-prefix=LOWERED --implicit-check-not=__enzyme_fwddiff %s
// RUN: %clang --target=nvptx64-nvidia-cuda -march=%gpu_arch -O2 -flto=thin -x ir %t/dev.ll %loadClangPoseidonEnzyme -mllvm -enzyme-lto-prelink=1 -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/gen1 -S -emit-llvm -o - | FileCheck --check-prefix=PGEN --implicit-check-not=__enzyme_fwddiff %s
// RUN: not %opt %t/dev.ll %loadPoseidon -passes="poseidon,poseidon-deferred-sites" -S -o /dev/null 2>&1 | FileCheck --check-prefix=BYNAME --implicit-check-not=error: %s

// A site whose body is the user's own derivative request is handled after
// Enzyme lowers that request. In an LTO pre-link compile, the phase of every
// HIP device compile, Enzyme defers the request to the post-link run unless
// -enzyme-lto-prelink is given, and a HIP device link loads no plugin: without
// the option the compile stops there and says why (with no profile, with one,
// and generating one) instead of leaving an undefined __enzyme_* symbol to the
// link; with it the request is lowered, and the kernel calls the module's
// copy of the tangent or, generating a profile, the profiled tangent. Run by
// pipeline name, the pass is not told its phase and names the deferral only as
// a possibility. The source is that of deferred_left_copy_nvptx.cu.

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

// DEFERRED: {{^}}error: <unknown>:0:0: in function jvp_body void (ptr, ptr, ptr, ptr): Poseidon: the marked body jvp_body still holds an __enzyme_* request that nothing lowered, which Enzyme leaves to the post-link run of an LTO pre-link compile; pass -mllvm -enzyme-lto-prelink=1 (for HIP, -Xclang -mllvm -Xclang -enzyme-lto-prelink=1){{$}}
// DEFERRED: exit status 1

// The kernel passes its locals to the module's copy of the lowered tangent,
// which computes y = x * x + 1 and dy = 2 * dx * x.
// LOWERED-LABEL: define dso_local ptx_kernel void @_Z3jvpPKdS0_PdS1_(
// LOWERED-SAME: ptr {{.*}}[[X:%[0-9]+]], ptr {{.*}}[[DX:%[0-9]+]], ptr {{.*}}[[Y:%[0-9]+]], ptr {{.*}}[[DY:%[0-9]+]])
// LOWERED-DAG: [[XV:%[0-9]+]] = load double, ptr [[X]],
// LOWERED-DAG: store double [[XV]], ptr [[XA:%[0-9]+]],
// LOWERED-DAG: [[DXV:%[0-9]+]] = load double, ptr [[DX]],
// LOWERED-DAG: store double [[DXV]], ptr [[DXA:%[0-9]+]],
// LOWERED-DAG: call {{.*}}void @fwddiffeqf.left(ptr {{[a-z ]*}}[[XA]], ptr {{[a-z ]*}}[[DXA]], ptr {{[a-z ]*}}[[Y]], ptr {{[a-z ]*}}[[DYA:%[0-9]+]])
// LOWERED-DAG: [[DYV:%[0-9]+]] = load double, ptr [[DYA]],
// LOWERED-DAG: store double [[DYV]], ptr [[DY]],
// LOWERED: {{^}}}
// LOWERED-LABEL: define internal {{.*}}void @fwddiffeqf.left(
// LOWERED-SAME: ptr {{.*}}[[QX:%[0-9]+]], ptr {{.*}}[[QDX:%[0-9]+]], ptr {{.*}}[[QY:%[0-9]+]], ptr {{.*}}[[QDY:%[0-9]+]])
// LOWERED-DAG: [[QDXV:%[0-9]+]] = load double, ptr [[QDX]],
// LOWERED-DAG: [[QXV:%[0-9]+]] = load double, ptr [[QX]],
// LOWERED-DAG: [[SQ:%[0-9]+]] = fmul {{.*}}double [[QXV]], [[QXV]]{{$}}
// LOWERED-DAG: [[QYV:%[0-9]+]] = fadd {{.*}}double [[SQ]], 1.000000e+00{{$}}
// LOWERED-DAG: [[T:%[0-9]+]] = fmul {{.*}}double [[QDXV]], 2.000000e+00{{$}}
// LOWERED-DAG: [[QDYV:%[0-9]+]] = fmul {{.*}}double [[T]], [[QXV]]{{$}}
// LOWERED-DAG: store double [[QDYV]], ptr [[QDY]],
// LOWERED-DAG: store double [[QYV]], ptr [[QY]],
// LOWERED: {{^}}}

// Generating a profile, the kernel passes each local with its shadow (x's and
// dx's zeroed, y's and dy's seeded 1.0) to the profiled tangent, which computes
// y and dy from the same expressions, each through its probe.
// PGEN-LABEL: define dso_local ptx_kernel void @_Z3jvpPKdS0_PdS1_(
// PGEN-SAME: ptr {{.*}}[[X:%[0-9]+]], ptr {{.*}}[[DX:%[0-9]+]], ptr {{.*}}[[Y:%[0-9]+]], ptr {{.*}}[[DY:%[0-9]+]])
// PGEN-DAG: [[XV:%[0-9]+]] = load double, ptr [[X]],
// PGEN-DAG: store double [[XV]], ptr [[XA:%[0-9]+]],
// PGEN-DAG: [[DXV:%[0-9]+]] = load double, ptr [[DX]],
// PGEN-DAG: store double [[DXV]], ptr [[DXA:%[0-9]+]],
// PGEN-DAG: call void @diffepreprocess_jvp_body(ptr {{[a-z ]*}}[[XA]], ptr {{[a-z ]*}}[[XS:%[0-9]+]], ptr {{[a-z ]*}}[[DXA]], ptr {{[a-z ]*}}[[DXS:%[0-9]+]], ptr {{[a-z ]*}}[[YA:%[0-9]+]], ptr {{[a-z ]*}}[[YS:%[0-9]+]], ptr {{[a-z ]*}}[[DYA:%[0-9]+]], ptr {{[a-z ]*}}[[DYS:%[0-9]+]])
// PGEN-DAG: store double 0.000000e+00, ptr [[XS]],
// PGEN-DAG: store double 0.000000e+00, ptr [[DXS]],
// PGEN-DAG: store double 1.000000e+00, ptr [[YS]],
// PGEN-DAG: store double 1.000000e+00, ptr [[DYS]],
// PGEN-DAG: [[YV:%[0-9]+]] = load double, ptr [[YA]],
// PGEN-DAG: store double [[YV]], ptr [[Y]],
// PGEN-DAG: [[DYV:%[0-9]+]] = load double, ptr [[DYA]],
// PGEN-DAG: store double [[DYV]], ptr [[DY]],
// PGEN: {{^}}}
// PGEN-LABEL: define internal void @diffepreprocess_jvp_body(
// PGEN-SAME: ptr {{.*}}[[BX:%[0-9]+]], ptr {{.*}}%{{[0-9]+}}, ptr {{.*}}[[BDX:%[0-9]+]], ptr {{.*}}%{{[0-9]+}}, ptr {{.*}}[[BY:%[0-9]+]], ptr {{.*}}%{{[0-9]+}}, ptr {{.*}}[[BDY:%[0-9]+]], ptr {{.*}}%{{[0-9]+}})
// PGEN-DAG: [[BDXV:%[0-9]+]] = load double, ptr [[BDX]],
// PGEN-DAG: [[BXV:%[0-9]+]] = load double, ptr [[BX]],
// PGEN-DAG: [[BT:%[0-9]+]] = fmul {{.*}}double [[BDXV]], 2.000000e+00,
// PGEN-DAG: [[BTP:%[0-9]+]] = call {{.*}}double @__poseidon_probe_preprocess_jvp_body_0_aug(double [[BT]])
// PGEN-DAG: [[BD:%[0-9]+]] = fmul {{.*}}double [[BTP]], [[BXV]],
// PGEN-DAG: [[BDP:%[0-9]+]] = call {{.*}}double @__poseidon_probe_preprocess_jvp_body_1_aug(double [[BD]])
// PGEN-DAG: [[BYV:%[0-9]+]] = call {{.*}}double @llvm.fmuladd.f64(double [[BXV]], double [[BXV]], double 1.000000e+00)
// PGEN-DAG: [[BYP:%[0-9]+]] = call {{.*}}double @__poseidon_probe_preprocess_jvp_body_2_aug(double [[BYV]])
// PGEN-DAG: store double [[BDP]], ptr [[BDY]],
// PGEN-DAG: store double [[BYP]], ptr [[BY]],
// PGEN: {{^}}}

// BYNAME: {{^}}error: <unknown>:0:0: in function jvp_body void (ptr, ptr, ptr, ptr): Poseidon: the marked body jvp_body still holds an __enzyme_* request that nothing lowered; add -fpass-plugin=<ClangEnzyme>.so after the Poseidon plugin, and in an LTO pre-link compile -mllvm -enzyme-lto-prelink=1{{$}}
