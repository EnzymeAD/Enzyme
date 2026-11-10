// REQUIRES: enzyme
// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -Xclang -disable-llvm-passes -S -emit-llvm -I%poseidon_profile_inc %s -o %t/dev.ll
// RUN: (env TMPDIR=%t %clang --target=nvptx64-nvidia-cuda -march=%gpu_arch -O2 -flto -x ir %t/dev.ll %loadClangPoseidonEnzyme -S -emit-llvm -o /dev/null 2>&1; echo "exit status $?") | FileCheck --check-prefix=DEFERRED --implicit-check-not=REPRODUCER --implicit-check-not="exit code" %s
// RUN: (env TMPDIR=%t %clang --target=nvptx64-nvidia-cuda -march=%gpu_arch -O2 -flto -x ir %t/dev.ll %loadClangPoseidonEnzyme -mllvm -poseidon-profile-use=%S/Inputs/other_sites -mllvm -poseidon-cache=%t/use -S -emit-llvm -o /dev/null 2>&1; echo "exit status $?") | FileCheck --check-prefix=DEFERRED --implicit-check-not=REPRODUCER --implicit-check-not="exit code" %s
// RUN: (env TMPDIR=%t %clang --target=nvptx64-nvidia-cuda -march=%gpu_arch -O2 -flto=thin -x ir %t/dev.ll %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/gen -S -emit-llvm -o /dev/null 2>&1; echo "exit status $?") | FileCheck --check-prefix=DEFERRED --implicit-check-not=REPRODUCER --implicit-check-not="exit code" %s
// RUN: %clang --target=nvptx64-nvidia-cuda -march=%gpu_arch -O2 -flto -x ir %t/dev.ll %loadClangPoseidonEnzyme -mllvm -enzyme-lto-prelink=1 -S -emit-llvm -o - 2>/dev/null | FileCheck --check-prefix=LOWERED --implicit-check-not=__enzyme_fwddiff %s
// RUN: %clang --target=nvptx64-nvidia-cuda -march=%gpu_arch -O2 -flto -x ir %t/dev.ll %loadClangPoseidonEnzyme -mllvm -enzyme-lto-prelink=1 -mllvm -poseidon-profile-use=%S/Inputs/other_sites -mllvm -poseidon-cache=%t/use1 -S -emit-llvm -o - 2>/dev/null | FileCheck --check-prefix=LOWERED --implicit-check-not=__enzyme_fwddiff %s
// RUN: %clang --target=nvptx64-nvidia-cuda -march=%gpu_arch -O2 -flto=thin -x ir %t/dev.ll %loadClangPoseidonEnzyme -mllvm -enzyme-lto-prelink=1 -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/gen1 -S -emit-llvm -o - 2>/dev/null | FileCheck --check-prefix=LOWERED --implicit-check-not=__enzyme_fwddiff %s
// RUN: not %opt %t/dev.ll %loadPoseidon -passes="poseidon,poseidon-late" -S -o /dev/null 2>&1 | FileCheck --check-prefix=BYNAME %s

// A site whose body is the user's own derivative request is handled after
// Enzyme has lowered that request. In an LTO pre-link compile, the phase every
// HIP device compile is in, Enzyme leaves the request to the post-link run
// unless -enzyme-lto-prelink is given, and a HIP device link loads no plugin:
// without the option the compile stops there and says why, without a profile,
// with one and generating one, instead of leaving an undefined __enzyme_*
// symbol to the link; with it the request is lowered. Run by pipeline name the
// pass is not told its phase and names the deferral only as a possibility. The
// source is that of deferred_left_copy_nvptx.cu.

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

// DEFERRED: Poseidon: the marked body jvp_body still holds an __enzyme_* request that nothing lowered. This is an LTO pre-link compile, where Enzyme leaves derivative requests to the post-link run unless -enzyme-lto-prelink is given; pass -mllvm -enzyme-lto-prelink=1
// DEFERRED: exit status 1

// LOWERED: define {{.*}}@fwddiffeqf

// BYNAME: Poseidon: the marked body jvp_body still holds an __enzyme_* request that nothing lowered. Add -fpass-plugin=<ClangEnzyme>.so after the Poseidon plugin. In an LTO pre-link pipeline Enzyme also leaves derivative requests to the post-link run unless -enzyme-lto-prelink is given.
