// REQUIRES: enzyme, amdgpu-registered-target, offload-new-driver
// RUN: rm -rf %t && mkdir -p %t/drv %t/opt %t/rdc
// RUN: (env TMPDIR=%t %clang -x hip --offload-arch=gfx942 -nogpulib -nogpuinc -O2 -c -I%poseidon_profile_inc %loadClangPoseidonEnzyme %s -o %t/plain.o 2>&1; echo "exit status $?") | FileCheck --check-prefix=DEFERRED --implicit-check-not=REPRODUCER --implicit-check-not="exit code" --implicit-check-not=ld.lld %s
// RUN: %clang -x hip --offload-arch=gfx942 -nogpulib -nogpuinc -O2 -c -I%poseidon_profile_inc %loadClangPoseidonEnzyme -Xclang -mllvm -Xclang -enzyme-lto-prelink=1 %s -o %t/opt/k.o
// RUN: cd %t/opt && llvm-objdump --offloading k.o > /dev/null && llvm-readelf --dyn-syms k.o.*gfx942 | FileCheck --check-prefix=CO --implicit-check-not=__enzyme %s
// RUN: (env TMPDIR=%t %poseidon_clangxx -x hip --offload-arch=gfx942 -nogpulib -nogpuinc -O2 -c %s -o %t/drv/none.o 2>&1; echo "exit status $?") | FileCheck --check-prefix=NOENZYME --implicit-check-not=REPRODUCER --implicit-check-not="exit code" --implicit-check-not=ld.lld %s
// RUN: %poseidon_clangxx -x hip --offload-arch=gfx942 -nogpulib -nogpuinc -O2 -c %loadClangEnzyme -Xclang -mllvm -Xclang -enzyme-lto-prelink=1 %s -o %t/drv/k.o
// RUN: cd %t/drv && llvm-objdump --offloading k.o > /dev/null && llvm-readelf --dyn-syms k.o.*gfx942 | FileCheck --check-prefix=CO --implicit-check-not=__enzyme %s
// RUN: %poseidon_clangxx -x hip --offload-arch=gfx942 -nogpulib -nogpuinc -fgpu-rdc -O2 -c %loadClangEnzyme -Xclang -mllvm -Xclang -enzyme-lto-prelink=1 %s -o %t/rdc/k.o
// RUN: llvm-offload-binary %t/rdc/k.o --image=file=%t/rdc/dev.bc,arch=gfx942
// RUN: llvm-dis %t/rdc/dev.bc -o - | FileCheck --check-prefix=DEV --implicit-check-not=__enzyme %s

// A HIP compile with -c, without ROCm (no device library, the few runtime
// declarations a kernel stub needs written out): the device half is an LTO
// pre-link compile, where Enzyme leaves derivative requests to a post-link run
// unless -enzyme-lto-prelink is given, and the device link the same -c runs
// loads no plugin. A site whose body is the user's own derivative request is
// handled after Enzyme has lowered that request: with both plugins and without
// the option the compile stops in the device compiler with the reason, not at
// the device link with an undefined __enzyme_* symbol, and without a crash;
// through poseidon-clang++, which loads Enzyme only to generate a profile, it
// stops asking for Enzyme's plugin.
// With the option, also through poseidon-clang++ given Enzyme's plugin and the
// option as clang is, the code object and the relocatable device code hold no
// request. The device code is that of deferred_left_copy_nvptx.cu.

#include <cstddef>

struct dim3 {
  unsigned x, y, z;
};
typedef struct ihipStream_t *hipStream_t;
extern "C" int hipLaunchKernel(const void *, dim3, dim3, void **, size_t,
                               hipStream_t);
extern "C" int __hipPushCallConfiguration(dim3, dim3, size_t, hipStream_t);
extern "C" int __hipPopCallConfiguration(dim3 *, dim3 *, size_t *,
                                         hipStream_t *);

#define __device__ __attribute__((device))
#define __global__ __attribute__((global))

#include "poseidon/poseidon.h"

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

// NOENZYME: Poseidon: the marked body jvp_body still holds an __enzyme_* request that nothing lowered. Add -fpass-plugin=<ClangEnzyme>.so after the Poseidon plugin.
// NOENZYME: exit status 1

// CO-NOT: UND {{[_a-zA-Z]}}
// CO: _Z3jvpPKdS0_PdS1_
// CO-NOT: UND {{[_a-zA-Z]}}

// DEV: define protected amdgpu_kernel void @_Z3jvpPKdS0_PdS1_(
// DEV: call {{.*}}@fwddiffeqf
