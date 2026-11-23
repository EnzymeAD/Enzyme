// REQUIRES: enzyme, amdgpu-registered-target, offload-new-driver
// RUN: rm -rf %t && mkdir -p %t/drv %t/opt %t/rdc
// RUN: (env TMPDIR=%t %clang -x hip --offload-arch=gfx942 -nogpulib -nogpuinc -O2 -c -I%poseidon_profile_inc %loadClangPoseidonEnzyme %s -o %t/plain.o 2>&1; echo "exit status $?") | FileCheck --check-prefix=DEFERRED --implicit-check-not=error: --implicit-check-not=REPRODUCER --implicit-check-not="exit code" --implicit-check-not=ld.lld %s
// RUN: %clang -x hip --offload-arch=gfx942 -nogpulib -nogpuinc -O2 -c -I%poseidon_profile_inc %loadClangPoseidonEnzyme -Xclang -mllvm -Xclang -enzyme-lto-prelink=1 %s -o %t/opt/k.o
// RUN: cd %t/opt && llvm-objdump --offloading k.o > /dev/null && llvm-readelf --dyn-syms k.o.*gfx942 | FileCheck --check-prefix=CO --implicit-check-not=__enzyme %s
// RUN: (env TMPDIR=%t %poseidon_clangxx -x hip --offload-arch=gfx942 -nogpulib -nogpuinc -O2 -c %s -o %t/drv/none.o 2>&1; echo "exit status $?") | FileCheck --check-prefix=NOENZYME --implicit-check-not=error: --implicit-check-not=REPRODUCER --implicit-check-not="exit code" --implicit-check-not=ld.lld %s
// RUN: %poseidon_clangxx -x hip --offload-arch=gfx942 -nogpulib -nogpuinc -O2 -c %loadClangEnzyme -Xclang -mllvm -Xclang -enzyme-lto-prelink=1 %s -o %t/drv/k.o
// RUN: cd %t/drv && llvm-objdump --offloading k.o > /dev/null && llvm-readelf --dyn-syms k.o.*gfx942 | FileCheck --check-prefix=CO --implicit-check-not=__enzyme %s
// RUN: %poseidon_clangxx -x hip --offload-arch=gfx942 -nogpulib -nogpuinc -fgpu-rdc -O2 -c %loadClangEnzyme -Xclang -mllvm -Xclang -enzyme-lto-prelink=1 %s -o %t/rdc/k.o
// RUN: llvm-offload-binary %t/rdc/k.o --image=file=%t/rdc/dev.bc,arch=gfx942
// RUN: llvm-dis %t/rdc/dev.bc -o - | FileCheck --check-prefix=DEV --implicit-check-not=__enzyme %s

// A HIP compile with -c and no ROCm (no device library; the runtime
// declarations a kernel stub needs are written out). Its device half is an LTO
// pre-link compile, where Enzyme defers derivative requests to a post-link run
// unless -enzyme-lto-prelink is given, and the device link of the same -c loads
// no plugin. A site whose body is the user's own derivative request is handled
// after Enzyme lowers that request: with both plugins and without the option,
// the compile stops in the device compiler with the reason (no crash, no
// undefined __enzyme_* symbol at the device link); through poseidon-clang++,
// which loads Enzyme only to generate a profile, it stops asking for Enzyme's
// plugin. With the option, given to clang or to poseidon-clang++ with Enzyme's
// plugin, the code object and the relocatable device code hold no request. The
// device code is that of deferred_left_copy_nvptx.cu.

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

// DEFERRED: deferred_site_lto_amdgpu.hip:[[@LINE-16]]:{{[0-9]+}}: error: Poseidon: the marked body jvp_body still holds an __enzyme_* request that nothing lowered. This is an LTO pre-link compile, where Enzyme leaves derivative requests to the post-link run unless -enzyme-lto-prelink is given; pass -mllvm -enzyme-lto-prelink=1 (for HIP as -Xclang -mllvm -Xclang -enzyme-lto-prelink=1).{{$}}
// DEFERRED: exit status 1

// NOENZYME: deferred_site_lto_amdgpu.hip:[[@LINE-19]]:{{[0-9]+}}: error: Poseidon: the marked body jvp_body still holds an __enzyme_* request that nothing lowered. Add -fpass-plugin=<ClangEnzyme>.so after the Poseidon plugin.{{$}}
// NOENZYME: exit status 1

// CO-NOT: UND {{[_a-zA-Z]}}
// CO: FUNC GLOBAL PROTECTED {{[0-9]+}} _Z3jvpPKdS0_PdS1_{{$}}
// CO-NOT: UND {{[_a-zA-Z]}}

// The kernel passes its locals to the module's copy of the lowered tangent,
// which computes y = x * x + 1 and dy = 2 * dx * x.
// DEV: define protected amdgpu_kernel void @_Z3jvpPKdS0_PdS1_(ptr addrspace(1) {{.*}}[[X:%[0-9]+]], ptr addrspace(1) {{.*}}[[DX:%[0-9]+]], ptr addrspace(1) {{.*}}[[Y:%[0-9]+]], ptr addrspace(1) {{.*}}[[DY:%[0-9]+]])
// DEV-DAG: call {{.*}}void @fwddiffeqf.left(ptr {{.*}}[[XP:%[0-9]+]], ptr {{.*}}[[DXP:%[0-9]+]], ptr {{.*}}[[YP:%[0-9]+]], ptr {{.*}}[[DYP:%[0-9]+]])
// DEV-DAG: [[XP]] = addrspacecast ptr addrspace(5) [[XA:%[0-9]+]] to ptr
// DEV-DAG: [[DXP]] = addrspacecast ptr addrspace(5) [[DXA:%[0-9]+]] to ptr
// DEV-DAG: [[YP]] = addrspacecast ptr addrspace(5) [[YA:%[0-9]+]] to ptr
// DEV-DAG: [[DYP]] = addrspacecast ptr addrspace(5) [[DYA:%[0-9]+]] to ptr
// DEV-DAG: [[XV:%[0-9]+]] = load double, ptr addrspace(1) [[X]]
// DEV-DAG: store double [[XV]], ptr addrspace(5) [[XA]]
// DEV-DAG: [[DXV:%[0-9]+]] = load double, ptr addrspace(1) [[DX]]
// DEV-DAG: store double [[DXV]], ptr addrspace(5) [[DXA]]
// DEV-DAG: [[YV:%[0-9]+]] = load double, ptr addrspace(5) [[YA]]
// DEV-DAG: store double [[YV]], ptr addrspace(1) [[Y]]
// DEV-DAG: [[DYV:%[0-9]+]] = load double, ptr addrspace(5) [[DYA]]
// DEV-DAG: store double [[DYV]], ptr addrspace(1) [[DY]]
// DEV: {{^}}}
// DEV: define internal {{.*}}void @fwddiffeqf.left(ptr {{.*}}[[QX:%[0-9]+]], ptr {{.*}}[[QDX:%[0-9]+]], ptr {{.*}}[[QY:%[0-9]+]], ptr {{.*}}[[QDY:%[0-9]+]])
// DEV-DAG: [[QDXV:%[0-9]+]] = load double, ptr [[QDX]]
// DEV-DAG: [[QXV:%[0-9]+]] = load double, ptr [[QX]]
// DEV-DAG: [[QYV:%[0-9]+]] = {{.*}}call {{.*}}double @llvm.fmuladd.f64(double [[QXV]], double [[QXV]], double 1.000000e+00)
// DEV-DAG: [[T:%[0-9]+]] = fmul {{.*}}double [[QDXV]], 2.000000e+00
// DEV-DAG: [[QDYV:%[0-9]+]] = fmul {{.*}}double [[T]], [[QXV]]
// DEV-DAG: store double [[QDYV]], ptr [[QDY]]
// DEV-DAG: store double [[QYV]], ptr [[QY]]
// DEV: {{^}}}
