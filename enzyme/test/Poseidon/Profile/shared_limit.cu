// REQUIRES: enzyme
// REQUIRES: nvptx-device-profiler || (amdgpu-registered-target && rocm)
// DEFINE: %{nvptx} = %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate
// DEFINE: %{amdgpu} = %clang -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate
// RUN: rm -rf %t && mkdir %t
// RUN: %if nvptx-device-profiler %{ %{nvptx} -mllvm -poseidon-cache=%t/nvptx -mllvm -poseidon-print -DFIT=3072 %s -o %t/nvptx.ll 2>&1 | FileCheck --check-prefix=GEN -DFIT=3072 %s %}
// RUN: %if nvptx-device-profiler %{ FileCheck --check-prefixes=IR,IR-NVPTX -DFIT=3072 -DBYTES=24576 %s < %t/nvptx.ll %}
// RUN: %if nvptx-device-profiler %{ cat %t/nvptx/launches_*.json | FileCheck --check-prefix=DESC -DFIT=3072 %s %}
// RUN: %if nvptx-device-profiler %{ not %{nvptx} -mllvm -poseidon-cache=%t/nvptx-over -DFIT=6144 %s -o /dev/null 2>&1 | FileCheck --check-prefix=OVER -DLIMIT=49152 %s %}
// RUN: %if amdgpu-registered-target && rocm %{ %{amdgpu} -mllvm -poseidon-cache=%t/amdgpu -mllvm -poseidon-print -DFIT=4096 %s -o %t/amdgpu.ll 2>&1 | FileCheck --check-prefix=GEN -DFIT=4096 %s %}
// RUN: %if amdgpu-registered-target && rocm %{ FileCheck --check-prefixes=IR,IR-AMDGPU -DFIT=4096 -DBYTES=32768 %s < %t/amdgpu.ll %}
// RUN: %if amdgpu-registered-target && rocm %{ cat %t/amdgpu/launches_*.json | FileCheck --check-prefix=DESC -DFIT=4096 %s %}
// RUN: %if amdgpu-registered-target && rocm %{ not %{amdgpu} -mllvm -poseidon-cache=%t/amdgpu-over -DFIT=6144 %s -o /dev/null 2>&1 | FileCheck --check-prefix=OVER -DLIMIT=65536 %s %}

// The derivative a profiled kernel runs keeps a shadow next to each of its
// floating-point shared buffers. A kernel whose buffers and their shadows
// exceed the static shared memory a kernel can declare (48 KiB on NVPTX, the
// 64 KiB local data share of a gfx942 workgroup on AMDGPU) stops the compile
// with an error that names them; one that reaches the limit exactly (FIT
// doubles) compiles.

#include "poseidon/poseidon.h"

#ifdef __NVPTX__
#define THREAD_X __nvvm_read_ptx_sreg_tid_x()
#define BLOCK_X __nvvm_read_ptx_sreg_ntid_x()
#else
#define THREAD_X __builtin_amdgcn_workitem_id_x()
#define BLOCK_X __builtin_amdgcn_workgroup_size_x()
#endif

template <int N>
__attribute__((global)) POSEIDON_OPTIMIZE void stage(const double *x,
                                                     double *y) {
  __attribute__((shared)) double buf[N];
  for (int i = THREAD_X; i < N; i += BLOCK_X)
    buf[i] = x[i] * x[i];
  for (int i = THREAD_X; i < N; i += BLOCK_X)
    y[i] = buf[N - 1 - i] * 3.0;
}

template __attribute__((global)) void stage<FIT>(const double *, double *);

// GEN: {{^}}[poseidon] _Z5stageILi[[FIT]]EEvPKdPd: site 0, 2 shadow buffer(s), output(s) 1{{$}}

// OVER: shared_limit.cu:[[@LINE-13]]:48: error: Poseidon: void stage<6144>(double const*, double*) reaches 98304 bytes of static shared memory once profiled, above the [[LIMIT]] a kernel can declare: its own buffers void stage<6144>(double const*, double*)::buf (49152 bytes); the shadows of its buffers void stage<6144>(double const*, double*)::buf (49152 bytes); the shadows of other kernels' buffers none. Profile the kernel with smaller shared buffers, or the kernels whose shadows it reaches in another translation unit.

// IR: @[[SHADOW:_ZZ5stageILi[0-9]+EEvPKdPdE3buf_shadow]] = {{.*}}addrspace(3) global {{\[}}[[FIT]] x double] undef,
// IR: @_ZZ5stageILi[[FIT]]EEvPKdPdE3buf = {{.*}}addrspace(3) global {{\[}}[[FIT]] x double] undef, {{.*}}!enzyme_shadow ![[SHMD:[0-9]+]]{{$}}
// The NVPTX kernel calls the derivative; the AMDGPU inliner folds it into the
// kernel.
// IR-NVPTX: define {{.*}}ptx_kernel void @_Z5stageILi[[FIT]]EEvPKdPd(
// IR-NVPTX-NEXT: call {{.*}}void @diffepreprocess__Z5stageILi[[FIT]]EEvPKdPd_poseidon_body(
// IR-NVPTX: define internal {{.*}}void @diffepreprocess__Z5stageILi[[FIT]]EEvPKdPd_poseidon_body(
// IR-AMDGPU: define {{.*}}amdgpu_kernel void @_Z5stageILi[[FIT]]EEvPKdPd(
// IR: call void @llvm.memset.p3.i64(ptr addrspace(3) {{.*}}@[[SHADOW]], i8 0, i64 [[BYTES]], i1 false)
// IR: call void @poseidonLogValueCUDA(
// IR: ![[SHMD]] = !{ptr addrspace(3) @[[SHADOW]]}

// DESC: "profile generation": [
// DESC-NEXT: "_Z5stageILi[[FIT]]EEvPKdPd 0 2 2 0 1 0 8"
// DESC-NEXT: ],
