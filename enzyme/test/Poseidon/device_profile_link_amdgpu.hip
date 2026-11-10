// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -S -emit-llvm -I%FPProfileInc %clangLoadPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c %s -o %t/dev.ll
// RUN: FileCheck --check-prefix=DEV %s < %t/dev.ll
// RUN: (%clang -x hip --offload-arch=gfx1100 --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -S -fno-crash-diagnostics -emit-llvm -I%FPProfileInc %clangLoadPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c1100 %s -o %t/other.ll 2>&1 || true) | FileCheck --check-prefix=ARCH %s
// RUN: %clang -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -S -emit-llvm -I%FPProfileInc %clangLoadPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c -DNO_SITE %s -o %t/nosite.ll
// RUN: FileCheck --check-prefix=NOSITE %s < %t/nosite.ll
// RUN: %clang -x hip --offload-arch=gfx1100 --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -S -emit-llvm -I%FPProfileInc %clangLoadPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c1100 -DNO_SITE %s -o %t/nosite1100.ll
// RUN: FileCheck --check-prefix=NOSITE %s < %t/nosite1100.ll
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-host-only -cuid=0 -O0 -S -emit-llvm -I%FPProfileInc %clangLoadPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c %s -o %t/host.ll
// RUN: FileCheck --check-prefix=HOST %s < %t/host.ll
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-host-only -fgpu-rdc -cuid=0 -O0 -S -emit-llvm -I%FPProfileInc %clangLoadPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c %s -o %t/rdc.ll
// RUN: FileCheck --check-prefix=RDC %s < %t/rdc.ll
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -S -emit-llvm -I%FPProfileInc %clangLoadPoseidonEnzyme %s -o %t/plain-dev.ll
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-host-only -cuid=0 -O0 -S -emit-llvm -I%FPProfileInc %clangLoadPoseidonEnzyme %s -o %t/plain-host.ll
// RUN: cat %t/plain-dev.ll %t/plain-host.ll | FileCheck --check-prefix=PLAIN %s
// RUN: %clang -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path --cuda-device-only --no-gpu-bundle-output -cuid=0 -O2 -c -I%FPProfileInc %clangLoadPoseidonEnzyme -Xclang -mllvm -Xclang -poseidon-profile-generate -Xclang -mllvm -Xclang -poseidon-cache=%t/c -DLOOP %s -o %t/loop.co
// RUN: llvm-readelf --dyn-syms %t/loop.co | FileCheck --check-prefix=CO %s
// REQUIRES: poseidon, enzyme, amdgpu-registered-target, rocm

// device_profile_link.cu for HIP: the AMDGPU device module of a profile
// generation gets the device half of the profiler, linked from the AMDGPU
// bitcode the plugin carries and local to the module, and a table every module
// defines; the host half of the same unit registers that table with the HIP
// runtime (__hipRegisterVar from __hip_register_globals) and with the profiler
// runtime, or with relocatable device code leaves a HIP offloading entry. The
// profiled code object links: the derivative of a loop whose trip count is
// known only as it runs grows its cache with malloc and frees it, which the
// profiler's bitcode defines. (-mllvm would also reach the device linker,
// which knows no Poseidon flag.) The bitcode carries the ROCm device libraries
// of the processor Poseidon was built for (POSEIDON_AMDGPU_ARCH), so a module
// for another one (here gfx1100) whose probes call into it is refused; one
// without probes takes the table alone.

#include <hip/hip_runtime.h>

#include "poseidon/poseidon.h"

#ifndef NO_SITE
POSEIDON_OPTIMIZE
#endif
__global__ void scale(double *x, double a) { x[0] = a * x[0] + 1.0; }

#ifdef LOOP
POSEIDON_OPTIMIZE __global__ void sum(const double *x, double *y) {
  double s = 0.0;
  for (int i = 0; s < 10.0; ++i)
    s += x[i] * x[i];
  y[0] = s;
}
#endif

// DEV-DAG: @llvm.compiler.used = {{.*}}@__poseidon_profile_table
// DEV-DAG: @__poseidon_profile_table = weak_odr protected addrspace(4) externally_initialized constant {{.*}} zeroinitializer
// DEV-DAG: define internal void @poseidonLogValueCUDA(
// DEV-DAG: define internal void @poseidonLogGradCUDA(
// DEV-DAG: define internal void @poseidonProfileBlockDimsCUDA(
// DEV-DAG: define internal void @poseidonProfileStaticCUDA(
// DEV-DAG: define amdgpu_kernel void @_Z5scalePdd(

// ARCH: Poseidon: the AMDGPU device profiler was built for '{{gfx[0-9a-f]+}}' (POSEIDON_AMDGPU_ARCH) but {{.*}} compiles for target-cpu 'gfx1100'

// NOSITE: @__poseidon_profile_table = weak_odr protected addrspace(4)
// NOSITE-NOT: poseidonLogValueCUDA

// HOST: @__poseidon_profile_table = internal global [[[N:[0-9]+]] x i8] zeroinitializer, align 8
// HOST: @llvm.global_ctors = appending global [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @__hip_module_ctor, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @__poseidon_profile_register_module_table, ptr null }]
// HOST: define internal void @__hip_register_globals(ptr %[[H:[0-9]+]])
// HOST: call void @__hipRegisterVar(ptr %[[H]], ptr @__poseidon_profile_table, ptr @[[NAME:[^,]+]], ptr @[[NAME]], i32 0, i64 [[N]], i32 1, i32 0)
// HOST-NEXT: ret void
// HOST: define internal void @__poseidon_profile_register_module_table()
// HOST-NEXT: call void @__poseidon_profile_register_module(ptr @__poseidon_profile_table)

// HIP's offload kind is 4 (CUDA's 2).
// RDC: @__poseidon_profile_table = internal global [[[N:[0-9]+]] x i8] zeroinitializer, align 8
// RDC: @.offloading.entry.__poseidon_profile_table = weak {{.*}}constant %struct.__tgt_offload_entry { i64 0, i16 1, i16 4, i32 16, ptr @__poseidon_profile_table, ptr @{{[^,]+}}, i64 [[N]], i64 0, ptr null }, section "llvm_offload_entries"
// RDC: @llvm.global_ctors = {{.*}} ptr @__poseidon_profile_register_module_table
// RDC: define internal void @__poseidon_profile_register_module_table()
// RDC-NEXT: call void @__poseidon_profile_register_module(ptr @__poseidon_profile_table)

// PLAIN-NOT: __poseidon_profile

// CO-NOT: UND {{[_a-zA-Z]}}
// CO: __poseidon_profile_table
// CO-NOT: UND {{[_a-zA-Z]}}
