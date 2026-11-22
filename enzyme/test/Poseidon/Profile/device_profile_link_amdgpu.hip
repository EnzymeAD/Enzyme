// REQUIRES: enzyme, amdgpu-registered-target, rocm
// DEFINE: %{clang} = %clang -x hip --rocm-path=%rocm_path -cuid=0 -I%poseidon_profile_inc \
// DEFINE:   %loadClangPoseidonEnzyme %s
// RUN: rm -rf %t && mkdir %t
// RUN: %{clang} --offload-arch=%amdgpu_arch --cuda-device-only -O2 -S -emit-llvm -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c -mllvm -poseidon-unit=scale -o %t/dev.ll
// RUN: FileCheck --check-prefix=DEV %s < %t/dev.ll
// RUN: mkdir -p %t/tmp && not env TMPDIR=%t/tmp %clang -x hip --offload-arch=gfx1100 --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c1100 -mllvm -poseidon-unit=scale %s -o %t/other.ll 2>&1 | FileCheck --check-prefix=ARCH --implicit-check-not=PLEASE %s
// RUN: mkdir -p %t/tmpg && not env TMPDIR=%t/tmpg %clang -x hip --offload-arch=gfx1100 --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -g -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c1100g -mllvm -poseidon-unit=scale %s -o %t/other-g.ll 2>&1 | FileCheck --check-prefix=ARCH --implicit-check-not=PLEASE %s
// RUN: ls %t/tmpg | count 0
// RUN: not ls %t/c1100/launches_* %t/c1100g/launches_*
// RUN: ls %t/tmp | count 0
// RUN: %{clang} --offload-arch=%amdgpu_arch --cuda-device-only -O2 -S -emit-llvm -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c -mllvm -poseidon-unit=nosite -DNO_SITE -o %t/nosite.ll
// RUN: FileCheck --check-prefix=NOSITE %s < %t/nosite.ll
// RUN: %{clang} --offload-arch=gfx1100 --cuda-device-only -O2 -S -emit-llvm -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c1100 -mllvm -poseidon-unit=nosite -DNO_SITE -o %t/nosite1100.ll
// RUN: FileCheck --check-prefix=NOSITE %s < %t/nosite1100.ll
// RUN: %{clang} --offload-arch=gfx942 --cuda-host-only -O0 -S -emit-llvm -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c -mllvm -poseidon-unit=scale -o %t/host.ll
// RUN: FileCheck --check-prefix=HOST %s < %t/host.ll
// RUN: %{clang} --offload-arch=gfx942 --cuda-host-only -fgpu-rdc -O0 -S -emit-llvm -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c -mllvm -poseidon-unit=scale -o %t/rdc.ll
// RUN: FileCheck --check-prefix=RDC %s < %t/rdc.ll
// RUN: %{clang} --offload-arch=gfx942 --cuda-device-only -O2 -S -emit-llvm -o %t/plain-dev.ll
// RUN: %{clang} --offload-arch=gfx942 --cuda-host-only -O0 -S -emit-llvm -o %t/plain-host.ll
// RUN: cat %t/plain-dev.ll %t/plain-host.ll | FileCheck --check-prefix=PLAIN %s
// RUN: %{clang} --offload-arch=%amdgpu_arch --cuda-device-only --no-gpu-bundle-output -O2 -c -Xclang -mllvm -Xclang -poseidon-profile-generate -Xclang -mllvm -Xclang -poseidon-cache=%t/c -Xclang -mllvm -Xclang -poseidon-unit=loop -DLOOP -o %t/loop.co
// RUN: llvm-readelf --dyn-syms %t/loop.co | FileCheck --check-prefix=CO %s

// The HIP form of device_profile_link.cu. The AMDGPU device module of a profile
// generation gets the device half of the profiler, local to the module, and a
// table every module defines; the host half registers the table with the HIP
// runtime (__hipRegisterVar) and the profiler runtime or, with relocatable
// device code, leaves a HIP offloading entry. The profiled code object links
// with no undefined symbol, malloc included (the derivative of a loop with a
// run-time trip count allocates its cache). The code object compile passes the
// flags with -Xclang -mllvm because the device linker knows no Poseidon flag.
// The profiler's bitcode is built for POSEIDON_AMDGPU_ARCH, so a module for
// another processor (gfx1100) whose probes call into it is refused with an
// error at the user's kernel and leaves no launch manifest and no temporary
// file; one without probes takes the table alone.

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

// ARCH: device_profile_link_amdgpu.hip:[[@LINE-19]]:{{[0-9]+}}: error: Poseidon: the AMDGPU device profiler was built for '{{gfx[0-9a-f]+}}' (POSEIDON_AMDGPU_ARCH) but scale(double*, double) compiles for target-cpu 'gfx1100'; build Poseidon for this processor, or compile the profiling build for that one

// NOSITE: @__poseidon_profile_table = weak_odr protected addrspace(4)
// NOSITE-NOT: poseidonLogValueCUDA

// HOST: @__poseidon_profile_table = internal global [[[N:[0-9]+]] x i8] zeroinitializer, align 8
// HOST: @llvm.global_ctors = appending global [3 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @__hip_module_ctor, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @__poseidon_profile_register_module_table, ptr null }, { i32, ptr, ptr } { i32 0, ptr @poseidon_register_static_preprocess__Z5scalePdd_poseidon_body, ptr null }]
// HOST: define internal {{(fastcc )?}}void @__hip_register_globals(ptr %[[H:[0-9]+]])
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
