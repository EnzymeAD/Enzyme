// REQUIRES: enzyme
// REQUIRES: nvptx-device-profiler || (amdgpu-registered-target && rocm)
// DEFINE: %{nvptx} = %clang -x cuda -nocudainc -nocudalib -S -emit-llvm \
// DEFINE:   -I%poseidon_profile_inc %loadClangPoseidonEnzyme %s
// DEFINE: %{amdgpu} = %clang -x hip --rocm-path=%rocm_path -cuid=0 -I%poseidon_profile_inc \
// DEFINE:   %loadClangPoseidonEnzyme %s
// DEFINE: %{gen} = -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c
// RUN: rm -rf %t && mkdir -p %t && echo fatbin > %t/k.fatbin

// RUN: %if nvptx-device-profiler %{ %{nvptx} --cuda-device-only --cuda-gpu-arch=%gpu_arch -O2 %{gen} -mllvm -poseidon-unit=scale -o %t/nvptx-dev.ll %}
// RUN: %if nvptx-device-profiler %{ FileCheck --check-prefixes=DEV,DEV-NVPTX %s < %t/nvptx-dev.ll %}
// RUN: %if nvptx-device-profiler %{ FileCheck --check-prefix=RT -DFN=poseidonLogValueCUDA --implicit-check-not='declare {{.*}}@poseidon' %s < %t/nvptx-dev.ll %}
// RUN: %if nvptx-device-profiler %{ FileCheck --check-prefix=RT -DFN=poseidonLogGradCUDA --implicit-check-not='declare {{.*}}@poseidon' %s < %t/nvptx-dev.ll %}
// RUN: %if nvptx-device-profiler %{ FileCheck --check-prefix=RT -DFN=poseidonProfileBlockDimsCUDA --implicit-check-not='declare {{.*}}@poseidon' %s < %t/nvptx-dev.ll %}
// RUN: %if nvptx-device-profiler %{ FileCheck --check-prefix=RT -DFN=poseidonProfileStaticCUDA --implicit-check-not='declare {{.*}}@poseidon' %s < %t/nvptx-dev.ll %}
// RUN: %if nvptx-device-profiler %{ %{nvptx} --cuda-device-only --cuda-gpu-arch=%gpu_arch -O2 %{gen} -mllvm -poseidon-unit=nosite -DNO_SITE -o %t/nvptx-nosite.ll %}
// RUN: %if nvptx-device-profiler %{ FileCheck --check-prefixes=NOSITE,NOSITE-NVPTX %s < %t/nvptx-nosite.ll %}
// RUN: %if nvptx-device-profiler %{ %{nvptx} --cuda-host-only -O0 -Xclang -fcuda-include-gpubinary -Xclang %t/k.fatbin %{gen} -mllvm -poseidon-unit=scale -o %t/nvptx-host.ll %}
// RUN: %if nvptx-device-profiler %{ FileCheck --check-prefix=HOST -DRT=cuda %s < %t/nvptx-host.ll %}
// RUN: %if nvptx-device-profiler %{ %{nvptx} --cuda-host-only -fgpu-rdc -O0 %{gen} -mllvm -poseidon-unit=scale -o %t/nvptx-rdc.ll %}
// RUN: %if nvptx-device-profiler %{ FileCheck --check-prefix=RDC -DKIND=2 --implicit-check-not=RegisterVar %s < %t/nvptx-rdc.ll %}
// RUN: %if nvptx-device-profiler %{ %{nvptx} --cuda-device-only --cuda-gpu-arch=%gpu_arch -O2 -o %t/nvptx-plain-dev.ll %}
// RUN: %if nvptx-device-profiler %{ %{nvptx} --cuda-host-only -O0 -Xclang -fcuda-include-gpubinary -Xclang %t/k.fatbin -o %t/nvptx-plain-host.ll %}
// RUN: %if nvptx-device-profiler %{ cat %t/nvptx-plain-dev.ll %t/nvptx-plain-host.ll | FileCheck --check-prefixes=PLAIN,PLAIN-NVPTX --implicit-check-not=__poseidon_profile --implicit-check-not=@poseidon %s %}

// RUN: %if amdgpu-registered-target && rocm %{ %{amdgpu} --offload-arch=%amdgpu_arch --cuda-device-only -O2 -S -emit-llvm %{gen} -mllvm -poseidon-unit=scale -o %t/amdgpu-dev.ll %}
// RUN: %if amdgpu-registered-target && rocm %{ FileCheck --check-prefixes=DEV,DEV-AMDGPU %s < %t/amdgpu-dev.ll %}
// RUN: %if amdgpu-registered-target && rocm %{ FileCheck --check-prefix=RT -DFN=poseidonLogValueCUDA --implicit-check-not='declare {{.*}}@poseidon' %s < %t/amdgpu-dev.ll %}
// RUN: %if amdgpu-registered-target && rocm %{ FileCheck --check-prefix=RT -DFN=poseidonLogGradCUDA --implicit-check-not='declare {{.*}}@poseidon' %s < %t/amdgpu-dev.ll %}
// RUN: %if amdgpu-registered-target && rocm %{ FileCheck --check-prefix=RT -DFN=poseidonProfileBlockDimsCUDA --implicit-check-not='declare {{.*}}@poseidon' %s < %t/amdgpu-dev.ll %}
// RUN: %if amdgpu-registered-target && rocm %{ FileCheck --check-prefix=RT -DFN=poseidonProfileStaticCUDA --implicit-check-not='declare {{.*}}@poseidon' %s < %t/amdgpu-dev.ll %}
// RUN: %if amdgpu-registered-target && rocm %{ mkdir -p %t/tmp && not env TMPDIR=%t/tmp %{amdgpu} --offload-arch=gfx1100 --cuda-device-only -O2 -S -emit-llvm %{gen}1100 -mllvm -poseidon-unit=scale -o %t/other.ll 2>&1 | FileCheck --check-prefix=ARCH --implicit-check-not=PLEASE -DARCH=%amdgpu_arch %s %}
// RUN: %if amdgpu-registered-target && rocm %{ mkdir -p %t/tmpg && not env TMPDIR=%t/tmpg %{amdgpu} --offload-arch=gfx1100 --cuda-device-only -O2 -g -S -emit-llvm %{gen}1100g -mllvm -poseidon-unit=scale -o %t/other-g.ll 2>&1 | FileCheck --check-prefix=ARCH --implicit-check-not=PLEASE -DARCH=%amdgpu_arch %s %}
// RUN: %if amdgpu-registered-target && rocm %{ ls %t/tmpg | count 0 %}
// RUN: %if amdgpu-registered-target && rocm %{ not ls %t/c1100/launches_* %t/c1100g/launches_* %}
// RUN: %if amdgpu-registered-target && rocm %{ ls %t/tmp | count 0 %}
// RUN: %if amdgpu-registered-target && rocm %{ %{amdgpu} --offload-arch=%amdgpu_arch --cuda-device-only -O2 -S -emit-llvm %{gen} -mllvm -poseidon-unit=nosite -DNO_SITE -o %t/amdgpu-nosite.ll %}
// RUN: %if amdgpu-registered-target && rocm %{ FileCheck --check-prefixes=NOSITE,NOSITE-AMDGPU %s < %t/amdgpu-nosite.ll %}
// RUN: %if amdgpu-registered-target && rocm %{ %{amdgpu} --offload-arch=gfx1100 --cuda-device-only -O2 -S -emit-llvm %{gen}1100 -mllvm -poseidon-unit=nosite -DNO_SITE -o %t/amdgpu-nosite1100.ll %}
// RUN: %if amdgpu-registered-target && rocm %{ FileCheck --check-prefixes=NOSITE,NOSITE-AMDGPU %s < %t/amdgpu-nosite1100.ll %}
// RUN: %if amdgpu-registered-target && rocm %{ %{amdgpu} --offload-arch=gfx942 --cuda-host-only -O0 -S -emit-llvm %{gen} -mllvm -poseidon-unit=scale -o %t/amdgpu-host.ll %}
// RUN: %if amdgpu-registered-target && rocm %{ FileCheck --check-prefix=HOST -DRT=hip %s < %t/amdgpu-host.ll %}
// RUN: %if amdgpu-registered-target && rocm %{ %{amdgpu} --offload-arch=gfx942 --cuda-host-only -fgpu-rdc -O0 -S -emit-llvm %{gen} -mllvm -poseidon-unit=scale -o %t/amdgpu-rdc.ll %}
// RUN: %if amdgpu-registered-target && rocm %{ FileCheck --check-prefix=RDC -DKIND=4 --implicit-check-not=RegisterVar %s < %t/amdgpu-rdc.ll %}
// RUN: %if amdgpu-registered-target && rocm %{ %{amdgpu} --offload-arch=gfx942 --cuda-device-only -O2 -S -emit-llvm -o %t/amdgpu-plain-dev.ll %}
// RUN: %if amdgpu-registered-target && rocm %{ %{amdgpu} --offload-arch=gfx942 --cuda-host-only -O0 -S -emit-llvm -o %t/amdgpu-plain-host.ll %}
// RUN: %if amdgpu-registered-target && rocm %{ cat %t/amdgpu-plain-dev.ll %t/amdgpu-plain-host.ll | FileCheck --check-prefixes=PLAIN,PLAIN-AMDGPU --implicit-check-not=__poseidon_profile --implicit-check-not=@poseidon %s %}
// RUN: %if amdgpu-registered-target && rocm %{ %{amdgpu} --offload-arch=%amdgpu_arch --cuda-device-only --no-gpu-bundle-output -O2 -c -Xclang -mllvm -Xclang -poseidon-profile-generate -Xclang -mllvm -Xclang -poseidon-cache=%t/c -Xclang -mllvm -Xclang -poseidon-unit=loop -DLOOP -o %t/loop.co %}
// RUN: %if amdgpu-registered-target && rocm %{ llvm-readelf --dyn-syms %t/loop.co | FileCheck --check-prefix=CO %s %}

// Profile generation without relocatable device code: the device module gets
// the device half of the profiler, linked from the bitcode the plugin carries,
// local to the module and with no target-cpu or target-features of the
// processor that bitcode was built for, and a table every module defines
// whether or not it has a probe; the host half of the same unit registers that
// table with the CUDA or HIP runtime and with the profiler runtime. With
// relocatable device code the registration is an offloading entry the linker
// wrapper registers; its offload kind is CUDA's (2) or HIP's (4).
//
// On AMDGPU the profiled code object also links with no undefined symbol,
// malloc included (the derivative of a loop with a run-time trip count
// allocates its cache); the code object compile passes the flags with -Xclang
// -mllvm because the device linker knows no Poseidon flag. The AMDGPU profiler
// bitcode is built for POSEIDON_AMDGPU_ARCH, so a module for another processor
// (gfx1100) whose probes call into it is refused with an error at the user's
// kernel and leaves no launch manifest and no temporary file; one without
// probes takes the table alone.

#include "poseidon/poseidon.h"

// The runtime declarations a kernel stub needs.
struct dim3 {
  unsigned x, y, z;
};
#ifdef __HIP__
typedef struct ihipStream_t *hipStream_t;
extern "C" int hipLaunchKernel(const void *, dim3, dim3, void **,
                               unsigned long, hipStream_t);
#else
typedef struct CUstream_st *cudaStream_t;
extern "C" int cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                                void **args, unsigned long sharedMem,
                                cudaStream_t stream);
#endif

#ifndef NO_SITE
POSEIDON_OPTIMIZE
#endif
__attribute__((global)) void scale(double *x, double a) {
  x[0] = a * x[0] + 1.0;
}

#ifdef LOOP
POSEIDON_OPTIMIZE __attribute__((global)) void sum(const double *x,
                                                   double *y) {
  double s = 0.0;
  for (int i = 0; s < 10.0; ++i)
    s += x[i] * x[i];
  y[0] = s;
}
#endif

// DEV-DAG: @llvm.compiler.used = {{.*}}@__poseidon_profile_table
// DEV-NVPTX-DAG: @__poseidon_profile_table = weak_odr dso_local addrspace(4) externally_initialized constant {{.*}} zeroinitializer
// DEV-AMDGPU-DAG: @__poseidon_profile_table = weak_odr protected addrspace(4) externally_initialized constant {{.*}} zeroinitializer
// DEV-DAG: define internal void @poseidonLogValueCUDA(
// DEV-DAG: define internal void @poseidonLogGradCUDA(
// DEV-DAG: define internal void @poseidonProfileBlockDimsCUDA(
// DEV-DAG: define internal void @poseidonProfileStaticCUDA(
// DEV-NVPTX-DAG: define ptx_kernel void @_Z5scalePdd(
// DEV-NVPTX-NOT: "target-cpu"="sm_60"
// DEV-AMDGPU-DAG: define amdgpu_kernel void @_Z5scalePdd(
// DEV-AMDGPU-DAG: define internal {{.*}}@__ockl_printf_append_string_n({{.*}}) {{.*}}#[[OCKL:[0-9]+]] {{.*}}{
// DEV-AMDGPU: attributes #[[OCKL]] = {
// DEV-AMDGPU-NOT: "target-features"
// DEV-AMDGPU-SAME: }

// ARCH: device_profile_link.cu:[[@LINE-29]]:30: error: Poseidon: the AMDGPU device profiler was built for '[[ARCH]]' (POSEIDON_AMDGPU_ARCH) but scale(double*, double) compiles for target-cpu 'gfx1100'; build Poseidon for this processor, or compile the profiling build for that one

// RT: define internal void @[[FN]]({{.*}}) {{.*}}#[[ATTRS:[0-9]+]] {{.*}}{
// RT: attributes #[[ATTRS]] = {
// RT-NOT: "target-
// RT-SAME: }

// NOSITE-NVPTX: @llvm.compiler.used = appending global [1 x ptr] [ptr addrspacecast (ptr addrspace(4) @__poseidon_profile_table to ptr)], section "llvm.metadata"
// NOSITE-NVPTX-NEXT: @__poseidon_profile_table = weak_odr dso_local addrspace(4) externally_initialized constant %struct.PoseidonCUDAProfileTable zeroinitializer, align 8{{(, !guid ![0-9]+)?$}}
// NOSITE-AMDGPU: @llvm.compiler.used = appending addrspace(1) global [{{[0-9]+}} x ptr] [{{.*}}, ptr addrspacecast (ptr addrspace(4) @__poseidon_profile_table to ptr)], section "llvm.metadata"
// NOSITE-AMDGPU-NEXT: @__poseidon_profile_table = weak_odr protected addrspace(4) externally_initialized constant %struct.PoseidonCUDAProfileTable zeroinitializer, align 8{{(, !guid ![0-9]+)?$}}
// NOSITE-NOT: poseidonLogValueCUDA
// NOSITE-NOT: @poseidon
// NOSITE-NVPTX: define {{.*}}ptx_kernel void @_Z5scalePdd(
// NOSITE-AMDGPU: define protected amdgpu_kernel void @_Z5scalePdd(
// NOSITE-NOT: @poseidon

// HOST: @__poseidon_profile_table = internal global [[[N:[0-9]+]] x i8] zeroinitializer, align 8
// HOST-NEXT: @[[NAME:[0-9]+]] = private unnamed_addr constant [25 x i8] c"__poseidon_profile_table\00"
// HOST: @llvm.global_ctors = appending global [3 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @__[[RT]]_module_ctor, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @__poseidon_profile_register_module_table, ptr null }, { i32, ptr, ptr } { i32 0, ptr @poseidon_register_static__Z5scalePdd, ptr null }]
// HOST: define internal {{(fastcc )?}}void @__[[RT]]_register_globals(ptr %[[H:[0-9]+]])
// HOST: call void @__[[RT]]RegisterVar(ptr %[[H]], ptr @__poseidon_profile_table, ptr @[[NAME]], ptr @[[NAME]], i32 0, i64 [[N]], i32 1, i32 0)
// HOST-NEXT: ret void
// HOST: define internal void @__poseidon_profile_register_module_table()
// HOST-NEXT: call void @__poseidon_profile_register_module(ptr @__poseidon_profile_table, i32 1, i32 1)
// HOST-NEXT: ret void

// RDC: @__poseidon_profile_table = internal global [[[N:[0-9]+]] x i8] zeroinitializer, align 8
// RDC-NEXT: @[[ENTRY:.offloading.entry_name[.0-9]*]] = internal unnamed_addr constant [25 x i8] c"__poseidon_profile_table\00", section ".llvm.rodata.offloading"
// RDC-NEXT: @.offloading.entry.__poseidon_profile_table = weak {{.*}}constant %struct.__tgt_offload_entry { i64 0, i16 1, i16 [[KIND]], i32 16, ptr @__poseidon_profile_table, ptr @[[ENTRY]], i64 [[N]], i64 0, ptr null }, section "llvm_offload_entries"
// RDC: @llvm.global_ctors = appending global [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @__poseidon_profile_register_module_table, ptr null }, { i32, ptr, ptr } { i32 0, ptr @poseidon_register_static__Z5scalePdd, ptr null }]
// RDC: define internal void @__poseidon_profile_register_module_table()
// RDC-NEXT: call void @__poseidon_profile_register_module(ptr @__poseidon_profile_table, i32 1, i32 1)
// RDC-NEXT: ret void

// PLAIN-NOT: __poseidon_profile
// PLAIN-NVPTX: define dso_local ptx_kernel void @_Z5scalePdd(
// PLAIN-AMDGPU: define protected amdgpu_kernel void @_Z5scalePdd(
// PLAIN: define dso_local void @_Z20__device_stub__scalePdd(

// CO-NOT: UND {{[_a-zA-Z]}}
// CO: OBJECT WEAK PROTECTED {{[0-9]+}} __poseidon_profile_table{{$}}
// CO-NOT: UND {{[_a-zA-Z]}}
// CO: FUNC GLOBAL PROTECTED {{[0-9]+}} _Z5scalePdd{{$}}
// CO-NOT: UND {{[_a-zA-Z]}}
// CO: FUNC GLOBAL PROTECTED {{[0-9]+}} _Z3sumPKdPd{{$}}
// CO-NOT: UND {{[_a-zA-Z]}}
