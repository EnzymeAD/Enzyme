// REQUIRES: enzyme, nvptx-device-profiler
// DEFINE: %{clang} = %clang -x cuda -nocudainc -nocudalib -S -emit-llvm \
// DEFINE:   -I%poseidon_profile_inc %loadClangPoseidonEnzyme %s
// RUN: rm -rf %t && mkdir -p %t && echo fatbin > %t/k.fatbin
// RUN: %{clang} --cuda-device-only --cuda-gpu-arch=%gpu_arch -O2 -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c -mllvm -poseidon-unit=scale -o %t/dev.ll
// RUN: FileCheck --check-prefix=DEV %s < %t/dev.ll
// RUN: FileCheck --check-prefix=RT -DFN=poseidonLogValueCUDA --implicit-check-not='declare {{.*}}@poseidon' %s < %t/dev.ll
// RUN: FileCheck --check-prefix=RT -DFN=poseidonLogGradCUDA --implicit-check-not='declare {{.*}}@poseidon' %s < %t/dev.ll
// RUN: FileCheck --check-prefix=RT -DFN=poseidonProfileBlockDimsCUDA --implicit-check-not='declare {{.*}}@poseidon' %s < %t/dev.ll
// RUN: FileCheck --check-prefix=RT -DFN=poseidonProfileStaticCUDA --implicit-check-not='declare {{.*}}@poseidon' %s < %t/dev.ll
// RUN: %{clang} --cuda-device-only --cuda-gpu-arch=%gpu_arch -O2 -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c -mllvm -poseidon-unit=nosite -DNO_SITE -o %t/nosite.ll
// RUN: FileCheck --check-prefix=NOSITE %s < %t/nosite.ll
// RUN: %{clang} --cuda-host-only -O0 -Xclang -fcuda-include-gpubinary -Xclang %t/k.fatbin -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c -mllvm -poseidon-unit=scale -o %t/host.ll
// RUN: FileCheck --check-prefix=HOST %s < %t/host.ll
// RUN: %{clang} --cuda-host-only -fgpu-rdc -O0 -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c -mllvm -poseidon-unit=scale -o %t/rdc.ll
// RUN: FileCheck --check-prefix=RDC --implicit-check-not=RegisterVar %s < %t/rdc.ll
// RUN: %{clang} --cuda-device-only --cuda-gpu-arch=%gpu_arch -O2 -o %t/plain-dev.ll
// RUN: %{clang} --cuda-host-only -O0 -Xclang -fcuda-include-gpubinary -Xclang %t/k.fatbin -o %t/plain-host.ll
// RUN: cat %t/plain-dev.ll %t/plain-host.ll | FileCheck --check-prefix=PLAIN --implicit-check-not=__poseidon_profile --implicit-check-not=@poseidon %s

// Profile generation without relocatable device code: the device module gets
// the device half of the profiler, linked from the bitcode the plugin carries,
// local to the module and with no target-cpu or target-features of the
// processor that bitcode was built for, and a table every module defines
// whether or not it has a probe; the host half of the same unit registers that
// table with the CUDA runtime and with the profiler runtime. With relocatable
// device code the registration is an offloading entry the linker wrapper
// registers.

#include "poseidon/poseidon.h"

#define __global__ __attribute__((global))

// What the launch stub needs of the CUDA runtime headers.
struct dim3 {
  unsigned x, y, z;
};
typedef struct CUstream_st *cudaStream_t;
extern "C" int cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                                void **args, unsigned long sharedMem,
                                cudaStream_t stream);

#ifndef NO_SITE
POSEIDON_OPTIMIZE
#endif
__global__ void scale(double *x, double a) { x[0] = a * x[0] + 1.0; }

// DEV-DAG: @llvm.compiler.used = {{.*}}@__poseidon_profile_table
// DEV-DAG: @__poseidon_profile_table = weak_odr dso_local addrspace(4) externally_initialized constant {{.*}} zeroinitializer
// DEV-DAG: define internal void @poseidonLogValueCUDA(
// DEV-DAG: define internal void @poseidonLogGradCUDA(
// DEV-DAG: define internal void @poseidonProfileBlockDimsCUDA(
// DEV-DAG: define internal void @poseidonProfileStaticCUDA(
// DEV-DAG: define ptx_kernel void @_Z5scalePdd(
// DEV-NOT: "target-cpu"="sm_60"

// RT: define internal void @[[FN]]({{.*}}) {{.*}}#[[ATTRS:[0-9]+]] {{.*}}{
// RT: attributes #[[ATTRS]] = {
// RT-NOT: "target-
// RT-SAME: }

// NOSITE: @llvm.compiler.used = appending global [1 x ptr] [ptr addrspacecast (ptr addrspace(4) @__poseidon_profile_table to ptr)], section "llvm.metadata"
// NOSITE-NEXT: @__poseidon_profile_table = weak_odr dso_local addrspace(4) externally_initialized constant %struct.PoseidonCUDAProfileTable zeroinitializer, align 8
// NOSITE-NOT: poseidonLogValueCUDA
// NOSITE-NOT: @poseidon
// NOSITE: define dso_local ptx_kernel void @_Z5scalePdd(
// NOSITE-NOT: @poseidon

// HOST: @__poseidon_profile_table = internal global [[[N:[0-9]+]] x i8] zeroinitializer, align 8
// HOST-NEXT: @[[NAME:[0-9]+]] = private unnamed_addr constant [25 x i8] c"__poseidon_profile_table\00"
// HOST: @llvm.global_ctors = appending global [3 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @__cuda_module_ctor, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @__poseidon_profile_register_module_table, ptr null }, { i32, ptr, ptr } { i32 0, ptr @poseidon_register_static_preprocess__Z5scalePdd_poseidon_body, ptr null }]
// HOST: define internal {{.*}}void @__cuda_register_globals(ptr %[[H:[0-9]+]])
// HOST: call void @__cudaRegisterVar(ptr %[[H]], ptr @__poseidon_profile_table, ptr @[[NAME]], ptr @[[NAME]], i32 0, i64 [[N]], i32 1, i32 0)
// HOST-NEXT: ret void
// HOST: define internal void @__poseidon_profile_register_module_table()
// HOST-NEXT: call void @__poseidon_profile_register_module(ptr @__poseidon_profile_table)
// HOST-NEXT: ret void

// RDC: @__poseidon_profile_table = internal global [[[N:[0-9]+]] x i8] zeroinitializer, align 8
// RDC-NEXT: @[[ENTRY:.offloading.entry_name[.0-9]*]] = internal unnamed_addr constant [25 x i8] c"__poseidon_profile_table\00", section ".llvm.rodata.offloading"
// RDC-NEXT: @.offloading.entry.__poseidon_profile_table = weak {{.*}}constant %struct.__tgt_offload_entry { i64 0, i16 1, i16 2, i32 16, ptr @__poseidon_profile_table, ptr @[[ENTRY]], i64 [[N]], i64 0, ptr null }, section "llvm_offload_entries"
// RDC: @llvm.global_ctors = appending global [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @__poseidon_profile_register_module_table, ptr null }, { i32, ptr, ptr } { i32 0, ptr @poseidon_register_static_preprocess__Z5scalePdd_poseidon_body, ptr null }]
// RDC: define internal void @__poseidon_profile_register_module_table()
// RDC-NEXT: call void @__poseidon_profile_register_module(ptr @__poseidon_profile_table)
// RDC-NEXT: ret void

// PLAIN-NOT: __poseidon_profile
// PLAIN: define dso_local ptx_kernel void @_Z5scalePdd(
// PLAIN: define dso_local void @_Z20__device_stub__scalePdd(
