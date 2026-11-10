// RUN: rm -rf %t && mkdir -p %t && echo fatbin > %t/k.fatbin
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%FPProfileInc %clangLoadPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c -mllvm -poseidon-unit=scale %s -o %t/dev.ll
// RUN: FileCheck --check-prefix=DEV %s < %t/dev.ll
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%FPProfileInc %clangLoadPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c -mllvm -poseidon-unit=nosite -DNO_SITE %s -o %t/nosite.ll
// RUN: FileCheck --check-prefix=NOSITE %s < %t/nosite.ll
// RUN: %clang -x cuda --cuda-host-only -nocudainc -nocudalib -O0 -S -emit-llvm -I%FPProfileInc -Xclang -fcuda-include-gpubinary -Xclang %t/k.fatbin %clangLoadPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c -mllvm -poseidon-unit=scale %s -o %t/host.ll
// RUN: FileCheck --check-prefix=HOST %s < %t/host.ll
// RUN: %clang -x cuda --cuda-host-only -fgpu-rdc -nocudainc -nocudalib -O0 -S -emit-llvm -I%FPProfileInc %clangLoadPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c -mllvm -poseidon-unit=scale %s -o %t/rdc.ll
// RUN: FileCheck --check-prefix=RDC %s < %t/rdc.ll
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%FPProfileInc %clangLoadPoseidonEnzyme %s -o %t/plain-dev.ll
// RUN: %clang -x cuda --cuda-host-only -nocudainc -nocudalib -O0 -S -emit-llvm -I%FPProfileInc -Xclang -fcuda-include-gpubinary -Xclang %t/k.fatbin %clangLoadPoseidonEnzyme %s -o %t/plain-host.ll
// RUN: cat %t/plain-dev.ll %t/plain-host.ll | FileCheck --check-prefix=PLAIN %s
// REQUIRES: poseidon, enzyme

// Profile generation without relocatable device code: the device module gets
// the device half of the profiler, linked from the bitcode the plugin carries
// and local to the module, and a table every module defines whether or not it
// has a probe; the host half of the same unit registers that table with the
// CUDA runtime and with the profiler runtime. With relocatable device code the
// registration is an offloading entry the linker wrapper registers.

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
// DEV-NOT: "target-cpu"="sm_60"

// NOSITE: @__poseidon_profile_table = weak_odr dso_local addrspace(4)
// NOSITE-NOT: poseidonLogValueCUDA

// HOST: @__poseidon_profile_table = internal global [[[N:[0-9]+]] x i8] zeroinitializer, align 8
// HOST: @llvm.global_ctors = appending global [3 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @__cuda_module_ctor, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @__poseidon_profile_register_module_table, ptr null }, { i32, ptr, ptr } { i32 0, ptr @poseidon_register_static_preprocess__Z5scalePdd_poseidon_body, ptr null }]
// HOST: define internal {{.*}}void @__cuda_register_globals(ptr %[[H:[0-9]+]])
// HOST: call void @__cudaRegisterVar(ptr %[[H]], ptr @__poseidon_profile_table, ptr @[[NAME:[^,]+]], ptr @[[NAME]], i32 0, i64 [[N]], i32 1, i32 0)
// HOST-NEXT: ret void
// HOST: define internal void @__poseidon_profile_register_module_table()
// HOST-NEXT: call void @__poseidon_profile_register_module(ptr @__poseidon_profile_table)

// RDC: @__poseidon_profile_table = internal global [[[N:[0-9]+]] x i8] zeroinitializer, align 8
// RDC: @.offloading.entry.__poseidon_profile_table = weak {{.*}}constant %struct.__tgt_offload_entry { i64 0, i16 1, i16 2, i32 16, ptr @__poseidon_profile_table, ptr @{{[^,]+}}, i64 [[N]], i64 0, ptr null }, section "llvm_offload_entries"
// RDC: @llvm.global_ctors = {{.*}} ptr @__poseidon_profile_register_module_table
// RDC: define internal void @__poseidon_profile_register_module_table()
// RDC-NEXT: call void @__poseidon_profile_register_module(ptr @__poseidon_profile_table)

// PLAIN-NOT: __poseidon_profile
