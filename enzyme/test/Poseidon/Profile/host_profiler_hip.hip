// REQUIRES: enzyme, amdgpu-registered-target, rocm
// DEFINE: %{clang} = %clang -S -emit-llvm -I%poseidon_profile_inc \
// DEFINE:   %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate \
// DEFINE:   -mllvm -poseidon-cache=%t/c
// RUN: rm -rf %t && mkdir %t
// RUN: %{clang} -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -mllvm -poseidon-compilation-id=scale %s -o %t/dev.ll
// RUN: %{clang} -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-host-only -cuid=0 -O0 -mllvm -poseidon-compilation-id=scale %s -o %t/host.ll
// RUN: FileCheck --check-prefix=HOST %s < %t/host.ll
// RUN: %{clang} -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-host-only -cuid=0 -O0 -mllvm -poseidon-compilation-id=main -DMAIN_ONLY %s -o %t/main.ll
// RUN: FileCheck --check-prefix=INIT %s < %t/main.ll
// RUN: %{clang} -x c++ -O0 -DMAIN_ONLY -DNO_HIP %s -o %t/cxx.ll
// RUN: FileCheck --check-prefix=CXX %s < %t/cxx.ll

// The host half of HIP profile generation: main binds the profiler first,
// with kernels or without; a profiled launch goes through the runtime helper
// with the popped configuration, the argument array and its pointer argument
// (index 0, offset 0, seeded as double). A C++ unit gets no call.

#ifndef NO_HIP
#include <hip/hip_runtime.h>
#endif

#include "poseidon/poseidon.h"

#ifndef MAIN_ONLY
POSEIDON_OPTIMIZE __global__ void scale(double *x, double a) {
  x[0] = a * x[0] + 1.0;
}

void launch(double *x) { scale<<<1, 1>>>(x, 2.0); }
#endif

int main() { return 0; }

// HOST: @_Z5scalePdd = {{.*}}constant ptr @_Z20__device_stub__scalePdd, align 8
// HOST: @poseidon.ptr._Z5scalePdd = private unnamed_addr constant [1 x i32] zeroinitializer
// HOST: @poseidon.off._Z5scalePdd = private unnamed_addr constant [1 x i32] zeroinitializer
// HOST: @poseidon.seed._Z5scalePdd = private unnamed_addr constant [1 x i32] [i32 8]
// HOST-LABEL: define {{.*}}void @_Z20__device_stub__scalePdd(
// HOST: %[[ARGS:[0-9]+]] = alloca ptr, i64 2
// HOST: call i32 @__hipPopCallConfiguration(ptr %[[GRID:[0-9]+]], ptr %[[BLOCK:[0-9]+]], ptr %[[SHMEMP:[0-9]+]], ptr %[[STREAMP:[0-9]+]])
// HOST: %[[SHMEM:[0-9]+]] = load i64, ptr %[[SHMEMP]],
// HOST: %[[STREAM:[0-9]+]] = load ptr, ptr %[[STREAMP]],
// HOST: call void @__poseidon_launch_profiled(ptr @_Z5scalePdd, i32 0, ptr %[[GRID]], ptr %[[BLOCK]], i64 %[[SHMEM]], ptr %[[STREAM]], i32 2, ptr %[[ARGS]], i32 1, ptr @poseidon.ptr._Z5scalePdd, ptr @poseidon.off._Z5scalePdd, ptr @poseidon.seed._Z5scalePdd)
// HOST-NOT: call {{.*}}@hipLaunchKernel(
// HOST-LABEL: define {{.*}}i32 @main(
// HOST-NEXT: call void @__poseidon_profile_init_device()
// INIT-LABEL: define {{.*}}i32 @main(
// INIT-NEXT: call void @__poseidon_profile_init_device()
// CXX-NOT: __poseidon_profile_init_device
