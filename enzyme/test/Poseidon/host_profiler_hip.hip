// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -S -emit-llvm -I%FPProfileInc %clangLoadPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c %s -o %t/dev.ll
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-host-only -cuid=0 -O0 -S -emit-llvm -I%FPProfileInc %clangLoadPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c %s -o %t/host.ll 2>&1 | FileCheck --check-prefix=MSG %s
// RUN: FileCheck --check-prefix=HOST %s < %t/host.ll
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-host-only -cuid=0 -O0 -S -emit-llvm -I%FPProfileInc %clangLoadPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c -DMAIN_ONLY %s -o %t/main.ll
// RUN: FileCheck --check-prefix=INIT %s < %t/main.ll
// RUN: %clang -x c++ -O0 -S -emit-llvm -I%FPProfileInc %clangLoadPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c -DMAIN_ONLY -DNO_HIP %s -o %t/cxx.ll
// RUN: FileCheck --check-prefix=CXX %s < %t/cxx.ll
// REQUIRES: poseidon, enzyme, amdgpu-registered-target, rocm

// The host half of a HIP profile generation: main binds the profiler before
// anything else runs, in a unit with kernels or without, and the launch of a
// profiled kernel goes through the runtime helper that supplies its shadows,
// handed the kernel's handle, the configuration HIP popped and the launch's
// argument array. A C++ unit links no GPU profiler and gets no call.

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

// MSG: [poseidon-profgen] launch of _Z5scalePdd now allocates and seeds 1 shadow buffer(s)
// MSG: [poseidon] FP profiler CUDA registration injected at main

// HOST-LABEL: define {{.*}}void @_Z20__device_stub__scalePdd(
// HOST: call i32 @__hipPopCallConfiguration(ptr %[[GRID:[0-9]+]], ptr %[[BLOCK:[0-9]+]], ptr %{{[0-9]+}}, ptr %{{[0-9]+}})
// HOST: call void @__poseidon_launch_profiled(ptr @_Z5scalePdd, i32 0, ptr %[[GRID]], ptr %[[BLOCK]], i64 %{{[0-9]+}}, ptr %{{[0-9]+}}, i32 2, ptr %{{[0-9]+}}, i32 1,
// HOST-NOT: call {{.*}}@hipLaunchKernel(
// HOST-LABEL: define {{.*}}i32 @main(
// HOST-NEXT: call void @__poseidon_profile_init_cuda()

// INIT-LABEL: define {{.*}}i32 @main(
// INIT-NEXT: call void @__poseidon_profile_init_cuda()

// CXX-NOT: __poseidon_profile_init_cuda
