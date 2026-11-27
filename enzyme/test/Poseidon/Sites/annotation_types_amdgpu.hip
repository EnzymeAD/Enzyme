// REQUIRES: enzyme, amdgpu-registered-target, rocm
// RUN: rm -rf %t && mkdir %t
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-device-only -foffload-lto -cuid=0 -O2 -c -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -Xclang -mllvm -Xclang -poseidon-profile-use=%S/../Inputs/unrelated_site_profile -Xclang -mllvm -Xclang -poseidon-cache=%t/c %s -o %t/dev.bc
// RUN: llvm-dis %t/dev.bc -o - | FileCheck --implicit-check-not='c"poseidon\00"' %s

// A HIP device half as bitcode the device linker reads back: once Enzyme's
// argument-less annotations and the site's leave, llvm.global.annotations takes
// the type of what is left (shift's), and the module reads back.

#include <hip/hip_runtime.h>

#include "poseidon/poseidon.h"

POSEIDON_OPTIMIZE_TAU(1e-9) __global__ void scale(double *x, double a) {
  x[0] = sqrt(a * x[0]) + 1.0;
}

__attribute__((annotate("kept", 1))) __global__ void shift(double *x) {
  x[1] = x[1] + 1.0;
}

// CHECK: [[FILE:@[^ ]+]] = private unnamed_addr addrspace(4) constant [{{[0-9]+}} x i8] c"{{.*}}annotation_types_amdgpu.hip\00", section "llvm.metadata"
// CHECK: [[KEPT:@[^ ]+]] = private unnamed_addr addrspace(4) constant [5 x i8] c"kept\00", section "llvm.metadata"
// CHECK: [[ARGS:@[^ ]+]] = private unnamed_addr addrspace(1) constant { i32 } { i32 1 }, section "llvm.metadata"
// CHECK: @llvm.global.annotations = appending addrspace(1) global [1 x { ptr addrspace(1), ptr addrspace(4), ptr addrspace(4), i32, ptr addrspace(1) }] [{ ptr addrspace(1), ptr addrspace(4), ptr addrspace(4), i32, ptr addrspace(1) } { ptr addrspace(1) addrspacecast (ptr @_Z5shiftPd to ptr addrspace(1)), ptr addrspace(4) [[KEPT]], ptr addrspace(4) [[FILE]], i32 [[@LINE-7]], ptr addrspace(1) [[ARGS]] }], section "llvm.metadata"

// CHECK: define protected amdgpu_kernel void @_Z5scalePdd(ptr addrspace(1) {{.*}}[[X:%[0-9]+]], double {{.*}}[[A:%[0-9]+]])
// CHECK-NEXT: [[XV:%[0-9]+]] = load double, ptr addrspace(1) [[X]]
// CHECK-NEXT: [[AX:%[0-9]+]] = fmul contract double [[A]], [[XV]]
// CHECK-NEXT: [[S:%[0-9]+]] = tail call contract {{.*}}double @llvm.sqrt.f64(double [[AX]])
// CHECK-NEXT: [[R:%[0-9]+]] = fadd contract double [[S]], 1.000000e+00
// CHECK-NEXT: store double [[R]], ptr addrspace(1) [[X]]
// CHECK-NEXT: ret void
