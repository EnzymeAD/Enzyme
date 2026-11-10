// REQUIRES: enzyme, amdgpu-registered-target, rocm
// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-device-only -foffload-lto -cuid=0 -O2 -c -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -Xclang -mllvm -Xclang -poseidon-profile-use=%S/Inputs/other_sites -Xclang -mllvm -Xclang -poseidon-cache=%t/c %s -o %t/dev.bc
// RUN: llvm-dis %t/dev.bc -o - | FileCheck %s

// A HIP unit compiles its device half to bitcode the device linker reads back
// (what -c does without --cuda-device-only). Enzyme's plugin annotates the
// device math functions it keeps without arguments, which on AMDGPU types the
// array before an annotation with an argument; once Enzyme releases its own,
// the array takes the type of what is left, and the module reads back. The
// site's annotation leaves with the module as written, so what is left is
// shift's.

#include <hip/hip_runtime.h>

#include "poseidon/poseidon.h"

POSEIDON_OPTIMIZE_TAU(1e-9) __global__ void scale(double *x, double a) {
  x[0] = sqrt(a * x[0]) + 1.0;
}

__attribute__((annotate("kept", 1))) __global__ void shift(double *x) {
  x[1] = x[1] + 1.0;
}

// CHECK: @llvm.global.annotations = appending addrspace(1) global [1 x { ptr addrspace(1), ptr addrspace(4), ptr addrspace(4), i32, ptr addrspace(1) }] [{{.*}} @_Z5shiftPd
