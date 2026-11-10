// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%FPProfileInc %clangLoadPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/cache %s -o - 2>/dev/null | FileCheck %s
// REQUIRES: poseidon, enzyme

// The profiled kernel runs the reverse pass and is launched with the block
// the original is launched with, so it is bounded to the largest block a
// launch may have; a kernel that declares its own bound keeps it.

#include "poseidon/poseidon.h"

POSEIDON_OPTIMIZE __attribute__((global)) void unbounded(double *y,
                                                         const double *x,
                                                         int n) {
  int i = __nvvm_read_ptx_sreg_tid_x();
  if (i < n)
    y[i] = x[i] * x[i] + y[i];
}

POSEIDON_OPTIMIZE __attribute__((global, launch_bounds(256))) void
bounded(double *y, const double *x, int n) {
  int i = __nvvm_read_ptx_sreg_tid_x();
  if (i < n)
    y[i] = x[i] * x[i] + y[i];
}

// CHECK-DAG: define {{.*}}ptx_kernel void @_Z9unboundedPdPKdi({{.*}}) {{.*}}#[[UNB:[0-9]+]]
// CHECK-DAG: define {{.*}}ptx_kernel void @_Z7boundedPdPKdi({{.*}}) {{.*}}#[[BND:[0-9]+]]
// CHECK-DAG: attributes #[[UNB]] = {{{.*}}"nvvm.maxntid"="1024"
// CHECK-DAG: attributes #[[BND]] = {{{.*}}"nvvm.maxntid"="256"
