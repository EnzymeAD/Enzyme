// REQUIRES: enzyme, nvptx-device-profiler
// DEFINE: %{gen} = %clang -x cuda -nocudainc -nocudalib --cuda-device-only \
// DEFINE:   --cuda-gpu-arch=%gpu_arch -O2 -S -emit-llvm -I%poseidon_profile_inc \
// DEFINE:   %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate %s
// RUN: rm -rf %t && mkdir %t
// RUN: %{gen} -mllvm -poseidon-cache=%t/c -o %t/dev.ll
// RUN: %{gen} -fgpu-rdc -mllvm -poseidon-cache=%t/r -o %t/rdc.ll
// RUN: cat %t/dev.ll %t/rdc.ll | FileCheck %s

// A static or anonymous-namespace kernel is local to its source file, though
// a device compile gives it external linkage (and with -fgpu-rdc a renamed
// symbol): its profile is named with the GUID of that identity, alike in
// both, and an external kernel's by its symbol alone.

#include "poseidon/poseidon.h"

#define __global__ __attribute__((global))

POSEIDON_OPTIMIZE __global__ static void step(double *x, int n) {
  x[n] = x[n] * x[n] + 1.0;
}

namespace {
POSEIDON_OPTIMIZE __global__ void scale(double *x, int n) {
  x[n] = x[n] / (x[n] + 2.0);
}
} // namespace

POSEIDON_OPTIMIZE __global__ void shared(double *x, int n) {
  x[n] = x[n] * 0.5 + x[n];
}

// CHECK-DAG: c"_ZL4stepPdi~[[STEP:[0-9a-f]+]]\00"
// CHECK-DAG: c"_ZN12_GLOBAL__N_15scaleEPdi~[[SCALE:[0-9a-f]+]]\00"
// CHECK-DAG: c"_Z6sharedPdi\00"
// CHECK: target triple
// CHECK-DAG: c"_ZL4stepPdi~[[STEP]]\00"
// CHECK-DAG: c"_ZN12_GLOBAL__N_15scaleEPdi~[[SCALE]]\00"
// CHECK-DAG: c"_Z6sharedPdi\00"
