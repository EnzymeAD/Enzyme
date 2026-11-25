// REQUIRES: enzyme, nvptx-device-profiler
// RUN: rm -rf %t && mkdir %t
// RUN: not %clang -x cuda -nocudainc -nocudalib --cuda-device-only --cuda-gpu-arch=%gpu_arch -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache= %s -o /dev/null 2>&1 | FileCheck %s
// RUN: %clang -x cuda -nocudainc -nocudalib --cuda-device-only --cuda-gpu-arch=%gpu_arch -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache= -mllvm -poseidon-launch-dir=%t/l %s -o /dev/null
// RUN: ls %t/l | FileCheck --check-prefix=MANIFEST %s

// A device compile whose sites hand launches to the host half needs a
// directory to keep them in: with the cache disabled it is an error that
// names -poseidon-launch-dir, and with one named the manifest is written
// there.

#include "poseidon/poseidon.h"

#define __global__ __attribute__((global))

POSEIDON_OPTIMIZE __global__ void scale(double *x, int n) {
  x[n] = x[n] * x[n] + 1.0;
}

// CHECK: error: Poseidon: this device compile hands the launches of its sites to its host half, and -poseidon-cache= names no directory to keep them in; pass -poseidon-launch-dir=<dir> to both halves

// MANIFEST: launches_launch_dir_required_nvptx.cu_{{[0-9A-F]+}}.json
