// REQUIRES: cuda-runtime
// RUN: not %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -UPOSEIDON_DEVICE_MATH -mllvm -poseidon-profile-use=%S/../Inputs/other_sites %s -o /dev/null 2>&1 | FileCheck %s
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites %s -o /dev/null 2>&1 | FileCheck --allow-empty --check-prefix=DEFINED %s
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -UPOSEIDON_DEVICE_MATH %s -o /dev/null 2>&1 | FileCheck --allow-empty --check-prefix=INACTIVE %s

// A device module with a site from poseidon.h compiled with the plugin and
// without POSEIDON_DEVICE_MATH, which keeps the libdevice functions a rewrite
// or a derivative calls, stops a profile action with an error that names the
// macro; a compile without a profile action leaves the site as written.

#include "poseidon/poseidon.h"

POSEIDON_OPTIMIZE __attribute__((global)) void wave(double *o, const double *a) {
  o[0] = a[0] * a[0] + a[1];
}

// CHECK: error: Poseidon: this device compilation has Poseidon sites from poseidon/poseidon.h and was compiled without POSEIDON_DEVICE_MATH, under which the header keeps the libdevice functions a rewrite or a site's derivative calls; define it (-DPOSEIDON_DEVICE_MATH), as poseidon-clang and the Poseidon CMake targets do{{$}}
// DEFINED-NOT: POSEIDON_DEVICE_MATH
// INACTIVE-NOT: POSEIDON_DEVICE_MATH
