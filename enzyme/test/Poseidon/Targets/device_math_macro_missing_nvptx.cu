// REQUIRES: cuda-runtime
// RUN: not %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -UPOSEIDON_DEVICE_MATH -mllvm -poseidon-profile-use=%S/../Inputs/other_sites %s -o /dev/null 2>&1 | FileCheck %s
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites %s -o /dev/null 2>&1 | FileCheck --allow-empty --check-prefix=DEFINED %s
// RUN: not %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -UPOSEIDON_DEVICE_MATH -DKERNELS -mllvm -poseidon-kernels=wave -mllvm -poseidon-profile-use=%S/../Inputs/other_sites %s -o /dev/null 2>&1 | FileCheck %s
// RUN: not %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -UPOSEIDON_DEVICE_MATH -DMARKER -mllvm -poseidon-profile-use=%S/../Inputs/other_sites %s -o /dev/null 2>&1 | FileCheck %s
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -UPOSEIDON_DEVICE_MATH %s -o /dev/null 2>&1 | FileCheck --allow-empty --check-prefix=INACTIVE %s

// A device module with a site compiled with the plugin and without
// POSEIDON_DEVICE_MATH, which keeps the libdevice functions a rewrite or a
// derivative calls, stops a profile action with an error that names the macro,
// whether the site is an annotation, a kernel -poseidon-kernels names or a
// marker call; a compile without a profile action leaves the site as written.

#include "poseidon/poseidon.h"

#if defined(KERNELS) || defined(MARKER)
#define SITE
#else
#define SITE POSEIDON_OPTIMIZE
#endif

SITE __attribute__((global)) void wave(double *o, const double *a) {
  o[0] = a[0] * a[0] + a[1];
}

#ifdef MARKER
__attribute__((device)) void body(double *o, const double *a) {
  o[0] = a[0] * a[0] + a[1];
}
extern "C" __attribute__((device)) void __poseidon_fp_optimize(void *, ...);
__attribute__((global)) void marked(double *o, const double *a) {
  __poseidon_fp_optimize((void *)body, o, a);
}
#endif

// CHECK: error: Poseidon: this device compilation has Poseidon sites and does not include poseidon/poseidon.h under POSEIDON_DEVICE_MATH, under which the header keeps the libdevice functions a rewrite or a site's derivative calls; include the header and define the macro (-DPOSEIDON_DEVICE_MATH), as poseidon-clang and the Poseidon CMake targets do{{$}}
// DEFINED-NOT: POSEIDON_DEVICE_MATH
// INACTIVE-NOT: POSEIDON_DEVICE_MATH
