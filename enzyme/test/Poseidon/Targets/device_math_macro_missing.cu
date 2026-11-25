// REQUIRES: cuda-toolkit || amdgpu-registered-target
// From LLVM 22 an sm_120 compile needs the PTX version of a CUDA installation.
// DEFINE: %{nvptx} = %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon %s -o /dev/null
// DEFINE: %{amdgpu} = %clang -x hip --offload-arch=%amdgpu_arch -nogpulib -nogpuinc --cuda-device-only -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon %s -o /dev/null
// DEFINE: %{use} = -mllvm -poseidon-profile-use=%S/../Inputs/unrelated_site_profile
// RUN: %if cuda-toolkit %{ not %{nvptx} -UPOSEIDON_DEVICE_MATH %{use} 2>&1 | FileCheck -DLIB=libdevice %s %}
// RUN: %if cuda-toolkit %{ %{nvptx} %{use} 2>&1 | FileCheck --allow-empty --check-prefix=DEFINED %s %}
// RUN: %if cuda-toolkit %{ not %{nvptx} -UPOSEIDON_DEVICE_MATH -DKERNELS -mllvm -poseidon-kernels=wave %{use} 2>&1 | FileCheck -DLIB=libdevice %s %}
// RUN: %if cuda-toolkit %{ not %{nvptx} -UPOSEIDON_DEVICE_MATH -DMARKER %{use} 2>&1 | FileCheck -DLIB=libdevice %s %}
// RUN: %if cuda-toolkit %{ %{nvptx} -UPOSEIDON_DEVICE_MATH 2>&1 | FileCheck --allow-empty --check-prefix=INACTIVE %s %}
// RUN: %if amdgpu-registered-target %{ not %{amdgpu} -UPOSEIDON_DEVICE_MATH %{use} 2>&1 | FileCheck -DLIB=ocml %s %}
// RUN: %if amdgpu-registered-target %{ %{amdgpu} %{use} 2>&1 | FileCheck --allow-empty --check-prefix=DEFINED %s %}
// RUN: %if amdgpu-registered-target %{ not %{amdgpu} -UPOSEIDON_DEVICE_MATH -DKERNELS -mllvm -poseidon-kernels=wave %{use} 2>&1 | FileCheck -DLIB=ocml %s %}
// RUN: %if amdgpu-registered-target %{ not %{amdgpu} -UPOSEIDON_DEVICE_MATH -DMARKER %{use} 2>&1 | FileCheck -DLIB=ocml %s %}
// RUN: %if amdgpu-registered-target %{ %{amdgpu} -UPOSEIDON_DEVICE_MATH 2>&1 | FileCheck --allow-empty --check-prefix=INACTIVE %s %}

// A device module with a site compiled with the plugin and without
// POSEIDON_DEVICE_MATH, which keeps the device math functions (libdevice on
// NVPTX, ocml on AMDGPU) a rewrite or a derivative calls, stops a profile
// action with an error that names the macro, whether the site is an
// annotation, a kernel -poseidon-kernels names or a marker call; a compile
// without a profile action leaves the site as written.

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
// Declared with the arguments of its one call: before LLVM 23 CUDA and HIP
// device code take no variadic function.
extern "C" __attribute__((device)) void __poseidon_fp_optimize(void *, double *,
                                                              const double *);
__attribute__((global)) void marked(double *o, const double *a) {
  __poseidon_fp_optimize((void *)body, o, a);
}
#endif

// CHECK: error: Poseidon: this device compilation has Poseidon sites and does not include poseidon/poseidon.h under POSEIDON_DEVICE_MATH, under which the header keeps the [[LIB]] functions a rewrite or a site's derivative calls; include the header and define the macro (-DPOSEIDON_DEVICE_MATH), as poseidon-clang and the Poseidon CMake targets do{{$}}
// DEFINED-NOT: POSEIDON_DEVICE_MATH
// INACTIVE-NOT: POSEIDON_DEVICE_MATH
