// REQUIRES: cuda-toolkit || amdgpu-registered-target
// From LLVM 22 an sm_120 compile needs the PTX version of a CUDA installation.
// DEFINE: %{nvptx} = %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon %s -o /dev/null
// DEFINE: %{amdgpu} = %clang -x hip --offload-arch=%amdgpu_arch -nogpulib -nogpuinc --cuda-device-only -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon %s -o /dev/null
// DEFINE: %{use} = -mllvm -poseidon-profile-use=%S/../Inputs/unrelated_site_profile
// DEFINE: %{waved} = %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -DWAVED -O2 -ffp-contract=on -S -I%poseidon_profile_inc
// DEFINE: %{wuse} = %{waved} %loadClangPoseidon -mllvm -poseidon-profile-use=%t/prof -mllvm -poseidon-early-prune -mllvm -poseidon-tau=1e-3 \
// DEFINE:   -mllvm -poseidon-enable-rewrite=0 -mllvm -poseidon-cost-model=%gpu_cost_model %s
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
// RUN: %if enzyme && cuda-toolkit && nvptx-registered-target %{ rm -rf %t && mkdir -p %t/prof %}
// RUN: %if enzyme && cuda-toolkit && nvptx-registered-target %{ %{waved} -emit-llvm %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/gen %s -o %t/gen.ll %}
// RUN: %if enzyme && cuda-toolkit && nvptx-registered-target %{ sed -n 's/^@poseidon.static.preprocess__Z5wavedPdPKdi_poseidon_body = .*c"\(.*\)\\00".*/\1/p' %t/gen.ll | sed 's/\\0A/\n/g' | cat - %S/Inputs/device_math_macro/records > %t/prof/_Z5wavedPdPKdi.fpprofile %}
// RUN: %if enzyme && cuda-toolkit && nvptx-registered-target %{ %{wuse} -mllvm -poseidon-cache=%t/defined -mllvm -poseidon-print -o %t/defined.ptx 2>&1 | FileCheck --check-prefix=WAVED %s %}
// RUN: %if enzyme && cuda-toolkit && nvptx-registered-target %{ not %{wuse} -UPOSEIDON_DEVICE_MATH -mllvm -poseidon-cache=%t/undefined -o /dev/null 2>&1 | FileCheck -DLIB=libdevice %s %}

// A device module with a site and no POSEIDON_DEVICE_MATH, which keeps the
// device math (libdevice, ocml) a rewrite or a derivative calls, stops a
// profile action with an error naming the macro, whether the site is an
// annotation, a kernel -poseidon-kernels names or a marker call; a compile
// without a profile action leaves it as written. With the macro, a rewrite of
// waved to single precision (cheapest at 1e-3; profile: the generating
// compile's static header and hand-written records) links __nv_expf.

#include "poseidon/poseidon.h"

#ifdef WAVED
POSEIDON_OPTIMIZE __global__ void waved(double *o, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double x = a[i];
    o[i] = exp(-x) * x + x;
  }
}
#else
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
#endif

// CHECK: error: Poseidon: this device compile has Poseidon sites and does not include poseidon/poseidon.h under POSEIDON_DEVICE_MATH, which keeps the [[LIB]] functions a rewrite or a site's derivative calls; include the header and define POSEIDON_DEVICE_MATH{{$}}
// DEFINED-NOT: POSEIDON_DEVICE_MATH
// INACTIVE-NOT: POSEIDON_DEVICE_MATH
// WAVED: {{^}}[poseidon] pin: _Z5wavedPdPKdi:P0=All FP64(0%) + FP32(100%){{$}}
