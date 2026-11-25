// REQUIRES: enzyme, cuda-runtime, nvptx-registered-target
// DEFINE: %{use} = %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch \
// DEFINE:   -O2 -ffp-contract=on -S -I%poseidon_profile_inc %loadClangPoseidon \
// DEFINE:   -mllvm -poseidon-profile-use=%t/prof -mllvm -poseidon-tau=1e-3 \
// DEFINE:   -mllvm -poseidon-enable-rewrite=0 \
// DEFINE:   -mllvm -poseidon-cost-model=%gpu_cost_model %s
// RUN: rm -rf %t && mkdir -p %t/prof
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -O2 -ffp-contract=on -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/gen %s -o %t/gen.ll
// RUN: sed -n 's/^@poseidon.static.preprocess__Z5wavedPdPKdi_poseidon_body = .*c"\(.*\)\\00".*/\1/p' %t/gen.ll | sed 's/\\0A/\n/g' | cat - %S/Inputs/device_math_macro/records > %t/prof/_Z5wavedPdPKdi.fpprofile
// RUN: %{use} -mllvm -poseidon-cache=%t/defined -mllvm -poseidon-print -o %t/defined.ptx 2>&1 | FileCheck --check-prefix=DEFINED %s
// RUN: not %{use} -UPOSEIDON_DEVICE_MATH -mllvm -poseidon-cache=%t/undefined -o /dev/null 2>&1 | FileCheck --check-prefix=UNDEFINED %s

// Single precision is cheapest at 1e-3, and the kernel calls no libdevice
// float function. poseidon.h links __nv_expf for the rewrite only under
// POSEIDON_DEVICE_MATH: with it the rewrite compiles, and without it the
// compile stops at its start with an error that names the macro. The profile is the static
// header of the generating compile followed by hand-written records.

#include "poseidon/poseidon.h"

POSEIDON_OPTIMIZE __global__ void waved(double *o, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double x = a[i];
    o[i] = exp(-x) * x + x;
  }
}

// DEFINED: {{^}}[poseidon] pin: _Z5wavedPdPKdi:P0=All FP64(0%) + FP32(100%){{$}}

// UNDEFINED: error: Poseidon: this device compilation has Poseidon sites and does not include poseidon/poseidon.h under POSEIDON_DEVICE_MATH, under which the header keeps the libdevice functions a rewrite or a site's derivative calls; include the header and define the macro (-DPOSEIDON_DEVICE_MATH), as poseidon-clang and the Poseidon CMake targets do{{$}}
