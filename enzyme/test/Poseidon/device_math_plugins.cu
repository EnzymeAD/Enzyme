// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -O2 -ffp-contract=on -S -DAS_WRITTEN -DNO_SITE %s -o %t/as_written.ptx
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -O2 -ffp-contract=on -S -I%FPProfileInc %clangLoadPoseidon %s -o %t/plain.ptx
// RUN: diff %t/as_written.ptx %t/plain.ptx
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -O2 -ffp-contract=on -S -I%FPProfileInc %clangLoadPoseidonEnzyme %s -o %t/plain.enzyme.ptx
// RUN: diff %t/as_written.ptx %t/plain.enzyme.ptx
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -O2 -ffp-contract=on -S -DNO_SITE -I%FPProfileInc %clangLoadPoseidon -mllvm -poseidon-profile-use=%t/none %s -o %t/use.nosite.ptx
// RUN: diff %t/as_written.ptx %t/use.nosite.ptx
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -O2 -ffp-contract=on -S -emit-llvm -I%FPProfileInc %clangLoadPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/cache %s -o %t/gen.ll
// RUN: FileCheck --check-prefix=CALLS %s < %t/gen.ll
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -O2 -ffp-contract=on -S -I%FPProfileInc %clangLoadPoseidon -mllvm -poseidon-profile-use=%t/none %s -o %t/use.ptx
// RUN: diff %t/as_written.ptx %t/use.ptx
// REQUIRES: poseidon, enzyme, cuda-runtime

// Enzyme's plugin is loaded to generate a profile and not to use one, so the
// form a site is profiled in must not depend on it: the libdevice math stays a
// call, which Enzyme differentiates by name, in both. Anything the plugins do
// for that, the libdevice functions poseidon.h links for a rewrite to call,
// and the annotation itself are gone from a compile that profiles or rewrites
// nothing.

#ifdef AS_WRITTEN
#define POSEIDON_OPTIMIZE __attribute__((annotate("poseidon")))
#else
#include "poseidon/poseidon.h"
#endif
#ifdef NO_SITE
#undef POSEIDON_OPTIMIZE
#define POSEIDON_OPTIMIZE
#endif

POSEIDON_OPTIMIZE __global__ void wave(float *o, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = a[i];
    o[i] = sqrtf(x) * expf(-x) + (1.0f - cosf(x)) / (x * x);
  }
}

// CALLS-DAG: call {{.*}}float @__nv_expf(
// CALLS-DAG: call {{.*}}float @__nv_cosf(
