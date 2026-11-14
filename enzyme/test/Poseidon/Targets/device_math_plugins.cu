// REQUIRES: enzyme, cuda-runtime
// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -O2 -ffp-contract=on -S -DAS_WRITTEN -DNO_SITE %s -o %t/as_written.ptx
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -O2 -ffp-contract=on -S -I%poseidon_profile_inc %loadClangPoseidon %s -o %t/plain.ptx
// RUN: diff %t/as_written.ptx %t/plain.ptx
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -O2 -ffp-contract=on -S -DNO_SITE -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites %s -o %t/use.nosite.ptx
// RUN: diff %t/as_written.ptx %t/use.nosite.ptx
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -O2 -ffp-contract=on -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/cache %s -o %t/gen.ll
// RUN: FileCheck --check-prefix=CALLS %s < %t/gen.ll
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -O2 -ffp-contract=on -S -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites %s -o %t/use.ptx
// RUN: diff %t/as_written.ptx %t/use.ptx
// RUN: mkdir -p %t/prof
// RUN: sed -n 's/^@fpprofile_static_preprocess__Z4wavePfPKfi_poseidon_body = .*c"\(.*\)\\00".*/\1Executed = 0\\0A/p' %t/gen.ll | sed 's/\\0A/\n/g' > %t/prof/preprocess__Z4wavePfPKfi_poseidon_body.fpprofile
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -O2 -ffp-contract=on -S -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%t/prof -mllvm -poseidon-print %s -o %t/use.profiled.ptx 2>&1 | FileCheck --check-prefix=PROFILED %s
// RUN: diff %t/as_written.ptx %t/use.profiled.ptx

// Enzyme's plugin is loaded to generate a profile and not to use one, so the
// form a site is profiled in must not depend on it: the libdevice math stays a
// call, which Enzyme differentiates by name, and the profile the run would
// write for a site it never executed (the static header the generating compile
// embeds) is accepted by a use compile with Poseidon's plugin alone. What
// Poseidon's plugin does for that, the libdevice functions poseidon.h links
// for a rewrite to call, and the annotation itself are gone from a compile
// that profiles or rewrites nothing.

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
    o[i] = sqrtf(x) * expf(-x) + (1.0f - cosf(x)) / (x * x) +
           powf(x, 1.5f) * sinf(x);
  }
}

// CALLS-DAG: call {{.*}}float @__nv_expf(
// CALLS-DAG: call {{.*}}float @__nv_cosf(
// CALLS-DAG: call {{.*}}float @__nv_powf(
// CALLS-DAG: call {{.*}}float @__nv_sinf(

// PROFILED: preprocess__Z4wavePfPKfi_poseidon_body.fpprofile); left unchanged
