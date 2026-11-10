// REQUIRES: enzyme, amdgpu-registered-target, rocm
// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -ffp-contract=on -S -DAS_WRITTEN -DNO_SITE %s -o %t/as_written.s
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -ffp-contract=on -S -I%poseidon_profile_inc %loadClangPoseidon %s -o %t/plain.s
// RUN: diff %t/as_written.s %t/plain.s
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -ffp-contract=on -S -I%poseidon_profile_inc %loadClangPoseidonEnzyme %s -o %t/plain.enzyme.s
// RUN: diff %t/as_written.s %t/plain.enzyme.s
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -ffp-contract=on -S -DNO_SITE -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/Inputs/other_sites %s -o %t/use.nosite.s
// RUN: diff %t/as_written.s %t/use.nosite.s
// RUN: %clang -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -ffp-contract=on -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/cache %s -o %t/gen.ll
// RUN: FileCheck --check-prefix=CALLS %s < %t/gen.ll
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -ffp-contract=on -S -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/Inputs/other_sites %s -o %t/use.s
// RUN: diff %t/as_written.s %t/use.s
// RUN: %clang -x hip --offload-arch=gfx90a --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -ffp-contract=on -S -DAS_WRITTEN -DNO_SITE %s -o %t/as_written.90a.s
// RUN: %clang -x hip --offload-arch=gfx90a --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -ffp-contract=on -S -I%poseidon_profile_inc %loadClangPoseidon %s -o %t/plain.90a.s
// RUN: diff %t/as_written.90a.s %t/plain.90a.s
// RUN: %clang -x hip --offload-arch=gfx90a --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -ffp-contract=on -S -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/Inputs/other_sites %s -o %t/use.90a.s
// RUN: diff %t/as_written.90a.s %t/use.90a.s

// device_math_plugins.cu for HIP: the ocml math of a site stays a call, which
// Enzyme differentiates by name, under profile generation, and nothing the
// plugins or poseidon.h add for it is left in a compile that profiles or
// rewrites nothing (gfx942, and gfx90a).

#include <hip/hip_runtime.h>

#ifdef AS_WRITTEN
#define POSEIDON_OPTIMIZE __attribute__((annotate("poseidon")))
#else
#include "poseidon/poseidon.h"
#endif
#ifdef NO_SITE
#undef POSEIDON_OPTIMIZE
#define POSEIDON_OPTIMIZE
#endif

POSEIDON_OPTIMIZE __global__ void wave(double *o, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double x = a[i];
    o[i] = sqrt(x) * exp(-x) + (1.0 - cos(x)) / (x * x);
  }
}

// HIP writes exp(double) as llvm.exp.f64, which Enzyme takes to the ocml
// function that implements it once poseidon.h has linked that.
// CALLS-DAG: call {{.*}}double @__ocml_cos_f64(
// CALLS-DAG: call {{.*}}double @__ocml_exp_f64(
// CALLS-DAG: call {{.*}}double @__ocml_sin_f64(
