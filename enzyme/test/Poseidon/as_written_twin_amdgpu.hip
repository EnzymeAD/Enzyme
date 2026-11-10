// REQUIRES: amdgpu-registered-target, rocm
// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -ffp-contract=on -S -I%poseidon_profile_inc %loadClangPoseidon %s -o %t/plain.s
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -ffp-contract=on -S -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/Inputs/other_sites -mllvm -poseidon-print %s -o %t/use.s 2> %t/use.err
// RUN: FileCheck --check-prefix=SOLVE %s < %t/use.err
// RUN: diff %t/plain.s %t/use.s
// RUN: FileCheck %s < %t/use.s
// RUN: %clang -x hip --offload-arch=gfx90a --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -ffp-contract=on -S -I%poseidon_profile_inc %loadClangPoseidon %s -o %t/plain.90a.s
// RUN: %clang -x hip --offload-arch=gfx90a --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -ffp-contract=on -S -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/Inputs/other_sites -mllvm -poseidon-print %s -o %t/use.90a.s 2> %t/use.90a.err
// RUN: FileCheck --check-prefix=SOLVE %s < %t/use.90a.err
// RUN: diff %t/plain.90a.s %t/use.90a.s
// RUN: FileCheck %s < %t/use.90a.s

// An AMDGPU device module with a site is compiled once more as written, and
// every function no rewrite reached is emitted from that compile: here no site
// has a profile, so the whole module is the one as written, although the solve
// kept ocml's sine out of line in it. The same for gfx942 and gfx90a.

#include <hip/hip_runtime.h>

#include "poseidon/poseidon.h"

POSEIDON_OPTIMIZE_TAU(1e-6)
__global__ void site(const double *x, double *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double a = x[i];
    y[i] = (3.0 * a + 0.5) * a + sin(a);
  }
}

__global__ void other(const double *x, double *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = sin(x[i]) + y[i];
}

// SOLVE: [poseidon] _Z4sitePKdPdi_poseidon_body: the profiled program did not contain this site (no profile at {{.*}}); left unchanged
// SOLVE: [poseidon] every site of {{.*}} was left alone; emitting the module as written

// CHECK-LABEL: _Z4sitePKdPdi:
// CHECK-NOT: s_swappc
// CHECK: s_endpgm
// CHECK-LABEL: _Z5otherPKdPdi:
// CHECK-NOT: s_swappc
// CHECK: s_endpgm
