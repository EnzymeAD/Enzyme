// REQUIRES: amdgpu-registered-target, rocm
// DEFINE: %{clang} = %clang -x hip --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 \
// DEFINE:   -ffp-contract=on -S -I%poseidon_profile_inc %loadClangPoseidon %s
// RUN: rm -rf %t && mkdir %t
// RUN: %{clang} --offload-arch=gfx942 -o %t/plain.s
// RUN: %{clang} --offload-arch=gfx942 -mllvm -poseidon-profile-use=%S/../Inputs/other_sites -mllvm -poseidon-print -o %t/use.s 2>&1 | FileCheck --check-prefix=SOLVE %s
// RUN: diff %t/plain.s %t/use.s
// RUN: FileCheck %s < %t/use.s
// RUN: %{clang} --offload-arch=gfx90a -o %t/plain.90a.s
// RUN: %{clang} --offload-arch=gfx90a -mllvm -poseidon-profile-use=%S/../Inputs/other_sites -mllvm -poseidon-print -o %t/use.90a.s 2>&1 | FileCheck --check-prefix=SOLVE %s
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

// SOLVE: [poseidon] site site(double const*, double*, int): accuracy target 1e-06
// SOLVE-NEXT: as_written_twin_amdgpu.hip:[[@LINE-15]]:{{[0-9]+}}: warning: Poseidon: no profile at {{.*}}/Inputs/other_sites/preprocess__Z4sitePKdPdi_poseidon_body.fpprofile: the profiled program did not contain this site (it was annotated after the profiling run, is in a translation unit the profiled program did not link, or was renamed); leaving the site unchanged
// SOLVE: [poseidon] _Z4sitePKdPdi_poseidon_body: the profiled program did not contain this site (no profile at {{.*}}/Inputs/other_sites/preprocess__Z4sitePKdPdi_poseidon_body.fpprofile); left unchanged
// SOLVE-NEXT: [poseidon] every site of {{.*}}as_written_twin_amdgpu.hip was left alone; emitting the module as written

// Both kernels compute ocml's sine inline and call nothing.
// CHECK-LABEL: _Z4sitePKdPdi:
// CHECK-NOT: s_swappc
// CHECK: v_trig_preop_f64
// CHECK-NOT: s_swappc
// CHECK: s_endpgm
// CHECK-LABEL: _Z5otherPKdPdi:
// CHECK-NOT: s_swappc
// CHECK: v_trig_preop_f64
// CHECK-NOT: s_swappc
// CHECK: s_endpgm
