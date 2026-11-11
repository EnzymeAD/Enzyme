// REQUIRES: enzyme, amdgpu-registered-target, rocm
// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-host-only -cuid=0 -O2 -S -emit-llvm -I%poseidon_profile_inc %s -o %t/plain.ll
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-host-only -cuid=0 -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/Inputs/other_sites -mllvm -poseidon-cache=%t/u %s -o %t/use.ll 2>&1 | FileCheck --allow-empty --check-prefix=HOST %s
// RUN: diff %t/plain.ll %t/use.ll
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-host-only -cuid=0 -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/g %s -o %t/gen.ll 2>&1 | FileCheck --check-prefix=GEN %s
// RUN: FileCheck --check-prefix=GENIR %s < %t/gen.ll
// RUN: %clang -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/d %s -o %t/dev.ll 2>&1 | FileCheck --check-prefix=DEV %s

// per_call_host_half_nvptx.cu for HIP: the host half of a HIP compilation
// holds no site of the call or the annotation form.

#include <hip/hip_runtime.h>

#include "poseidon/poseidon.h"

#define HD __host__ __device__

HD void vol(const double *a, double *v) {
  double J = a[0] * a[1] - a[2] * a[3];
  *v = J * (J - 1.0);
}

HD double qpoint(const double *g) {
  double a[4] = {g[0], g[1], g[2], g[3]}, v;
  __poseidon_optimize(vol, a, &v);
  return 2.0 * v;
}

POSEIDON_OPTIMIZE HD double area(double r) { return 3.14159 * r * r; }

double host_only(const double *g) {
  double a[4] = {g[3], g[2], g[1], g[0]}, v;
  __poseidon_optimize(vol, a, &v);
  return v;
}

__global__ void k(const double *g, double *r, int n) {
  int i = threadIdx.x;
  if (i < n)
    r[i] = qpoint(g + 4 * i);
}

int main() {
  const int n = 64;
  static double g[4 * n], r[n];
  for (int i = 0; i < 4 * n; ++i)
    g[i] = 1.0 + 1e-3 * (i % 7);
  for (int i = 0; i < n; ++i)
    r[i] = qpoint(g + 4 * i) + area(g[i]) + host_only(g + 4 * i);
  return r[7] > 0.0;
}

// HOST-NOT: {{.}}

// GEN-NOT: warning
// GEN: [poseidon] FP profiler CUDA registration injected at main
// GEN-NOT: {{.}}

// GENIR-NOT: call {{.*}}@poseidonLogValue(
// GENIR-NOT: call {{.*}}@poseidonLogGrad(
// GENIR-NOT: __poseidon_host_shadows
// GENIR-NOT: poseidonRegisterProfileStatic
// GENIR-NOT: preprocess_

// DEV: [poseidon-profgen] vol(double const*, double*) at call 0 in qpoint(double const*): 2 shadow object(s), output(s) 1
