// REQUIRES: enzyme
// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -x cuda --cuda-host-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc %s -o %t/plain.ll
// RUN: %clang -x cuda --cuda-host-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites -mllvm -poseidon-cache=%t/u %s -o %t/use.ll 2>&1 | FileCheck --allow-empty --check-prefix=HOST %s
// RUN: diff %t/plain.ll %t/use.ll
// RUN: %clang -x cuda --cuda-host-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/g %s -o %t/gen.ll 2>&1 | FileCheck --check-prefix=GEN %s
// RUN: FileCheck --check-prefix=GENIR %s < %t/gen.ll
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/d %s -o %t/dev.ll 2>&1 | FileCheck --check-prefix=DEV %s

// In a CUDA or HIP compilation only the device half holds sites: one
// compilation has one device and one cost model. The host half of a
// host-device function that holds the call form, or carries the annotation
// form, compiles as written, silently, in profile generation and in profile
// use, and so does a host function: neither becomes a host site, and no
// record of the host half shares a name with the device half's.

#include "poseidon/poseidon.h"

#define HD __attribute__((host, device))

struct dim3 {
  unsigned x, y, z;
};
typedef struct CUstream_st *cudaStream_t;
extern "C" int cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                                void **args, unsigned long sharedMem,
                                cudaStream_t stream);

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

__attribute__((global)) void k(const double *g, double *r, int n) {
  int i = __nvvm_read_ptx_sreg_tid_x();
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

// Main binds the profiler, as in any unit of a profiled program.
// GEN-NOT: warning
// GEN: [poseidon] FP profiler CUDA registration injected at main
// GEN-NOT: {{.}}

// GENIR-NOT: call {{.*}}@poseidonLogValue(
// GENIR-NOT: call {{.*}}@poseidonLogGrad(
// GENIR-NOT: __poseidon_host_shadows
// GENIR-NOT: poseidonRegisterProfileStatic
// GENIR-NOT: preprocess_

// DEV: [poseidon-profgen] vol(double const*, double*) at call 0 in qpoint(double const*): 2 shadow object(s), output(s) 1
