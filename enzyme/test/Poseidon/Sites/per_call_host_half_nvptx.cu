// REQUIRES: enzyme, cuda-runtime
// Without a CUDA installation clang emits the old cudaLaunch API in the host half.
// DEFINE: %{clang} = %clang -x cuda --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 \
// DEFINE:   -S -emit-llvm -I%poseidon_profile_inc %s
// DEFINE: %{body} = sed -e '/__poseidon_profile_init_cuda/d' -e '/^; Function Attrs/d' \
// DEFINE:   -e '/^attributes #/d' -e '/^$/d' -e 's/ #[0-9][0-9]*//g'
// RUN: rm -rf %t && mkdir %t
// RUN: %{clang} --cuda-host-only -o %t/plain.ll
// RUN: %{clang} --cuda-host-only %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites -mllvm -poseidon-cache=%t/u -o %t/use.ll 2>&1 \
// RUN:   | FileCheck --allow-empty --check-prefix=HOST %s
// RUN: diff %t/plain.ll %t/use.ll
// RUN: %{clang} --cuda-host-only %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/g -mllvm -poseidon-print -o %t/gen.ll 2>&1 \
// RUN:   | FileCheck --check-prefix=GEN %s
// RUN: FileCheck --check-prefix=GENIR %s < %t/gen.ll
// RUN: FileCheck --check-prefix=BIND %s < %t/gen.ll
// RUN: %{body} %t/plain.ll > %t/plain.body
// RUN: %{body} %t/gen.ll > %t/gen.body
// RUN: diff %t/plain.body %t/gen.body
// RUN: %{clang} --cuda-device-only %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/d -mllvm -poseidon-print -o %t/dev.ll 2>&1 \
// RUN:   | FileCheck --check-prefix=DEV %s

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

// In profile generation main initializes the profiler, and the rest of the
// host half is the plain compile's IR up to attribute groups.
// BIND: define {{.*}}i32 @main()
// BIND-NEXT: call void @__poseidon_profile_init_cuda()
// BIND: declare void @__poseidon_profile_init_cuda()

// The call in qpoint is lowered twice: in qpoint and where k inlines it.
// DEV: per_call_host_half_nvptx.cu:[[@LINE-44]]:{{[0-9]+}}: warning: Poseidon: area(double) is a device function no kernel of this translation unit calls, so it is left as written; on the device, put POSEIDON_OPTIMIZE on a kernel or on a function a kernel calls
// DEV: [poseidon] vol(double const*, double*) at call 0 in qpoint(double const*): 2 shadow object(s), output(s) 1
// DEV-NEXT: [poseidon] vol(double const*, double*) at call 0 in qpoint(double const*): 2 shadow object(s), output(s) 1
// DEV-NEXT: [poseidon] block-geometry probe placed in 1 profiled function(s)
// DEV-NOT: {{shadow (buffer|object)\(s\)}}
