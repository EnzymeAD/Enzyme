// REQUIRES: enzyme, nvptx-device-profiler
// RUN: rm -rf %t && mkdir %t
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O3 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/cache -mllvm -poseidon-print %s -o %t/gen.ll 2> %t/gen.err
// RUN: FileCheck --check-prefix=GEN %s < %t/gen.err
// RUN: FileCheck --check-prefix=PROBES %s < %t/gen.ll
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O3 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/empty -DEMPTY_ONLY %s -o /dev/null 2>&1 | FileCheck --check-prefix=EMPTY %s
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O3 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites %s -o /dev/null 2>&1 | FileCheck --check-prefix=USE %s

// A site is everything its kernel runs on the device. The annotated lambda
// only calls `element`, which stays a call at -O3 (here because of noinline;
// in an application because of its size), and the arithmetic in it is the
// site's: it is folded into the kernel, profiled and solved. A recursive
// callee cannot be folded in, which is reported, and a site with no
// floating-point operation says so instead of being dropped silently, also
// when it is the module's only one.

#include "poseidon/poseidon.h"

#define HD __attribute__((host, device))

#ifndef EMPTY_ONLY
template <typename BODY>
__attribute__((global)) static void forall_kernel(int n, BODY body) {
  int i = __nvvm_read_ptx_sreg_ctaid_x() * __nvvm_read_ptx_sreg_ntid_x() +
          __nvvm_read_ptx_sreg_tid_x();
  if (i < n)
    body(i);
}

__attribute__((device, noinline)) void element(int i, const double *x,
                                               double *y) {
  double v = x[i];
  y[i] = (v * v + 3.0) * v + 1.0 / (v + 2.0);
}

__attribute__((device)) double series(int k, double v) {
  return k == 0 ? 1.0 : 1.0 + v * series(k - 1, v) / k;
}

void apply(int n, const double *x, double *y) {
  auto body = [=] HD POSEIDON_OPTIMIZE(int i) { element(i, x, y); };
  void *k = (void *)forall_kernel<decltype(body)>;
  (void)k;
}

POSEIDON_OPTIMIZE __attribute__((global)) void deep(double *y, int k) {
  y[0] = series(k, y[0]);
}
#endif

POSEIDON_OPTIMIZE __attribute__((global)) void copy(double *y,
                                                    const double *x) {
  y[__nvvm_read_ptx_sreg_tid_x()] = x[__nvvm_read_ptx_sreg_tid_x()];
}

// GEN-DAG: warning: {{.*}}Poseidon: deep(double*, int) calls series(int, double), which the pass cannot fold into the site; the floating-point work it does is neither profiled nor rewritten
// GEN-DAG: warning: {{.*}}Poseidon: copy(double*, double const*) holds no floating-point operation the pass profiles; the kernel is not profiled and stays as written
// GEN-DAG: [poseidon-profgen] _ZL13forall_kernelIZ5applyiPKdPdEUliE_EviT_: site {{[0-9]+}}, 2 shadow buffer(s), output(s) 1[+8]

// The four operations of `element` (two of them fused multiply-adds), each
// logged by the profiled kernel.
// PROBES-COUNT-4: call void @poseidonLogValueCUDA({{.*}}@poseidon_site_preprocess__ZL13forall_kernelIZ5applyiPKdPdEUliE_EviT__poseidon_body to ptr), i64 {{[0-3]}},
// PROBES-NOT: @poseidon_site_preprocess__ZL13forall_kernelIZ5applyiPKdPdEUliE_EviT__poseidon_body to ptr), i64 4,

// EMPTY: warning: {{.*}}Poseidon: copy(double*, double const*) holds no floating-point operation the pass profiles

// USE: warning: {{.*}}Poseidon: _Z4copyPdPKd_poseidon_body holds no floating-point operation the pass rewrites; leaving the site unchanged
