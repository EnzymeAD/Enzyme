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
// site's: it is folded into the kernel and profiled there, and profile use
// looks the site up under the kernel's name. A recursive callee cannot be
// folded in, which is reported, and a site with no floating-point operation
// says so instead of being dropped silently, also when it is the module's
// only one.

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

// GEN: annotated_lambda_callee_nvptx.cu:[[@LINE-10]]:{{[0-9]+}}: warning: Poseidon: deep(double*, int) calls series(int, double), which the pass cannot fold into the site; the floating-point work it does is neither profiled nor rewritten
// GEN: annotated_lambda_callee_nvptx.cu:[[@LINE-6]]:{{[0-9]+}}: warning: Poseidon: copy(double*, double const*) holds no floating-point operation the pass profiles; the kernel is not profiled and stays as written
// GEN: [poseidon] block-geometry probe placed in 2 profiled function(s)
// GEN: [poseidon] _Z4deepPdi: site 0, 1 shadow buffer(s), output(s) 0
// GEN-NEXT: [poseidon] _ZL13forall_kernelIZ5applyiPKdPdEUliE_EviT_: site 1, 2 shadow buffer(s), output(s) 1[+8]
// GEN-NEXT: 2 warnings generated when compiling for {{.*}}

// The four operations of `element` (two of them fused multiply-adds, which
// log three operands), each logged by the profiled kernel's body, which calls
// `element` no more.
// PROBES: define internal fastcc void @diffepreprocess__ZL13forall_kernelIZ5applyiPKdPdEUliE_EviT__poseidon_body(
// PROBES-NOT: call {{.*}}@_Z7elementiPKdPd(
// PROBES: call void @poseidonLogValueCUDA(ptr addrspacecast (ptr addrspace(1) @poseidon_site_preprocess__ZL13forall_kernelIZ5applyiPKdPdEUliE_EviT__poseidon_body to ptr), i64 0, double %{{[0-9]+}}, i64 3,
// PROBES-NOT: call {{.*}}@_Z7elementiPKdPd(
// PROBES: call void @poseidonLogValueCUDA(ptr addrspacecast (ptr addrspace(1) @poseidon_site_preprocess__ZL13forall_kernelIZ5applyiPKdPdEUliE_EviT__poseidon_body to ptr), i64 1, double %{{[0-9]+}}, i64 2,
// PROBES-NOT: call {{.*}}@_Z7elementiPKdPd(
// PROBES: call void @poseidonLogValueCUDA(ptr addrspacecast (ptr addrspace(1) @poseidon_site_preprocess__ZL13forall_kernelIZ5applyiPKdPdEUliE_EviT__poseidon_body to ptr), i64 2, double %{{[0-9]+}}, i64 2,
// PROBES-NOT: call {{.*}}@_Z7elementiPKdPd(
// PROBES: call void @poseidonLogValueCUDA(ptr addrspacecast (ptr addrspace(1) @poseidon_site_preprocess__ZL13forall_kernelIZ5applyiPKdPdEUliE_EviT__poseidon_body to ptr), i64 3, double %{{[0-9]+}}, i64 3,
// PROBES-NOT: call {{.*}}@_Z7elementiPKdPd(
// PROBES-NOT: @poseidon_site_preprocess__ZL13forall_kernelIZ5applyiPKdPdEUliE_EviT__poseidon_body to ptr), i64 4,
// PROBES: {{^}}}
// PROBES-NOT: @poseidon_site_preprocess__ZL13forall_kernelIZ5applyiPKdPdEUliE_EviT__poseidon_body to ptr), i64 4,

// EMPTY: annotated_lambda_callee_nvptx.cu:[[@LINE-29]]:{{[0-9]+}}: warning: Poseidon: copy(double*, double const*) holds no floating-point operation the pass profiles; the kernel is not profiled and stays as written
// EMPTY: 1 warning generated when compiling for {{.*}}

// USE: annotated_lambda_callee_nvptx.cu:[[@LINE-37]]:{{[0-9]+}}: warning: Poseidon: deep(double*, int) calls series(int, double), which the pass cannot fold into the site; the floating-point work it does is neither profiled nor rewritten
// USE: annotated_lambda_callee_nvptx.cu:[[@LINE-38]]:{{[0-9]+}}: warning: Poseidon: no profile at {{.*}}/_Z4deepPdi.fpprofile:
// USE: annotated_lambda_callee_nvptx.cu:[[@LINE-34]]:{{[0-9]+}}: warning: Poseidon: _Z4copyPdPKd_poseidon_body holds no floating-point operation the pass rewrites; leaving the site unchanged
// USE: annotated_lambda_callee_nvptx.cu:[[@LINE-63]]:{{[0-9]+}}: warning: Poseidon: no profile at {{.*}}/_ZL13forall_kernelIZ5applyiPKdPdEUliE_EviT_~0735ed719e65441e.fpprofile:
// USE: 4 warnings generated when compiling for {{.*}}
