// REQUIRES: enzyme
// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/cache %s -o %t/gen.ll 2>&1 | FileCheck --check-prefix=GEN %s
// RUN: cat %t/cache/launches_*.json | FileCheck --check-prefix=DESC %s
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites -mllvm -poseidon-print %s -o %t/use.ll 2>&1 | FileCheck --check-prefix=USE %s

// POSEIDON_OPTIMIZE on the lambda an mfem::forall body is written as makes
// every kernel that calls it a site, with the lambda's target. The lambda
// reaches the kernel as one by-value aggregate of DeviceTensors (MFEM's
// layout: capacity, data, sizes), so each data pointer is a field of the
// kernel's parameter: profile generation shadows each of them and seeds the
// one written.

#include "poseidon/poseidon.h"

#define HD __attribute__((host, device))
#define FOREACH_Y(i, N)                                                        \
  for (int i = __nvvm_read_ptx_sreg_tid_y(); i < N;                            \
       i += __nvvm_read_ptx_sreg_ntid_y())

template <int D> struct DeviceTensor {
  int capacity;
  double *data;
  int sizes[D];
  HD double &operator()(int a, int b) const { return data[a + sizes[0] * b]; }
};

template <typename BODY> __attribute__((global)) static void CuKernel2D(int N, BODY body) {
  int k = __nvvm_read_ptx_sreg_ctaid_x() * __nvvm_read_ptx_sreg_ntid_z() +
          __nvvm_read_ptx_sreg_tid_z();
  if (k >= N)
    return;
  body(k);
}

// A kernel that computes more than the lambda is not its site.
template <typename BODY> __attribute__((global)) static void Scaled(int N, double *s, BODY body) {
  int k = __nvvm_read_ptx_sreg_ctaid_x();
  if (k >= N)
    return;
  s[k] = s[k] * 2.0;
  body(k);
}

void reduce(int nq, int ne, int nd, double *Q, double *G, double *y) {
  DeviceTensor<2> QR{nq * ne, Q, {nq, ne}};
  DeviceTensor<2> GR{nq * nd, G, {nq, nd}};
  DeviceTensor<2> Y{nd * ne, y, {nd, ne}};
  auto body = [=] HD POSEIDON_OPTIMIZE_TAU(1e-9) (int e) {
    FOREACH_Y(i, nd) {
      double sum = 0.0;
      for (int p = 0; p < nq; ++p)
        sum += QR(p, e) * GR(p, i);
      Y(i, e) += sum;
    }
  };
  void *k1 = (void *)CuKernel2D<decltype(body)>;
  void *k2 = (void *)Scaled<decltype(body)>;
  (void)k1;
  (void)k2;
}

// An annotated device function no kernel calls is not a site.
POSEIDON_OPTIMIZE __attribute__((device)) double unused(double x) { return x * x; }

// GEN: warning: {{.*}}Poseidon: void Scaled<{{.*}}>(int, double*, {{.*}}) calls reduce({{.*}})::'lambda'(int)::operator()(int) const, which carry POSEIDON_OPTIMIZE, and does floating-point work outside them (in void Scaled<{{.*}}); a kernel is the site of an annotated function only when it is a wrapper around the call. Emitting the original computation.
// GEN: warning: {{.*}}Poseidon: unused(double) is a device function no kernel of this translation unit calls; on the device, POSEIDON_OPTIMIZE marks a kernel or a function a kernel calls. Emitting the original computation.
// GEN: [poseidon-profgen] _ZL10CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT_: site 0, 3 shadow buffer(s), output(s) 1[+64]
// GEN-NOT: [poseidon-profgen] {{.*}}Scaled

// The kernel, its two launch arguments, its three pointers (all in argument
// 1), the eight-byte seed of the one written, and where each sits in the
// lambda.
// DESC: _ZL10CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT_ 0 2 3 1 1 1 0 0 8 off 16 40 64

// USE: [poseidon] site void CuKernel2D<reduce({{.*}})::'lambda'(int)>(int, reduce({{.*}})::'lambda'(int)): accuracy target 1e-09
// USE: no profile at {{.*}}other_sites/preprocess__ZL10CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT__poseidon_body.fpprofile: the profiled program did not contain this site
