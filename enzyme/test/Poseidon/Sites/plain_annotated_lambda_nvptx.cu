// REQUIRES: cuda-runtime
// From LLVM 22 an sm_120 compile needs the PTX version of a CUDA installation.
// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc -DUNANNOTATED %s -o %t/unannotated.ll
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon %s -o %t/plain.ll 2>&1 | FileCheck --check-prefix=WARN %s
// RUN: diff %t/unannotated.ll %t/plain.ll
// RUN: FileCheck %s < %t/plain.ll

// Without a profile action an annotated lambda compiles as if it were not
// annotated: its kernels are not made to inline it, and the annotation does
// not count as a second use of it, which would cost a static lambda the
// inliner's last-call bonus.

#include "poseidon/poseidon.h"

#ifdef UNANNOTATED
#define SITE
#else
#define SITE POSEIDON_OPTIMIZE
#endif
#define HD __attribute__((host, device))

template <typename BODY>
__attribute__((global)) static void forall(int n, BODY body) {
  int i = __nvvm_read_ptx_sreg_ctaid_x() * __nvvm_read_ptx_sreg_ntid_x() +
          __nvvm_read_ptx_sreg_tid_x();
  if (i < n)
    body(i);
}

#define CHAIN(a, b, c)                                                         \
  _Pragma("unroll") for (int k = 0; k < 96; ++k) {                             \
    a = a * b + c[k];                                                          \
    b = b * a - c[k + 96];                                                     \
  }

// Too large to inline even with the bonus: the kernel calls it.
template <int N> void large(int n, const double *c, double *x, double *y) {
  auto body = [=] HD(int i) SITE {
    double a = x[i], b = y[i];
    CHAIN(a, b, c)
    CHAIN(b, a, c)
    CHAIN(a, b, c)
    y[i] = a + b * N;
  };
  (void)(void *)forall<decltype(body)>;
}
template void large<1>(int, const double *, double *, double *);

// A static lambda inlined only with the bonus.
void small(int n, const double *c, double *x, double *y) {
  auto body = [=] HD(int i) SITE {
    double a = x[i], b = y[i];
    CHAIN(a, b, c)
    CHAIN(b, a, c)
    y[i] = a + b;
  };
  (void)(void *)forall<decltype(body)>;
}

// WARN-COUNT-2: warning: {{.*}}compiled without -poseidon-profile-generate or -poseidon-profile-use=<dir>

// CHECK-LABEL: define {{.*}} @_ZL6forallIZ5largeILi1EEviPKdPdS3_EUliE_EviT_(
// CHECK: call {{.*}} @_ZZ5largeILi1EEviPKdPdS2_ENKUliE_clEi(
// CHECK-LABEL: define {{.*}} @_ZL6forallIZ5smalliPKdPdS2_EUliE_EviT_(
// CHECK-NOT: call {{.*}}@_ZZ5small
// CHECK: ret void
// CHECK-NOT: define {{.*}}@_ZZ5small
