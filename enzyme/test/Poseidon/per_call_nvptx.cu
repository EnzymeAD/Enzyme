// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O1 -S -emit-llvm -I%FPProfileInc %clangLoadPoseidon -mllvm -poseidon-profile-use=%t/none %s -o %t/k.ll 2>&1 | FileCheck %s
// RUN: FileCheck --check-prefix=IR %s < %t/k.ll
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O1 -S -emit-llvm -I%FPProfileInc %clangLoadPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c %s -o %t/gen.ll 2>&1 | FileCheck --check-prefix=GEN %s
// RUN: FileCheck --check-prefix=GENIR %s < %t/gen.ll
// REQUIRES: poseidon, enzyme

// In device code a per-call site is a site as on the host. The pass makes the
// shadow of every local object a call passes, zero, with the seed on what the
// call writes; a pointer into device memory leaves the call as written, and an
// object holding pointers is passed as a constant.

#include "poseidon/poseidon.h"

struct holder {
  const double *c;
};

__attribute__((device)) double sq(double x) { return x * x + 1.0; }

__attribute__((device)) void scale(const double *x, double *y) {
  y[0] = x[0] * x[0] + x[1];
  y[1] = x[1] * x[1];
}

__attribute__((device)) void weigh(const holder &h, const double *x,
                                   double *y) {
  y[0] = h.c[0] * x[0] * x[0];
}

__attribute__((global)) void k(double *x) {
  x[0] = __poseidon_optimize(sq, x[1]);
  double a[2] = {x[2], x[3]}, b[2];
  __poseidon_optimize(scale, a, b);
  x[4] = b[0] + b[1];
  __poseidon_optimize(scale, x + 5, x + 7);
  holder h{x + 9};
  double c, d = x[10];
  __poseidon_optimize(weigh, h, &d, &c);
  x[11] = c;
}

// CHECK-DAG: [poseidon] _Z2sqd.poseidon._Z1kPd.0: no profile at {{.*}}; left unchanged
// CHECK-DAG: [poseidon] _Z5scalePKdPd.poseidon._Z1kPd.1: no profile at {{.*}}; left unchanged
// CHECK-DAG: [poseidon] _Z5weighRK6holderPKdPd.poseidon._Z1kPd.3: no profile at {{.*}}; left unchanged
// CHECK-DAG: warning: {{.*}}Poseidon: scale(double const*, double*) at call 2 in k(double*) takes argument 1 as a pointer into memory that is not a local object of k(double*), so this call is not a Poseidon site: the pass makes shadows of local objects only, and __poseidon_fp_optimize takes the caller's own. The call runs as written.
// IR-LABEL: define {{.*}} @_Z1kPd(
// IR-NOT: call
// IR: ret void
// IR-NOT: __poseidon_fp_optimize
// IR-NOT: __poseidon_site

// GEN-DAG: [poseidon-profgen] scale(double const*, double*) at call 1 in k(double*): 2 shadow object(s), output(s) 1
// GEN-DAG: [poseidon-profgen] weigh(holder const&, double const*, double*) at call 3 in k(double*): 2 shadow object(s), output(s) 2
// GENIR-LABEL: define {{.*}} @_Z1kPd(
// GENIR: call void @llvm.memset
// GENIR: store double 1.000000e+00
// GENIR: call {{.*}} @diffe{{.*}}_Z5scalePKdPd.poseidon._Z1kPd.1(ptr {{%[0-9]+}}, ptr {{%[0-9]+}}, ptr {{%[0-9]+}}, ptr {{%[0-9]+}})
// GENIR-NOT: _Z5scalePKdPd.poseidon._Z1kPd.2(
// The holder has no shadow; d and c do.
// GENIR: call {{.*}} @diffe{{.*}}_Z5weighRK6holderPKdPd.poseidon._Z1kPd.3(ptr {{%[0-9]+}}, ptr {{%[0-9]+}}, ptr {{%[0-9]+}}, ptr {{%[0-9]+}}, ptr {{%[0-9]+}})
// GENIR: ret void
