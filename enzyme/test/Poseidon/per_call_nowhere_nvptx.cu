// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O1 -S -emit-llvm -I%FPProfileInc %clangLoadPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c %s -o %t/gen.ll 2>&1 | FileCheck %s
// RUN: FileCheck --check-prefix=IR %s < %t/gen.ll
// REQUIRES: poseidon, enzyme

// The address of an empty functor is a pointer the call's copy never reads, so
// the optimizer passes poison in its place by the time the site is lowered. It
// points into no object, as a null pointer does, and is passed as a constant.

#include "poseidon/poseidon.h"

struct square {
  __attribute__((device)) void operator()(const double &x, double &y) const {
    y = x * x + 1.0;
  }
};

template <typename f_t>
__attribute__((device)) void apply(const f_t *f, const double *x, double *y) {
  (*f)(*x, *y);
}

__attribute__((global)) void k(double *x) {
  const square f{};
  double a = x[0], b;
  __poseidon_optimize(apply<square>, &f, &a, &b);
  x[1] = 2.0 * b;
}

// CHECK-NOT: warning
// CHECK: [poseidon-profgen] void apply<square>(square const*, double const*, double*) at call 0 in k(double*): 2 shadow object(s), output(s) 2
// CHECK-NOT: warning
// IR-LABEL: define {{.*}} @_Z1kPd(
// IR: call {{.*}} @diffe{{.*}}_Z5applyI6squareEvPKT_PKdPd.poseidon._Z1kPd.0(
// IR: ret void
