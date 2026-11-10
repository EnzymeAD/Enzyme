// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O1 -S -emit-llvm -I%FPProfileInc %clangLoadPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c %s -o %t/gen.ll 2>&1 | FileCheck %s
// RUN: FileCheck --check-prefix=IR %s < %t/gen.ll
// REQUIRES: poseidon, enzyme

// The outputs of a call site are what the program reads after the call: the
// pass seeds the shadow of an object the call writes only when an instruction
// after the call, or the call itself when it runs again, may read the object
// before its lifetime ends. p is written and never read, t is read after the
// call, and s is read by the next call only.

#include "poseidon/poseidon.h"

__attribute__((device)) void both(const double *x, double *p, double *t) {
  p[0] = x[0] * x[0];
  t[0] = 2.0 * x[0] * x[1];
}

__attribute__((device)) void step(double *s, double *y) {
  y[0] = s[0] * s[0];
  s[0] = s[0] + 0.5 * y[0];
}

__attribute__((global)) void k(double *x, int n) {
  for (int i = 0; i < n; ++i) {
    double in[2] = {x[2 * i], x[2 * i + 1]}, p, t;
    __poseidon_optimize(both, in, &p, &t);
    x[2 * i] = 3.0 * t;
  }
  double s = x[0];
  for (int i = 0; i < n; ++i) {
    double y;
    __poseidon_optimize(step, &s, &y);
    x[i + 1] = 3.0 * y;
  }
}

// CHECK-DAG: [poseidon-profgen] both(double const*, double*, double*) at call 0 in k(double*, int): 3 shadow object(s), output(s) 2, not read after the call: 1
// CHECK-DAG: [poseidon-profgen] step(double*, double*) at call 1 in k(double*, int): 2 shadow object(s), output(s) 0 1
// The seeds of t, s and y.
// IR-LABEL: define {{.*}} @_Z1kPdi(
// IR-COUNT-3: store double 1.000000e+00
// IR-NOT: store double 1.000000e+00
// IR: ret void
