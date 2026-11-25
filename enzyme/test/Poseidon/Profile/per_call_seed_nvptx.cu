// REQUIRES: enzyme, nvptx-device-profiler
// RUN: rm -rf %t && mkdir %t
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O1 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c -mllvm -poseidon-print %s -o %t/gen.ll 2>&1 | FileCheck %s
// RUN: FileCheck --check-prefix=IR %s < %t/gen.ll
// RUN: FileCheck --check-prefixes=SHADOW,BOTH %s < %t/gen.ll
// RUN: FileCheck --check-prefixes=SHADOW,STEP %s < %t/gen.ll

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

// CHECK-DAG: [poseidon] both(double const*, double*, double*) at call 0 in k(double*, int): 3 shadow object(s), output(s) 2, not read after the call: 1
// CHECK-DAG: [poseidon] step(double*, double*) at call 1 in k(double*, int): 2 shadow object(s), output(s) 0 1
// The seeds of t, s and y, and no other.
// IR-LABEL: define {{.*}} @_Z1kPdi(
// IR-COUNT-3: store double 1.000000e+00
// IR-NOT: store double 1.000000e+00
// IR: {{^}}}

// SHADOW-LABEL: define {{.*}} @_Z1kPdi(
// SHADOW-SAME: ptr {{.*}}%[[X:[0-9]+]], i32
// SHADOW: %[[IN:[0-9]+]] = alloca [2 x double], align {{[0-9]+}}
// SHADOW: %[[DIN:[0-9]+]] = alloca [2 x double], align {{[0-9]+}}
// SHADOW: %[[P:[0-9]+]] = alloca double, align {{[0-9]+}}
// SHADOW: %[[DP:[0-9]+]] = alloca double, align {{[0-9]+}}
// SHADOW: %[[T:[0-9]+]] = alloca double, align {{[0-9]+}}
// SHADOW: %[[DT:[0-9]+]] = alloca double, align {{[0-9]+}}
// SHADOW: %[[S:[0-9]+]] = alloca double, align {{[0-9]+}}
// SHADOW: %[[DS:[0-9]+]] = alloca double, align {{[0-9]+}}
// SHADOW: %[[Y:[0-9]+]] = alloca double, align {{[0-9]+}}
// SHADOW: %[[DY:[0-9]+]] = alloca double, align {{[0-9]+}}

// The input's shadow and p's are zero, t's is seeded; t is read after the call.
// BOTH: call void @llvm.memset.p0.i64(ptr {{.*}}%[[DIN]], i8 0, i64 16, i1 false)
// BOTH-NEXT: store i64 0, ptr %[[DP]],
// BOTH-NEXT: store double 1.000000e+00, ptr %[[DT]],
// BOTH-NEXT: call {{.*}} @diffe{{.*}}_Z4bothPKdPdS1_.poseidon._Z1kPdi.0(ptr %[[IN]], ptr %[[DIN]], ptr %[[P]], ptr %[[DP]], ptr %[[T]], ptr %[[DT]])
// BOTH: %[[TV:[0-9]+]] = load double, ptr %[[T]],
// BOTH: fmul {{.*}}double %[[TV]], 3.000000e+00
// BOTH: {{^}}}

// s holds x[0] and is read again by the call only, y after the call: both
// shadows are seeded.
// STEP: %[[SV:[0-9]+]] = load double, ptr %[[X]],
// STEP-NEXT: store double %[[SV]], ptr %[[S]],
// STEP: store double 1.000000e+00, ptr %[[DS]],
// STEP-NEXT: store double 1.000000e+00, ptr %[[DY]],
// STEP-NEXT: call {{.*}} @diffe{{.*}}_Z4stepPdS_.poseidon._Z1kPdi.1(ptr %[[S]], ptr %[[DS]], ptr %[[Y]], ptr %[[DY]])
// STEP-NOT: load {{.*}}, ptr %[[S]],
// STEP: %[[YV:[0-9]+]] = load double, ptr %[[Y]],
// STEP-NOT: load {{.*}}, ptr %[[S]],
// STEP: fmul {{.*}}double %[[YV]], 3.000000e+00
// STEP-NOT: load {{.*}}, ptr %[[S]],
// STEP: {{^}}}
