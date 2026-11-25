// REQUIRES: enzyme, nvptx-device-profiler
// RUN: rm -rf %t && mkdir %t
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O1 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites %s -o %t/k.ll 2>&1 | FileCheck %s
// RUN: FileCheck --check-prefix=IR %s < %t/k.ll
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O1 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c -mllvm -poseidon-print %s -o %t/gen.ll 2>&1 | FileCheck --check-prefix=GEN %s
// RUN: FileCheck --check-prefix=GENIR %s < %t/gen.ll

// In device code a per-call site is a site as on the host. The pass makes the
// shadow of every local object a call passes, zero, with the seed on what the
// call writes; two pointers into device memory at a constant distance share
// one shadow, which covers what the call reaches through them; and an object
// holding pointers is passed as a constant.

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

// CHECK-DAG: per_call_nvptx.cu:[[@LINE-12]]:30: warning: Poseidon: no profile at {{.*}}_Z2sqd.poseidon._Z1kPd.0.fpprofile:
// CHECK-DAG: per_call_nvptx.cu:[[@LINE-13]]:30: warning: Poseidon: no profile at {{.*}}_Z5scalePKdPd.poseidon._Z1kPd.1.fpprofile:
// CHECK-DAG: per_call_nvptx.cu:[[@LINE-14]]:30: warning: Poseidon: no profile at {{.*}}_Z5scalePKdPd.poseidon._Z1kPd.2.fpprofile:
// CHECK-DAG: per_call_nvptx.cu:[[@LINE-15]]:30: warning: Poseidon: no profile at {{.*}}_Z5weighRK6holderPKdPd.poseidon._Z1kPd.3.fpprofile:
// CHECK-NOT: cannot be a Poseidon site
// CHECK: {{^}}4 warnings generated when compiling for {{.+}}.{{$}}
// IR-LABEL: define {{.*}} @_Z1kPd(
// IR-NOT: call
// IR: ret void
// IR-NOT: __poseidon_fp_optimize
// IR-NOT: __poseidon_site

// GEN-DAG: [poseidon-profgen] scale(double const*, double*) at call 1 in k(double*): 2 shadow object(s), output(s) 1
// GEN-DAG: [poseidon-profgen] scale(double const*, double*) at call 2 in k(double*): 1 shadow object(s), output(s) 1
// GEN-DAG: [poseidon-profgen] weigh(holder const&, double const*, double*) at call 3 in k(double*): 2 shadow object(s), output(s) 2
// GENIR-LABEL: define {{.*}} @_Z1kPd(
// GENIR-SAME: ptr {{.*}}%[[XP:[0-9]+]])
// GENIR-DAG: %[[X:[0-9]+]] = alloca [32 x i8], align {{[0-9]+}}
// GENIR-DAG: %[[A:[0-9]+]] = alloca [2 x double], align {{[0-9]+}}
// GENIR-DAG: %[[DA:[0-9]+]] = alloca [2 x double], align {{[0-9]+}}
// GENIR-DAG: %[[B:[0-9]+]] = alloca [2 x double], align {{[0-9]+}}
// GENIR-DAG: %[[DB:[0-9]+]] = alloca [2 x double], align {{[0-9]+}}
// GENIR-DAG: %[[H:[0-9]+]] = alloca %struct.holder, align {{[0-9]+}}
// GENIR-DAG: %[[C:[0-9]+]] = alloca double, align {{[0-9]+}}
// GENIR-DAG: %[[DC:[0-9]+]] = alloca double, align {{[0-9]+}}
// GENIR-DAG: %[[D:[0-9]+]] = alloca double, align {{[0-9]+}}
// GENIR-DAG: %[[DD:[0-9]+]] = alloca double, align {{[0-9]+}}
// A call that passes no pointer has no shadow.
// GENIR: %[[X1P:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[XP]], i64 8
// GENIR: %[[X1:[0-9]+]] = load double, ptr %[[X1P]],
// GENIR: %[[SQ:[0-9]+]] = {{.*}}call {{.*}}double @diffe{{.*}}_Z2sqd.poseidon._Z1kPd.0(double %[[X1]])
// GENIR: store double %[[SQ]], ptr %[[XP]],
// a and b are local: a holds x[2] and x[3], its shadow is zero; b is read
// after the call, its shadow has the seed in both elements the call writes.
// GENIR: %[[X2P:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[XP]], i64 16
// GENIR: %[[X2:[0-9]+]] = load double, ptr %[[X2P]],
// GENIR: store double %[[X2]], ptr %[[A]],
// GENIR: %[[A1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[A]], i64 8
// GENIR: %[[X3P:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[XP]], i64 24
// GENIR: %[[X3:[0-9]+]] = load double, ptr %[[X3P]],
// GENIR: store double %[[X3]], ptr %[[A1]],
// GENIR: call void @llvm.memset.p0.i64(ptr {{.*}}%[[DA]], i8 0, i64 16, i1 false)
// GENIR: store double 1.000000e+00, ptr %[[DB]],
// GENIR: %[[DB1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[DB]], i64 8
// GENIR: store double 1.000000e+00, ptr %[[DB1]],
// GENIR: call {{.*}} @diffe{{.*}}_Z5scalePKdPd.poseidon._Z1kPd.1(ptr %[[A]], ptr %[[DA]], ptr %[[B]], ptr %[[DB]])
// GENIR: load double, ptr %[[B]],
// GENIR: %[[B1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[B]], i64 8
// GENIR: load double, ptr %[[B1]],
// x + 5 and x + 7 share one shadow of the 32 bytes the call reaches, the second
// 16 bytes into it, as far as the pointers are apart: zero where the call reads,
// the seed where it writes.
// GENIR: %[[X5:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[XP]], i64 40
// GENIR: %[[X7:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[XP]], i64 56
// GENIR: call void @llvm.memset.p0.i64(ptr {{.*}}%[[X]], i8 0, i64 16, i1 false)
// GENIR: %[[Y:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[X]], i64 16
// GENIR: store double 1.000000e+00, ptr %[[Y]],
// GENIR: %[[Y1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[X]], i64 24
// GENIR: store double 1.000000e+00, ptr %[[Y1]],
// GENIR: call {{.*}} @diffe{{.*}}_Z5scalePKdPd.poseidon._Z1kPd.2(ptr {{(nonnull )?}}%[[X5]], ptr %[[X]], ptr {{(nonnull )?}}%[[X7]], ptr %[[Y]])
// The holder, which holds x + 9, has no shadow; d holds x[10] and its shadow is
// zero; c is read after the call and its shadow seeded.
// GENIR: %[[X9:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[XP]], i64 72
// GENIR: store ptr %[[X9]], ptr %[[H]],
// GENIR: %[[X10P:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[XP]], i64 80
// GENIR: %[[X10:[0-9]+]] = load double, ptr %[[X10P]],
// GENIR: store double %[[X10]], ptr %[[D]],
// GENIR: store i64 0, ptr %[[DD]],
// GENIR: store double 1.000000e+00, ptr %[[DC]],
// GENIR: call {{.*}} @diffe{{.*}}_Z5weighRK6holderPKdPd.poseidon._Z1kPd.3(ptr %[[H]], ptr %[[D]], ptr %[[DD]], ptr %[[C]], ptr %[[DC]])
// GENIR: %[[CV:[0-9]+]] = load double, ptr %[[C]],
// GENIR: %[[X11P:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[XP]], i64 88
// GENIR: store double %[[CV]], ptr %[[X11P]],
// GENIR: ret void
