// REQUIRES: enzyme
// RUN: rm -rf %t && mkdir %t
// RUN: %clang -x c++ -O1 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/cache %s -o %t/gen.ll 2>&1 | FileCheck --implicit-check-not='warning:' %s
// RUN: FileCheck --check-prefix=IR --implicit-check-not=__enzyme --implicit-check-not=poseidonLog --implicit-check-not=__poseidon %s < %t/gen.ll

// An mfem::forall body compiled for the host only: the annotation on the lambda
// marks the GPU kernels that call it, and a host build has none. The lambda's
// closure is no allocation the profiler could shadow, so the body is left as
// written rather than made a host site. Outside CUDA an attribute parses only
// after the lambda's parameter list.

#include "poseidon/poseidon.h"

template <typename BODY> void forall(int n, BODY &&body) {
  for (int i = 0; i < n; ++i)
    body(i);
}

void axpy(int n, double a, const double *x, double *y) {
  forall(n, [=] (int i) POSEIDON_OPTIMIZE_TAU(1e-9) { y[i] += a * x[i]; });
}

// CHECK: annotated_lambda_host.cpp:[[@LINE-3]]:{{[0-9]+}}: warning: Poseidon: axpy(int, double, double const*, double*)::$_0::operator()(int) const is a lambda; POSEIDON_OPTIMIZE on a lambda marks the GPU kernels that call it, and this compilation has none. Emitting the original computation.
// IR-NOT: __enzyme_autodiff
// IR-NOT: poseidonLogValue
// IR-LABEL: define {{.*}}@_Z4axpyidPKdPd(
// IR-SAME: i32 {{.*}}%{{[0-9]+}}, double {{.*}}[[A:%[0-9]+]], ptr {{.*}}[[X:%[0-9]+]], ptr {{.*}}[[Y:%[0-9]+]])
// IR: [[I:%[0-9]+]] = phi i64
// IR: [[XP:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[X]], i64 [[I]]
// IR-NEXT: [[XI:%[0-9]+]] = load double, ptr [[XP]]
// IR-NEXT: [[YP:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[Y]], i64 [[I]]
// IR-NEXT: [[YI:%[0-9]+]] = load double, ptr [[YP]]
// IR-NEXT: [[R:%[0-9]+]] = tail call double @llvm.fmuladd.f64(double [[A]], double [[XI]], double [[YI]])
// IR-NEXT: store double [[R]], ptr [[YP]]
// IR-LABEL: define internal void @"_ZZ4axpyidPKdPdENK3$_0clEi"(
// IR-SAME: ptr {{.*}}[[CL:%[0-9]+]], i32 {{.*}}[[J:%[0-9]+]])
// IR: [[AP:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[CL]], i64 8
// IR-NEXT: [[LA:%[0-9]+]] = load double, ptr [[AP]]
// IR-NEXT: [[XFP:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[CL]], i64 16
// IR-NEXT: [[LX:%[0-9]+]] = load ptr, ptr [[XFP]]
// IR-NEXT: [[IDX:%[0-9]+]] = sext i32 [[J]] to i64
// IR-NEXT: [[XJP:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[LX]], i64 [[IDX]]
// IR-NEXT: [[XJ:%[0-9]+]] = load double, ptr [[XJP]]
// IR-NEXT: [[LY:%[0-9]+]] = load ptr, ptr [[CL]]
// IR-NEXT: [[YJP:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[LY]], i64 [[IDX]]
// IR-NEXT: [[YJ:%[0-9]+]] = load double, ptr [[YJP]]
// IR-NEXT: [[S:%[0-9]+]] = tail call double @llvm.fmuladd.f64(double [[LA]], double [[XJ]], double [[YJ]])
// IR-NEXT: store double [[S]], ptr [[YJP]]
// IR-NEXT: ret void
