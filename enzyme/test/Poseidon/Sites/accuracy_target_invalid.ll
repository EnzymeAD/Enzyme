; RUN: rm -rf %t
; RUN: not %clang -x ir -Wno-override-module -O1 -S -emit-llvm %s %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/unrelated_site_profile -mllvm -poseidon-cache=%t -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error: --implicit-check-not=warning:

; An accuracy target of 0, a negative one or NaN is an error at its site in
; every form a site takes: an annotation, a call through either identity
; (__poseidon_optimize_tau with fn's own arguments or with an explicit list),
; and a marker call's poseidon_tau after an explicit or an automatic list.


@.str = private unnamed_addr constant [9 x i8] c"poseidon\00", section "llvm.metadata"
@.file = private unnamed_addr constant [7 x i8] c"tau.cc\00", section "llvm.metadata"
@.tau_zero = private unnamed_addr constant { double } { double 0.000000e+00 }, section "llvm.metadata"
@.tau_neg = private unnamed_addr constant { double } { double -1.000000e-06 }, section "llvm.metadata"
@.tau_nan = private unnamed_addr constant { double } { double 0x7FF8000000000000 }, section "llvm.metadata"
@llvm.global.annotations = appending global [3 x { ptr, ptr, ptr, i32, ptr }] [{ ptr, ptr, ptr, i32, ptr } { ptr @annotated_zero, ptr @.str, ptr @.file, i32 1, ptr @.tau_zero }, { ptr, ptr, ptr, i32, ptr } { ptr @annotated_neg, ptr @.str, ptr @.file, i32 2, ptr @.tau_neg }, { ptr, ptr, ptr, i32, ptr } { ptr @annotated_nan, ptr @.str, ptr @.file, i32 3, ptr @.tau_nan }], section "llvm.metadata"

define double @annotated_zero(double %x) {
  %r = fmul double %x, %x
  ret double %r
}

define double @annotated_neg(double %x) {
  %r = fmul double %x, %x
  ret double %r
}

define double @annotated_nan(double %x) {
  %r = fmul double %x, %x
  ret double %r
}

define double @body(double %x) {
  %r = fmul double %x, %x
  ret double %r
}

define double @identity_zero(double %x) {
  %f = call ptr @__poseidon_site_tau(ptr @body, double 0.000000e+00)
  %r = call double %f(double %x)
  ret double %r
}

define double @explicit_identity_zero(double %x) {
  %f = call ptr @__poseidon_site_explicit_tau(ptr @body, double 0.000000e+00)
  %r = call double (...) %f(metadata !"enzyme_out", double %x)
  ret double %r
}

define double @marker_explicit_zero(double %x) {
  %r = call double (ptr, ...) @__poseidon_fp_optimize(ptr @body, metadata !"enzyme_out", double %x, metadata !"poseidon_tau", double 0.000000e+00)
  ret double %r
}

define double @marker_automatic_zero(double %x) {
  %r = call double (ptr, ...) @__poseidon_fp_optimize(ptr @body, double %x, metadata !"poseidon_tau", double 0.000000e+00)
  ret double %r
}

define double @identity_neg(double %x) {
  %f = call ptr @__poseidon_site_tau(ptr @body, double -1.000000e-06)
  %r = call double %f(double %x)
  ret double %r
}

define double @explicit_identity_neg(double %x) {
  %f = call ptr @__poseidon_site_explicit_tau(ptr @body, double -1.000000e-06)
  %r = call double (...) %f(metadata !"enzyme_out", double %x)
  ret double %r
}

define double @marker_explicit_neg(double %x) {
  %r = call double (ptr, ...) @__poseidon_fp_optimize(ptr @body, metadata !"enzyme_out", double %x, metadata !"poseidon_tau", double -1.000000e-06)
  ret double %r
}

define double @marker_automatic_neg(double %x) {
  %r = call double (ptr, ...) @__poseidon_fp_optimize(ptr @body, double %x, metadata !"poseidon_tau", double -1.000000e-06)
  ret double %r
}

define double @identity_nan(double %x) {
  %f = call ptr @__poseidon_site_tau(ptr @body, double 0x7FF8000000000000)
  %r = call double %f(double %x)
  ret double %r
}

define double @explicit_identity_nan(double %x) {
  %f = call ptr @__poseidon_site_explicit_tau(ptr @body, double 0x7FF8000000000000)
  %r = call double (...) %f(metadata !"enzyme_out", double %x)
  ret double %r
}

define double @marker_explicit_nan(double %x) {
  %r = call double (ptr, ...) @__poseidon_fp_optimize(ptr @body, metadata !"enzyme_out", double %x, metadata !"poseidon_tau", double 0x7FF8000000000000)
  ret double %r
}

define double @marker_automatic_nan(double %x) {
  %r = call double (ptr, ...) @__poseidon_fp_optimize(ptr @body, double %x, metadata !"poseidon_tau", double 0x7FF8000000000000)
  ret double %r
}

declare ptr @__poseidon_site_tau(ptr, double)
declare ptr @__poseidon_site_explicit_tau(ptr, double)
declare double @__poseidon_fp_optimize(ptr, ...)

; CHECK-DAG: error: <unknown>:0:0: in function annotated_zero double (double): Poseidon: annotated_zero carries the accuracy target 0, which is not a positive finite number
; CHECK-DAG: error: <unknown>:0:0: in function annotated_neg double (double): Poseidon: annotated_neg carries the accuracy target -1e-06, which is not a positive finite number
; CHECK-DAG: error: <unknown>:0:0: in function annotated_nan double (double): Poseidon: annotated_nan carries the accuracy target nan, which is not a positive finite number
; CHECK-DAG: error: <unknown>:0:0: in function identity_zero double (double): Poseidon: the accuracy target 0 of body is not a positive finite number
; CHECK-DAG: error: <unknown>:0:0: in function identity_neg double (double): Poseidon: the accuracy target -1e-06 of body is not a positive finite number
; CHECK-DAG: error: <unknown>:0:0: in function identity_nan double (double): Poseidon: the accuracy target nan of body is not a positive finite number
; CHECK-DAG: error: <unknown>:0:0: in function explicit_identity_zero double (double): Poseidon: the accuracy target 0 of body is not a positive finite number
; CHECK-DAG: error: <unknown>:0:0: in function explicit_identity_neg double (double): Poseidon: the accuracy target -1e-06 of body is not a positive finite number
; CHECK-DAG: error: <unknown>:0:0: in function explicit_identity_nan double (double): Poseidon: the accuracy target nan of body is not a positive finite number
; CHECK-DAG: error: <unknown>:0:0: in function marker_explicit_zero double (double): Poseidon: the accuracy target 0 of body is not a positive finite number
; CHECK-DAG: error: <unknown>:0:0: in function marker_explicit_neg double (double): Poseidon: the accuracy target -1e-06 of body is not a positive finite number
; CHECK-DAG: error: <unknown>:0:0: in function marker_explicit_nan double (double): Poseidon: the accuracy target nan of body is not a positive finite number
; CHECK-DAG: error: <unknown>:0:0: in function marker_automatic_zero double (double): Poseidon: the accuracy target 0 of body is not a positive finite number
; CHECK-DAG: error: <unknown>:0:0: in function marker_automatic_neg double (double): Poseidon: the accuracy target -1e-06 of body is not a positive finite number
; CHECK-DAG: error: <unknown>:0:0: in function marker_automatic_nan double (double): Poseidon: the accuracy target nan of body is not a positive finite number
; CHECK: 15 errors generated.
