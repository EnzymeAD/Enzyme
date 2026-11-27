; DEFINE: %{clang} = %clang -x ir -Wno-override-module -O1 -S -emit-llvm %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/unrelated_site_profile -o /dev/null
; RUN: rm -rf %t && mkdir %t
; RUN: sed 's/@TAU@/0.000000e+00/g' %s > %t/zero.ll
; RUN: sed 's/@TAU@/-1.000000e-06/g' %s > %t/neg.ll
; RUN: sed 's/@TAU@/0x7FF8000000000000/g' %s > %t/nan.ll
; RUN: not %{clang} %t/zero.ll -mllvm -poseidon-cache=%t/zero 2>&1 | FileCheck %s -DV=0 --implicit-check-not=error: --implicit-check-not=warning:
; RUN: not %{clang} %t/neg.ll -mllvm -poseidon-cache=%t/neg 2>&1 | FileCheck %s -DV=-1e-06 --implicit-check-not=error: --implicit-check-not=warning:
; RUN: not %{clang} %t/nan.ll -mllvm -poseidon-cache=%t/nan 2>&1 | FileCheck %s -DV=nan --implicit-check-not=error: --implicit-check-not=warning:

; An accuracy target of 0, a negative one or NaN is an error at its site in
; every form a site takes: an annotation, a call through either identity, and
; a marker call's poseidon_tau after an explicit or an automatic list.

; CHECK-DAG: error: <unknown>:0:0: in function annotated double (double): Poseidon: annotated carries the accuracy target [[V]], which is not a positive finite number
; CHECK-DAG: error: <unknown>:0:0: in function identity double (double): Poseidon: the accuracy target [[V]] of body is not a positive finite number
; CHECK-DAG: error: <unknown>:0:0: in function explicit_identity double (double): Poseidon: the accuracy target [[V]] of body is not a positive finite number
; CHECK-DAG: error: <unknown>:0:0: in function marker_explicit double (double): Poseidon: the accuracy target [[V]] of body is not a positive finite number
; CHECK-DAG: error: <unknown>:0:0: in function marker_automatic double (double): Poseidon: the accuracy target [[V]] of body is not a positive finite number
; CHECK: 5 errors generated.

@.str = private unnamed_addr constant [9 x i8] c"poseidon\00", section "llvm.metadata"
@.file = private unnamed_addr constant [7 x i8] c"tau.cc\00", section "llvm.metadata"
@.tau = private unnamed_addr constant { double } { double @TAU@ }, section "llvm.metadata"
@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }] [{ ptr, ptr, ptr, i32, ptr } { ptr @annotated, ptr @.str, ptr @.file, i32 1, ptr @.tau }], section "llvm.metadata"

define double @annotated(double %x) {
  %r = fmul double %x, %x
  ret double %r
}

define double @body(double %x) {
  %r = fmul double %x, %x
  ret double %r
}

define double @identity(double %x) {
  %f = call ptr @__poseidon_site_tau(ptr @body, double @TAU@)
  %r = call double %f(double %x)
  ret double %r
}

define double @explicit_identity(double %x) {
  %f = call ptr @__poseidon_site_explicit_tau(ptr @body, double @TAU@)
  %r = call double (...) %f(metadata !"enzyme_out", double %x)
  ret double %r
}

define double @marker_explicit(double %x) {
  %r = call double (ptr, ...) @__poseidon_fp_optimize(ptr @body, metadata !"enzyme_out", double %x, metadata !"poseidon_tau", double @TAU@)
  ret double %r
}

define double @marker_automatic(double %x) {
  %r = call double (ptr, ...) @__poseidon_fp_optimize(ptr @body, double %x, metadata !"poseidon_tau", double @TAU@)
  ret double %r
}

declare ptr @__poseidon_site_tau(ptr, double)
declare ptr @__poseidon_site_explicit_tau(ptr, double)
declare double @__poseidon_fp_optimize(ptr, ...)
