; REQUIRES: enzyme
; RUN: %opt < %s %loadPoseidon -passes=poseidon -poseidon-profile-generate -S | FileCheck --check-prefix=REQ %s
; RUN: %opt < %s %loadPoseidonEnzyme -passes="poseidon,enzyme,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" -enzyme-preopt=false -poseidon-profile-generate -S | FileCheck %s

; A marker with a fixed list of parameters, as a C++ instantiation declares
; it, whose poseidon_tau and target the request leaves out: the request takes
; a variadic type, since the marker's has parameters for both.

@enzyme_dup = global i32 0
@poseidon_tau = global i32 0

define void @scale(ptr %x, ptr %y) {
entry:
  %a = load double, ptr %x
  %m = fmul fast double %a, %a
  store double %m, ptr %y
  ret void
}

define double @sq(double %x, double %y) {
entry:
  %a = fmul fast double %x, %x
  %b = fmul fast double %y, %y
  %s = fadd fast double %a, %b
  ret double %s
}

define void @test_void(ptr %x, ptr %dx, ptr %y, ptr %dy) {
entry:
  %tau = load i32, ptr @poseidon_tau
  %dup = load i32, ptr @enzyme_dup
  call void @__poseidon_fp_optimize(ptr @scale, i32 %tau, double 1.0e-6, i32 %dup, ptr %x, ptr %dx, i32 %dup, ptr %y, ptr %dy)
  ret void
}

define void @test_unused(double %x, double %y) {
entry:
  %tau = load i32, ptr @poseidon_tau
  %0 = call double @__enzyme_fp_optimize(ptr @sq, i32 %tau, double 1.0e-6, double %x, double %y)
  ret void
}

declare void @__poseidon_fp_optimize(ptr, i32, double, i32, ptr, ptr, i32, ptr, ptr)
declare double @__enzyme_fp_optimize(ptr, i32, double, double, double)

; REQ-LABEL: define void @test_void(
; REQ: call void (ptr, ...) @__enzyme_autodiff_poseidon(ptr @preprocess_scale, i32 %dup, ptr %x, ptr %dx, i32 %dup, ptr %y, ptr %dy)
; REQ-LABEL: define void @test_unused(
; REQ: call double (ptr, ...) @__enzyme_autodiff_poseidon.1(ptr @preprocess_sq, double %x, double %y)

; CHECK-LABEL: define void @test_void(
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @diffepreprocess_scale(ptr %x, ptr %dx, ptr %y, ptr %dy)
; CHECK-NEXT:   ret void
; CHECK-LABEL: define void @test_unused(
; CHECK-NEXT: entry:
; CHECK-NEXT:   call {{.*}}{ double, double } @diffepreprocess_sq(double %x, double %y, double 1.000000e+00)
; CHECK-NEXT:   ret void
