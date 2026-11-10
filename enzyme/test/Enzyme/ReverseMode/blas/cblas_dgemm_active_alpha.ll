; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

; The CBLAS calling convention passes alpha by value, so its adjoint must
; accumulate into the differential keyed by the original argument rather
; than its clone in the new function.

declare void @cblas_dgemm(i32, i32, i32, i32, i32, i32, double, ptr, i32, ptr, i32, double, ptr, i32)

declare double @__enzyme_autodiff(...)

define void @f(double %alpha, ptr %A, ptr %B, ptr %C) {
entry:
  call void @cblas_dgemm(i32 101, i32 111, i32 111, i32 3, i32 3, i32 3, double %alpha, ptr %A, i32 3, ptr %B, i32 3, double 0.000000e+00, ptr %C, i32 3)
  ret void
}

define double @active(double %alpha, ptr %A, ptr %B, ptr %C, ptr %dC) {
entry:
  %r = call double (...) @__enzyme_autodiff(ptr @f, metadata !"enzyme_out", double %alpha, metadata !"enzyme_const", ptr %A, metadata !"enzyme_const", ptr %B, metadata !"enzyme_dup", ptr %C, ptr %dC)
  ret double %r
}

; CHECK-NOT: @cblas_dlascl
; CHECK-LABEL: define internal { double } @diffef(double %alpha, ptr nocapture readonly %A, ptr nocapture readonly %B, ptr nocapture %C, ptr nocapture %"C'")
; CHECK: %[[ALPHA_DIFFE:.+]] = alloca double
; CHECK: store double 0.000000e+00, ptr %[[ALPHA_DIFFE]]
; CHECK: invert{{.*}}:
; CHECK: %[[DOT:.+]] = call fast double @cblas_ddot(i32 9, ptr nocapture readonly %"C'", i32 1, ptr nocapture readonly %[[PRODUCT:.+]], i32 1)
; CHECK: %[[OLD:.+]] = load double, ptr %[[ALPHA_DIFFE]]
; CHECK-NEXT: %[[NEW:.+]] = fadd fast double %[[OLD]], %[[DOT]]
; CHECK-NEXT: store double %[[NEW]], ptr %[[ALPHA_DIFFE]]
; CHECK: Gfor.body.i:
; CHECK: %[[GEP:.+]] = getelementptr double, ptr %"C'", i32
; CHECK-NEXT: store double 0.000000e+00, ptr %[[GEP]]
; CHECK: __enzyme_cblas_lascl_double_32.exit:
; CHECK: %[[RESULT:.+]] = load double, ptr %[[ALPHA_DIFFE]]
; CHECK: insertvalue { double } {{(undef|poison)}}, double %[[RESULT]], 0
; CHECK-LABEL: define internal void @__enzyme_cblas_lascl_double_32(i32 %layout, i32 %type, double %cfrom, double %cto, i32 %M, i32 %N, ptr noalias nocapture %A, i32 %lda)
