;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

define void @wrapper(double %a, double* %x) {
entry:
  tail call void @cblas_dscal(i32 3, double %a, double* %x, i32 1)
  ret void
}

declare void @cblas_dscal(i32, double, double*, i32)

define void @caller(double %a, double* %x, double* %_x) {
entry:
  %call = tail call double @__enzyme_autodiff(i8* bitcast (void (double, double*)* @wrapper to i8*), double %a, double* %x, double* %_x)
  ret void
}

declare double @__enzyme_autodiff(i8*, double, double*, double*)

;CHECK:define internal { double } @diffewrapper(double %a, double* %x, double* %"x'") {
;CHECK-NEXT:entry:
;CHECK-NEXT:  tail call void @cblas_dscal(i32 3, double %a, double* %x, i32 1)
;CHECK-NEXT:  call void @cblas_dscal(i32 3, double %a, double* %"x'", i32 1)
;CHECK-NEXT:  %0 = call fast double @cblas_ddot(i32 3, double* nocapture readonly %x, i32 1, double* nocapture readonly %"x'", i32 1)
;CHECK-NEXT:  %1 = insertvalue { double } undef, double %0, 0
;CHECK-NEXT:  ret { double } %1
;CHECK-NEXT:}
