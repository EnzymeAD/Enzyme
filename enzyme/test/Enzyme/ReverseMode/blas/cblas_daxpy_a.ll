;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

define void @wrapper(i32 %n, double %a, double* %x, double* %y) {
entry:
  tail call void @cblas_daxpy(i32 %n, double %a, double* %x, i32 1, double* %y, i32 1)
  ret void
}

declare void @cblas_daxpy(i32, double, double*, i32, double*, i32)

define void @caller(double %alpha, double* %x, double* %_x, double* %y, double* %_y) {
entry:
  %call = tail call double @__enzyme_autodiff(i8* bitcast (void (i32, double, double*, double*)* @wrapper to i8*), i32 3, double %alpha, double* %x, double* %_x, double* %y, double* %_y)
  ret void
}

declare double @__enzyme_autodiff(i8*, i32, double, double*, double*, double*, double*)

;CHECK:define internal { double } @diffewrapper(i32 %n, double %a, double* %x, double* %"x'", double* %y, double* %"y'") {
;CHECK-NEXT:entry:
;CHECK-NEXT:  tail call void @cblas_daxpy(i32 %n, double %a, double* %x, i32 1, double* %y, i32 1)
;CHECK-NEXT:  call void @cblas_daxpy(i32 %n, double %a, double* %"y'", i32 1, double* %"x'", i32 1)
;CHECK-NEXT:  %0 = call fast double @cblas_ddot(i32 %n, double* nocapture readonly %"y'", i32 1, double* nocapture readonly %x, i32 1)
;CHECK-NEXT:  %1 = insertvalue { double } undef, double %0, 0
;CHECK-NEXT:  ret { double } %1
;CHECK-NEXT:}
