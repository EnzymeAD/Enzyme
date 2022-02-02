;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

define void @wrapper(i32 %n, double %a, double* %x, i32 %incx) {
entry:
  tail call void @cblas_dscal(i32 %n, double %a, double* %x, i32 %incx)
  ret void
}

declare void @cblas_dscal(i32, double, double*, i32)

define void @caller(i32 %n, double %a, double* %x, double* %_x, i32 %incx) {
entry:
  %call = tail call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (i32, double, double*, i32)* @wrapper to i8*), i32 %n, double %a, double* %x, double* %_x, i32 %incx)
  ret void
}

declare double @__enzyme_autodiff(i8*, ...)

;CHECK:define internal { double } @diffewrapper(i32 %n, double %a, double* %x, double* %"x'", i32 %incx) {
;CHECK-NEXT:entry:
;CHECK-NEXT:  tail call void @cblas_dscal(i32 %n, double %a, double* %x, i32 %incx)
;CHECK-NEXT:  call void @cblas_dscal(i32 %n, double %a, double* %"x'", i32 %incx)
;CHECK-NEXT:  %0 = call fast double @cblas_ddot(i32 %n, double* nocapture readonly %x, i32 %incx, double* nocapture readonly %"x'", i32 %incx)
;CHECK-NEXT:  %1 = insertvalue { double } undef, double %0, 0
;CHECK-NEXT:  ret { double } %1
;CHECK-NEXT:}
