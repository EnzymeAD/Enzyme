;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

@enzyme_const = common global i32 0, align 4

define void @wrapper(i32 %n, double %a, double* %x, i32 %incx) {
entry:
  tail call void @cblas_dscal(i32 %n, double %a, double* %x, i32 %incx)
  ret void
}

declare void @cblas_dscal(i32, double, double*, i32)

define void @caller(i32 %n, double %a, double* %x, double* %_x, i32 %incx) {
entry:
  %0 = load i32, i32* @enzyme_const, align 4
  %call = tail call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (i32, double, double*, i32)* @wrapper to i8*), i32 %n, i32 %0, double %a, double* %x, double* %_x, i32 %incx)
  ret void
}

declare double @__enzyme_autodiff(i8*, ...)

;CHECK:define internal void @diffewrapper(i32 %n, double %a, double* %x, double* %"x'", i32 %incx) {
;CHECK-NEXT:entry:
;CHECK-NEXT:  tail call void @cblas_dscal(i32 %n, double %a, double* %x, i32 %incx)
;CHECK-NEXT:  call void @cblas_dscal(i32 %n, double %a, double* %"x'", i32 %incx)
;CHECK-NEXT:  ret void
;CHECK-NEXT:}
