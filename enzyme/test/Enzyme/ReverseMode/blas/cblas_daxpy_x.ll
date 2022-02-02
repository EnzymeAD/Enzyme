;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

@enzyme_const = common global i32 0, align 4

define void @wrapper(i32 %n, double %alpha, double* %x, i32 %incx, double* %y, i32 %incy) {
entry:
  tail call void @cblas_daxpy(i32 %n, double %alpha, double* %x, i32 %incx, double* %y, i32 %incy)
  ret void
}

declare void @cblas_daxpy(i32, double, double*, i32, double*, i32)

define void @caller(i32 %n, double %alpha, double* %x, double* %_x, i32 %incx, double* %y, double* %_y, i32 %incy) {
entry:
  %0 = load i32, i32* @enzyme_const, align 4
  tail call void (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (i32, double, double*, i32, double*, i32)* @wrapper to i8*), i32 %n, i32 %0, double %alpha, double* %x, double* %_x, i32 %incx, double* %y, double* %_y, i32 %incy)
  ret void
}

declare void @__enzyme_autodiff(i8*, ...)

;CHECK:define internal void @diffewrapper(i32 %n, double %alpha, double* %x, double* %"x'", i32 %incx, double* %y, double* %"y'", i32 %incy) {
;CHECK-NEXT:entry:
;CHECK-NEXT:  tail call void @cblas_daxpy(i32 %n, double %alpha, double* %x, i32 %incx, double* %y, i32 %incy)
;CHECK-NEXT:  call void @cblas_daxpy(i32 %n, double %alpha, double* %"y'", i32 %incy, double* %"x'", i32 %incx)
;CHECK-NEXT:  ret void
;CHECK-NEXT:}
