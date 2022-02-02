;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

define void @wrapper(i32 %n, double* %x, i32 %incx, double* %y, i32 %incy) {
entry:
  tail call void @cblas_dswap(i32 %n, double* %x, i32 %incx, double* %y, i32 %incy)
  store double 0.000000e+00, double* %x, align 8
  store double 1.000000e+00, double* %y, align 8
  ret void
}

declare void @cblas_dswap(i32, double*, i32, double*, i32)

define void @caller(i32 %n, double* %x, double* %_x, i32 %incx, double* %y, double* %_y, i32 %incy) {
entry:
  tail call void (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (i32, double*, i32, double*, i32)* @wrapper to i8*), i32 %n, double* %x, double* %_x, i32 %incx, double* %y, double* %_y, i32 %incy)
  ret void
}

declare void @__enzyme_autodiff(i8*, ...)

;CHECK:define internal void @diffewrapper(i32 %n, double* %x, double* %"x'", i32 %incx, double* %y, double* %"y'", i32 %incy) {
;CHECK-NEXT:entry:
;CHECK-NEXT:  tail call void @cblas_dswap(i32 %n, double* %x, i32 %incx, double* %y, i32 %incy)
;CHECK-NEXT:  store double 0.000000e+00, double* %x, align 8
;CHECK-NEXT:  store double 1.000000e+00, double* %y, align 8
;CHECK-NEXT:  store double 0.000000e+00, double* %"y'", align 8
;CHECK-NEXT:  store double 0.000000e+00, double* %"x'", align 8
;CHECK-NEXT:  call void @cblas_dswap(i32 %n, double* %"x'", i32 %incx, double* %"y'", i32 %incy)
;CHECK-NEXT:  ret void
;CHECK-NEXT:}
