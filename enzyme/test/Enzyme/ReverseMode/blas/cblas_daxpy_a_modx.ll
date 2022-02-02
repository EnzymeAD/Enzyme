;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

define void @wrapper(i32 %n, double %alpha, double* %x, i32 %incx, double* %y, i32 %incy) {
entry:
  tail call void @cblas_daxpy(i32 %n, double %alpha, double* %x, i32 %incx, double* %y, i32 %incy)
  store double 0.000000e+00, double* %x, align 8
  ret void
}

declare void @cblas_daxpy(i32, double, double*, i32, double*, i32)

define void @caller(i32 %n, double %alpha, double* %x, double* %_x, i32 %incx, double* %y, double* %_y, i32 %incy) {
entry:
  tail call void (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (i32, double, double*, i32, double*, i32)* @wrapper to i8*), i32 %n, double %alpha, double* %x, double* %_x, i32 %incx, double* %y, double* %_y, i32 %incy)
  ret void
}

declare void @__enzyme_autodiff(i8*, ...)

;CHECK:define internal { double } @diffewrapper(i32 %n, double %alpha, double* %x, double* %"x'", i32 %incx, double* %y, double* %"y'", i32 %incy) {
;CHECK-NEXT:entry:
;CHECK-NEXT:  %0 = zext i32 %n to i64
;CHECK-NEXT:  %mallocsize = mul i64 %0, ptrtoint (double* getelementptr (double, double* null, i32 1) to i64)
;CHECK-NEXT:  %malloccall = tail call i8* @malloc(i64 %mallocsize)
;CHECK-NEXT:  %1 = bitcast i8* %malloccall to double*
;CHECK-NEXT:  call void @__enzyme_memcpy_doubleda0sa0stride(double* %1, double* %x, i32 %n, i32 %incx)
;CHECK-NEXT:  tail call void @cblas_daxpy(i32 %n, double %alpha, double* %x, i32 %incx, double* %y, i32 %incy)
;CHECK-NEXT:  store double 0.000000e+00, double* %x, align 8
;CHECK-NEXT:  store double 0.000000e+00, double* %"x'", align 8
;CHECK-NEXT:  call void @cblas_daxpy(i32 %n, double %alpha, double* %"y'", i32 %incy, double* %"x'", i32 1)
;CHECK-NEXT:  %2 = call fast double @cblas_ddot(i32 %n, double* nocapture readonly %"y'", i32 %incy, double* nocapture readonly %1, i32 1)
;CHECK-NEXT:  %3 = insertvalue { double } undef, double %2, 0
;CHECK-NEXT:  ret { double } %3
;CHECK-NEXT:}
