;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

define void @wrapper(i32 %n, double %alpha, double* %x, i32 %incx, double* %y, i32 %incy, double* %a, i32 %lda) {
entry:
  tail call void @cblas_dsyr2(i32 102, i32 122, i32 %n, double %alpha, double* %x, i32 %incx, double* %y, i32 %incy, double* %a, i32 %lda)
  ret void
}

declare void @cblas_dsyr2(i32, i32, i32, double, double*, i32, double*, i32, double*, i32)

define void @active(i32 %n, double %alpha, i32 %lda, double* %x, double* %_x, i32 %incx, double* %y, double* %_y, i32 %incy, double* %a, double* %_a) {
entry:
  tail call void @__enzyme_autodiff(i8* bitcast (void (i32, double, double*, i32, double*, i32, double*, i32)* @wrapper to i8*), i32 %n, double %alpha, double* %x, double* %_x, i32 %incx, double* %y, double* %_y, i32 %incy, double* %a, double* %_a, i32 %lda)
  ret void
}

declare void @__enzyme_autodiff(i8*, i32, double, double*, double*, i32, double*, double*, i32, double*, double*, i32)

;CHECK: define void @active
;CHECK-NEXT: entry
;CHECK-NEXT: call { double } @[[active:.+]](

;CHECK:define internal { double } @[[active]](i32 %n, double %alpha, double* %x, double* %"x'", i32 %incx, double* %y, double* %"y'", i32 %incy, double* %a, double* %"a'", i32 %lda)
;CHECK-NEXT:entry:
;CHECK-NEXT:  %0 = zext i32 %n to i64
;CHECK-NEXT:  %mallocsize = mul i64 %0, ptrtoint (double* getelementptr (double, double* null, i32 1) to i64)
;CHECK-NEXT:  %malloccall = tail call i8* @malloc(i64 %mallocsize)
;CHECK-NEXT:  %1 = bitcast i8* %malloccall to double*
;CHECK-NEXT:  tail call void @cblas_dsyr2(i32 102, i32 122, i32 %n, double %alpha, double* %x, i32 %incx, double* %y, i32 %incy, double* %a, i32 %lda)
;CHECK-NEXT:  %2 = fadd fast double 2.000000e+00, %alpha
;CHECK-NEXT:  call void @cblas_dsymv(i32 102, i32 122, i32 %n, double %2, double* %"a'", i32 %lda, double* %x, i32 %incx, double 1.000000e+00, double* %"y'", i32 %incy)
;CHECK-NEXT:  %3 = fadd fast double 2.000000e+00, %alpha
;CHECK-NEXT:  call void @cblas_dsymv(i32 102, i32 122, i32 %n, double %3, double* %"a'", i32 %lda, double* %y, i32 %incy, double 1.000000e+00, double* %"y'", i32 %incx)
;CHECK-NEXT:  call void @cblas_dsymv(i32 102, i32 122, i32 %n, double 2.000000e+00, double* %"a'", i32 %lda, double* %y, i32 %incy, double 0.000000e+00, double* %1, i32 1)
;CHECK-NEXT:  %4 = call fast double @cblas_ddot(i32 %n, double* nocapture readonly %x, i32 %incy, double* nocapture readonly %1, i32 1)
;CHECK-NEXT:  %5 = insertvalue { double } undef, double %4, 0
;CHECK-NEXT:  ret { double } %5
;CHECK-NEXT:}
