;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

define void @wrapper(i32 %m, i32 %n, double %alpha, double* %a, i32 %lda, double* %x, i32 %incx, double %beta, double* %y, i32 %incy) {
entry:
  tail call void @cblas_dgemv(i32 102, i32 111, i32 %m, i32 %n, double %alpha, double* %a, i32 %lda, double* %x, i32 %incx, double %beta, double* %y, i32 %incy)
  ret void
}

declare void @cblas_dgemv(i32, i32, i32, i32, double, double*, i32, double*, i32, double, double*, i32)

define void @wrapperMod(i32 %m, i32 %n, double %alpha, double* %a, i32 %lda, double* %x, i32 %incx, double %beta, double* %y, i32 %incy) {
entry:
  tail call void @cblas_dgemv(i32 102, i32 111, i32 %m, i32 %n, double %alpha, double* %a, i32 %lda, double* %x, i32 %incx, double %beta, double* %y, i32 %incy)
  store double 0.000000e+00, double* %a, align 8
  ret void
}

define void @active(i32 %m, i32 %n, double %alpha, double* %a, double* %_a, i32 %lda, double* %x, double* %_x, i32 %incx, double %beta, double* %y, double* %_y, i32 %incy) {
entry:
  tail call void @__enzyme_autodiff(i8* bitcast (void (i32, i32, double, double*, i32, double*, i32, double, double*, i32)* @wrapper to i8*), i32 %m, i32 %n, double %alpha, double* %a, double* %_a, i32 %lda, double* %x, double* %_x, i32 %incx, double %beta, double* %y, double* %_y, i32 %incy)
  ret void
}

declare void @__enzyme_autodiff(i8*, i32, i32, double, double*, double*, i32, double*, double*, i32, double, double*, double*, i32)

define void @activeMod(i32 %m, i32 %n, double %alpha, double* %a, double* %_a, i32 %lda, double* %x, double* %_x, i32 %incx, double %beta, double* %y, double* %_y, i32 %incy) {
entry:
  tail call void @__enzyme_autodiff(i8* bitcast (void (i32, i32, double, double*, i32, double*, i32, double, double*, i32)* @wrapperMod to i8*), i32 %m, i32 %n, double %alpha, double* %a, double* %_a, i32 %lda, double* %x, double* %_x, i32 %incx, double %beta, double* %y, double* %_y, i32 %incy)
  ret void
}

;CHECK: define void @active
;CHECK-NEXT: entry
;CHECK-NEXT: call { double, double } @[[active:.+]](

;CHECK: define void @activeMod
;CHECK-NEXT: entry
;CHECK-NEXT: call { double, double } @[[activeMod:.+]](

;CHECK:define internal { double, double } @[[active]](i32 %m, i32 %n, double %alpha, double* %a, double* %"a'", i32 %lda, double* %x, double* %"x'", i32 %incx, double %beta, double* %y, double* %"y'", i32 %incy)
;CHECK-NEXT:entry:
;CHECK-NEXT:  %0 = zext i32 %m to i64
;CHECK-NEXT:  %mallocsize = mul i64 %0, ptrtoint (double* getelementptr (double, double* null, i32 1) to i64)
;CHECK-NEXT:  %malloccall = tail call i8* @malloc(i64 %mallocsize)
;CHECK-NEXT:  %1 = bitcast i8* %malloccall to double*
;CHECK-NEXT:  tail call void @cblas_dgemv(i32 102, i32 111, i32 %m, i32 %n, double %alpha, double* %a, i32 %lda, double* %x, i32 %incx, double %beta, double* %y, i32 %incy)
;CHECK-NEXT:  call void @cblas_dscal(i32 %n, double %beta, double* %"y'", i32 %incy)
;CHECK-NEXT:  call void @cblas_dgemv(i32 102, i32 112, i32 %n, i32 %m, double %alpha, double* %a, i32 %lda, double* %"y'", i32 %incy, double 1.000000e+00, double* %"x'", i32 %incx)
;CHECK-NEXT:  call void @cblas_dger(i32 102, i32 %m, i32 %n, double %alpha, double* %"y'", i32 %incy, double* %"x'", i32 %incx, double* %a, i32 %lda)
;CHECK-NEXT:  call void @cblas_dgemv(i32 102, i32 111, i32 %m, i32 %n, double 1.000000e+00, double* %a, i32 %lda, double* %x, i32 %incx, double 0.000000e+00, double* %1, i32 1)
;CHECK-NEXT:  %2 = call fast double @cblas_ddot(i32 %m, double* nocapture readonly %"y'", i32 %incy, double* nocapture readonly %1, i32 1)
;CHECK-NEXT:  %3 = call fast double @cblas_ddot(i32 %m, double* nocapture readonly %"y'", i32 %incy, double* nocapture readonly %y, i32 %incy)
;CHECK-NEXT:  %4 = insertvalue { double, double } undef, double %2, 0
;CHECK-NEXT:  %5 = insertvalue { double, double } %4, double %3, 1
;CHECK-NEXT:  ret { double, double } %5
;CHECK-NEXT:}

;CHECK:define internal { double, double } @[[activeMod]](i32 %m, i32 %n, double %alpha, double* %a, double* %"a'", i32 %lda, double* %x, double* %"x'", i32 %incx, double %beta, double* %y, double* %"y'", i32 %incy)
;CHECK-NEXT:entry:
;CHECK-NEXT:  %0 = zext i32 %n to i64
;CHECK-NEXT:  %mallocsize = mul i64 %0, ptrtoint (double* getelementptr (double, double* null, i32 1) to i64)
;CHECK-NEXT:  %malloccall = tail call i8* @malloc(i64 %mallocsize)
;CHECK-NEXT:  %1 = bitcast i8* %malloccall to double*
;CHECK-NEXT:  call void @__enzyme_memcpy_doubleda0sa0stride(double* %1, double* %x, i32 %n, i32 %incx)
;CHECK-NEXT:  %2 = zext i32 %m to i64
;CHECK-NEXT:  %mallocsize1 = mul i64 %2, ptrtoint (double* getelementptr (double, double* null, i32 1) to i64)
;CHECK-NEXT:  %malloccall2 = tail call i8* @malloc(i64 %mallocsize1)
;CHECK-NEXT:  %3 = bitcast i8* %malloccall2 to double*
;CHECK-NEXT:  call void @__enzyme_memcpy_doubleda0sa0stride(double* %3, double* %y, i32 %m, i32 %incy)
;CHECK-NEXT:  %4 = mul i32 %m, %n
;CHECK-NEXT:  %5 = zext i32 %4 to i64
;CHECK-NEXT:  %mallocsize3 = mul i64 %5, ptrtoint (double* getelementptr (double, double* null, i32 1) to i64)
;CHECK-NEXT:  %malloccall4 = tail call i8* @malloc(i64 %mallocsize3)
;CHECK-NEXT:  %6 = bitcast i8* %malloccall4 to double*
;CHECK-NEXT:  call void @__enzyme_memcpy_doubleda0sa0matrix(double* %6, double* %a, i32 %m, i32 %n, i32 %lda, i32 102)
;CHECK-NEXT:  %7 = zext i32 %m to i64
;CHECK-NEXT:  %mallocsize5 = mul i64 %7, ptrtoint (double* getelementptr (double, double* null, i32 1) to i64)
;CHECK-NEXT:  %malloccall6 = tail call i8* @malloc(i64 %mallocsize5)
;CHECK-NEXT:  %8 = bitcast i8* %malloccall6 to double*
;CHECK-NEXT:  tail call void @cblas_dgemv(i32 102, i32 111, i32 %m, i32 %n, double %alpha, double* %a, i32 %lda, double* %x, i32 %incx, double %beta, double* %y, i32 %incy)
;CHECK-NEXT:  store double 0.000000e+00, double* %a, align 8
;CHECK-NEXT:  store double 0.000000e+00, double* %"a'", align 8
;CHECK-NEXT:  call void @cblas_dscal(i32 %n, double %beta, double* %"y'", i32 1)
;CHECK-NEXT:  call void @cblas_dgemv(i32 102, i32 112, i32 %n, i32 %m, double %alpha, double* %6, i32 %lda, double* %"y'", i32 1, double 1.000000e+00, double* %"x'", i32 1)
;CHECK-NEXT:  call void @cblas_dger(i32 102, i32 %m, i32 %n, double %alpha, double* %"y'", i32 1, double* %"x'", i32 1, double* %6, i32 %lda)
;CHECK-NEXT:  call void @cblas_dgemv(i32 102, i32 111, i32 %m, i32 %n, double 1.000000e+00, double* %6, i32 %lda, double* %1, i32 1, double 0.000000e+00, double* %8, i32 1)
;CHECK-NEXT:  %9 = call fast double @cblas_ddot(i32 %m, double* nocapture readonly %"y'", i32 1, double* nocapture readonly %8, i32 1)
;CHECK-NEXT:  %10 = call fast double @cblas_ddot(i32 %m, double* nocapture readonly %"y'", i32 1, double* nocapture readonly %3, i32 1)
;CHECK-NEXT:  %11 = insertvalue { double, double } undef, double %9, 0
;CHECK-NEXT:  %12 = insertvalue { double, double } %11, double %10, 1
;CHECK-NEXT:  ret { double, double } %12
;CHECK-NEXT:}
