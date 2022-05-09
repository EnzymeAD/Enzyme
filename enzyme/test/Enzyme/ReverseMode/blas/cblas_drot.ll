;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

define void @wrapper(i32 %n, double* %x, i32 %incx, double* %y, i32 %incy, double %c, double %s) {
entry:
  tail call void @cblas_drot(i32 %n, double* %x, i32 %incx, double* %y, i32 %incy, double %c, double %s)
  ret void
}

declare void @cblas_drot(i32, double*, i32, double*, i32, double, double)

define void @wrapperMod(i32 %n, double* %x, i32 %incx, double* %y, i32 %incy, double %c, double %s) {
entry:
  tail call void @cblas_drot(i32 %n, double* %x, i32 %incx, double* %y, i32 %incy, double %c, double %s)
  store double 0.000000e+00, double* %x, align 8
  ret void
}

define void @active(i32 %n, double* %x, double* %_x, i32 %incx, double* %y, double* %_y, i32 %incy, double %c, double %s) {
entry:
  %call = tail call double @__enzyme_autodiff(i8* bitcast (void (i32, double*, i32, double*, i32, double, double)* @wrapper to i8*), i32 %n, double* %x, double* %_x, i32 %incx, double* %y, double* %_y, i32 %incy, double %c, double %s)
  ret void
}

declare double @__enzyme_autodiff(i8*, i32, double*, double*, i32, double*, double*, i32, double, double)

define void @activeMod(i32 %n, double* %x, double* %_x, i32 %incx, double* %y, double* %_y, i32 %incy, double %c, double %s) {
entry:
  %call = tail call double @__enzyme_autodiff(i8* bitcast (void (i32, double*, i32, double*, i32, double, double)* @wrapperMod to i8*), i32 %n, double* %x, double* %_x, i32 %incx, double* %y, double* %_y, i32 %incy, double %c, double %s)
  ret void
}

;CHECK: define void @active
;CHECK-NEXT: entry
;CHECK-NEXT: call { double, double } @[[active:.+]](

;CHECK: define void @activeMod
;CHECK-NEXT: entry
;CHECK-NEXT: call { double, double } @[[activeMod:.+]](

;CHECK:define internal { double, double } @[[active]](i32 %n, double* %x, double* %"x'", i32 %incx, double* %y, double* %"y'", i32 %incy, double %c, double %s)
;CHECK-NEXT:entry:
;CHECK-NEXT:  tail call void @cblas_drot(i32 %n, double* %x, i32 %incx, double* %y, i32 %incy, double %c, double %s)
;CHECK-NEXT:  call void @cblas_drot(i32 %n, double* %"y'", i32 %incy, double* %"x'", i32 %incx, double %c, double %s)
;CHECK-NEXT:  %0 = call fast double @cblas_ddot(i32 %n, double* nocapture readonly %"y'", i32 %incy, double* nocapture readonly %y, i32 %incy)
;CHECK-NEXT:  %1 = call fast double @cblas_ddot(i32 %n, double* nocapture readonly %"x'", i32 %incx, double* nocapture readonly %x, i32 %incx)
;CHECK-NEXT:  %2 = fadd fast double %0, %1
;CHECK-NEXT:  %3 = call fast double @cblas_ddot(i32 %n, double* nocapture readonly %"y'", i32 %incy, double* nocapture readonly %x, i32 %incx)
;CHECK-NEXT:  %4 = call fast double @cblas_ddot(i32 %n, double* nocapture readonly %"x'", i32 %incx, double* nocapture readonly %y, i32 %incy)
;CHECK-NEXT:  %5 = fadd fast double %3, %4
;CHECK-NEXT:  %6 = insertvalue { double, double } undef, double %2, 0
;CHECK-NEXT:  %7 = insertvalue { double, double } %6, double %5, 1
;CHECK-NEXT:  ret { double, double } %7
;CHECK-NEXT:}

;CHECK:define internal { double, double } @[[activeMod]](i32 %n, double* %x, double* %"x'", i32 %incx, double* %y, double* %"y'", i32 %incy, double %c, double %s)
;CHECK-NEXT:entry:
;CHECK-NEXT:  %0 = zext i32 %n to i64
;CHECK-NEXT:  %mallocsize = mul i64 %0, ptrtoint (double* getelementptr (double, double* null, i32 1) to i64)
;CHECK-NEXT:  %malloccall = tail call i8* @malloc(i64 %mallocsize)
;CHECK-NEXT:  %1 = bitcast i8* %malloccall to double*
;CHECK-NEXT:  call void @__enzyme_memcpy_doubleda0sa0stride(double* %1, double* %x, i32 %n, i32 %incx)
;CHECK-NEXT:  %2 = zext i32 %n to i64
;CHECK-NEXT:  %mallocsize1 = mul i64 %2, ptrtoint (double* getelementptr (double, double* null, i32 1) to i64)
;CHECK-NEXT:  %malloccall2 = tail call i8* @malloc(i64 %mallocsize1)
;CHECK-NEXT:  %3 = bitcast i8* %malloccall2 to double*
;CHECK-NEXT:  call void @__enzyme_memcpy_doubleda0sa0stride(double* %3, double* %y, i32 %n, i32 %incy)
;CHECK-NEXT:  tail call void @cblas_drot(i32 %n, double* %x, i32 %incx, double* %y, i32 %incy, double %c, double %s)
;CHECK-NEXT:  store double 0.000000e+00, double* %x, align 8
;CHECK-NEXT:  store double 0.000000e+00, double* %"x'", align 8
;CHECK-NEXT:  call void @cblas_drot(i32 %n, double* %"y'", i32 1, double* %"x'", i32 1, double %c, double %s)
;CHECK-NEXT:  %4 = call fast double @cblas_ddot(i32 %n, double* nocapture readonly %"y'", i32 1, double* nocapture readonly %3, i32 1)
;CHECK-NEXT:  %5 = call fast double @cblas_ddot(i32 %n, double* nocapture readonly %"x'", i32 1, double* nocapture readonly %1, i32 1)
;CHECK-NEXT:  %6 = fadd fast double %4, %5
;CHECK-NEXT:  %7 = call fast double @cblas_ddot(i32 %n, double* nocapture readonly %"y'", i32 1, double* nocapture readonly %1, i32 1)
;CHECK-NEXT:  %8 = call fast double @cblas_ddot(i32 %n, double* nocapture readonly %"x'", i32 1, double* nocapture readonly %3, i32 1)
;CHECK-NEXT:  %9 = fadd fast double %7, %8
;CHECK-NEXT:  %10 = insertvalue { double, double } undef, double %6, 0
;CHECK-NEXT:  %11 = insertvalue { double, double } %10, double %9, 1
;CHECK-NEXT:  ret { double, double } %11
;CHECK-NEXT:}
