;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

define void @wrapper(i32 %n, double* %x, i32 %incx, double* %y, i32 %incy) {
entry:
  tail call void @cblas_dcopy(i32 %n, double* %x, i32 %incx, double* %y, i32 %incy)
  ret void
}

declare void @cblas_dcopy(i32, double*, i32, double*, i32)

define void @wrapperMod(i32 %n, double* %x, i32 %incx, double* %y, i32 %incy) {
entry:
  tail call void @cblas_dcopy(i32 %n, double* %x, i32 %incx, double* %y, i32 %incy)
  store double 0.000000e+00, double* %x, align 8
  store double 1.000000e+00, double* %y, align 8
  ret void
}

define void @active(i32 %n, double* %x, double* %_x, i32 %incx, double* %y, double* %_y, i32 %incy) {
entry:
  tail call void (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (i32, double*, i32, double*, i32)* @wrapper to i8*), i32 %n, double* %x, double* %_x, i32 %incx, double* %y, double* %_y, i32 %incy)
  ret void
}

declare void @__enzyme_autodiff(i8*, ...)

define void @activeMod(i32 %n, double* %x, double* %_x, i32 %incx, double* %y, double* %_y, i32 %incy) {
entry:
  tail call void (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (i32, double*, i32, double*, i32)* @wrapperMod to i8*), i32 %n, double* %x, double* %_x, i32 %incx, double* %y, double* %_y, i32 %incy)
  ret void
}

;CHECK: define void @active
;CHECK-NEXT: entry
;CHECK-NEXT: call void @[[active:.+]](

;CHECK: define void @activeMod
;CHECK-NEXT: entry
;CHECK-NEXT: call void @[[activeMod:.+]](

;CHECK:define internal void @[[active]](i32 %n, double* %x, double* %"x'", i32 %incx, double* %y, double* %"y'", i32 %incy)
;CHECK-NEXT:entry:
;CHECK-NEXT:  tail call void @cblas_dcopy(i32 %n, double* %x, i32 %incx, double* %y, i32 %incy)
;CHECK-NEXT:  call void @cblas_daxpy(i32 %n, double 1.000000e+00, double* %"y'", i32 %incy, double* %"x'", i32 %incx)
;CHECK-NEXT:  ret void
;CHECK-NEXT:}

;CHECK:define internal void @[[activeMod]](i32 %n, double* %x, double* %"x'", i32 %incx, double* %y, double* %"y'", i32 %incy)
;CHECK-NEXT:entry:
;CHECK-NEXT:  tail call void @cblas_dcopy(i32 %n, double* %x, i32 %incx, double* %y, i32 %incy)
;CHECK-NEXT:  store double 0.000000e+00, double* %x, align 8
;CHECK-NEXT:  store double 1.000000e+00, double* %y, align 8
;CHECK-NEXT:  store double 0.000000e+00, double* %"y'", align 8
;CHECK-NEXT:  store double 0.000000e+00, double* %"x'", align 8
;CHECK-NEXT:  call void @cblas_daxpy(i32 %n, double 1.000000e+00, double* %"y'", i32 %incy, double* %"x'", i32 %incx)
;CHECK-NEXT:  ret void
;CHECK-NEXT:}
