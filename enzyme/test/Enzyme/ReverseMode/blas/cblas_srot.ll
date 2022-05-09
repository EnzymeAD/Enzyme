;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

define void @wrapper(i32 %n, float* %x, i32 %incx, float* %y, i32 %incy, float %c, float %s) {
entry:
  tail call void @cblas_srot(i32 %n, float* %x, i32 %incx, float* %y, i32 %incy, float %c, float %s)
  ret void
}

declare void @cblas_srot(i32, float*, i32, float*, i32, float, float)

define void @wrapperMod(i32 %n, float* %x, i32 %incx, float* %y, i32 %incy, float %c, float %s) {
entry:
  tail call void @cblas_srot(i32 %n, float* %x, i32 %incx, float* %y, i32 %incy, float %c, float %s)
  store float 0.000000e+00, float* %x, align 4
  ret void
}

define void @active(i32 %n, float* %x, float* %_x, i32 %incx, float* %y, float* %_y, i32 %incy, float %c, float %s) {
entry:
  %call = tail call float @__enzyme_autodiff(i8* bitcast (void (i32, float*, i32, float*, i32, float, float)* @wrapper to i8*), i32 %n, float* %x, float* %_x, i32 %incx, float* %y, float* %_y, i32 %incy, float %c, float %s)
  ret void
}

declare float @__enzyme_autodiff(i8*, i32, float*, float*, i32, float*, float*, i32, float, float)

define void @activeMod(i32 %n, float* %x, float* %_x, i32 %incx, float* %y, float* %_y, i32 %incy, float %c, float %s) {
entry:
  %call = tail call float @__enzyme_autodiff(i8* bitcast (void (i32, float*, i32, float*, i32, float, float)* @wrapperMod to i8*), i32 %n, float* %x, float* %_x, i32 %incx, float* %y, float* %_y, i32 %incy, float %c, float %s)
  ret void
}

;CHECK: define void @active
;CHECK-NEXT: entry
;CHECK-NEXT: call { float, float } @[[active:.+]](

;CHECK: define void @activeMod
;CHECK-NEXT: entry
;CHECK-NEXT: call { float, float } @[[activeMod:.+]](

;CHECK:define internal { float, float } @[[active]](i32 %n, float* %x, float* %"x'", i32 %incx, float* %y, float* %"y'", i32 %incy, float %c, float %s)
;CHECK-NEXT:entry:
;CHECK-NEXT:  tail call void @cblas_srot(i32 %n, float* %x, i32 %incx, float* %y, i32 %incy, float %c, float %s)
;CHECK-NEXT:  call void @cblas_srot(i32 %n, float* %"y'", i32 %incy, float* %"x'", i32 %incx, float %c, float %s)
;CHECK-NEXT:  %0 = call fast float @cblas_sdot(i32 %n, float* nocapture readonly %"y'", i32 %incy, float* nocapture readonly %y, i32 %incy)
;CHECK-NEXT:  %1 = call fast float @cblas_sdot(i32 %n, float* nocapture readonly %"x'", i32 %incx, float* nocapture readonly %x, i32 %incx)
;CHECK-NEXT:  %2 = fadd fast float %0, %1
;CHECK-NEXT:  %3 = call fast float @cblas_sdot(i32 %n, float* nocapture readonly %"y'", i32 %incy, float* nocapture readonly %x, i32 %incx)
;CHECK-NEXT:  %4 = call fast float @cblas_sdot(i32 %n, float* nocapture readonly %"x'", i32 %incx, float* nocapture readonly %y, i32 %incy)
;CHECK-NEXT:  %5 = fadd fast float %3, %4
;CHECK-NEXT:  %6 = insertvalue { float, float } undef, float %2, 0
;CHECK-NEXT:  %7 = insertvalue { float, float } %6, float %5, 1
;CHECK-NEXT:  ret { float, float } %7
;CHECK-NEXT:}

;CHECK:define internal { float, float } @[[activeMod]](i32 %n, float* %x, float* %"x'", i32 %incx, float* %y, float* %"y'", i32 %incy, float %c, float %s)
;CHECK-NEXT:entry:
;CHECK-NEXT:  %0 = zext i32 %n to i64
;CHECK-NEXT:  %mallocsize = mul i64 %0, ptrtoint (float* getelementptr (float, float* null, i32 1) to i64)
;CHECK-NEXT:  %malloccall = tail call i8* @malloc(i64 %mallocsize)
;CHECK-NEXT:  %1 = bitcast i8* %malloccall to float*
;CHECK-NEXT:  call void @__enzyme_memcpy_floatda0sa0stride(float* %1, float* %x, i32 %n, i32 %incx)
;CHECK-NEXT:  %2 = zext i32 %n to i64
;CHECK-NEXT:  %mallocsize1 = mul i64 %2, ptrtoint (float* getelementptr (float, float* null, i32 1) to i64)
;CHECK-NEXT:  %malloccall2 = tail call i8* @malloc(i64 %mallocsize1)
;CHECK-NEXT:  %3 = bitcast i8* %malloccall2 to float*
;CHECK-NEXT:  call void @__enzyme_memcpy_floatda0sa0stride(float* %3, float* %y, i32 %n, i32 %incy)
;CHECK-NEXT:  tail call void @cblas_srot(i32 %n, float* %x, i32 %incx, float* %y, i32 %incy, float %c, float %s)
;CHECK-NEXT:  store float 0.000000e+00, float* %x, align 4
;CHECK-NEXT:  store float 0.000000e+00, float* %"x'", align 4
;CHECK-NEXT:  call void @cblas_srot(i32 %n, float* %"y'", i32 1, float* %"x'", i32 1, float %c, float %s)
;CHECK-NEXT:  %4 = call fast float @cblas_sdot(i32 %n, float* nocapture readonly %"y'", i32 1, float* nocapture readonly %3, i32 1)
;CHECK-NEXT:  %5 = call fast float @cblas_sdot(i32 %n, float* nocapture readonly %"x'", i32 1, float* nocapture readonly %1, i32 1)
;CHECK-NEXT:  %6 = fadd fast float %4, %5
;CHECK-NEXT:  %7 = call fast float @cblas_sdot(i32 %n, float* nocapture readonly %"y'", i32 1, float* nocapture readonly %1, i32 1)
;CHECK-NEXT:  %8 = call fast float @cblas_sdot(i32 %n, float* nocapture readonly %"x'", i32 1, float* nocapture readonly %3, i32 1)
;CHECK-NEXT:  %9 = fadd fast float %7, %8
;CHECK-NEXT:  %10 = insertvalue { float, float } undef, float %6, 0
;CHECK-NEXT:  %11 = insertvalue { float, float } %10, float %9, 1
;CHECK-NEXT:  ret { float, float } %11
;CHECK-NEXT:}
