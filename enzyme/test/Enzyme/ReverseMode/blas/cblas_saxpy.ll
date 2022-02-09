;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

@enzyme_const = common global i32 0, align 4

define void @wrapper(i32 %n, float %alpha, float* %x, i32 %incx, float* %y, i32 %incy) {
entry:
  tail call void @cblas_saxpy(i32 %n, float %alpha, float* %x, i32 %incx, float* %y, i32 %incy)
  ret void
}

declare void @cblas_saxpy(i32, float, float*, i32, float*, i32)

define void @wrapperMod(i32 %n, float %alpha, float* %x, i32 %incx, float* %y, i32 %incy) {
entry:
  tail call void @cblas_saxpy(i32 %n, float %alpha, float* %x, i32 %incx, float* %y, i32 %incy)
  store float 0.000000e+00, float* %x, align 4
  ret void
}

define void @active(i32 %n, float %alpha, float* %x, float* %_x, i32 %incx, float* %y, float* %_y, i32 %incy) {
entry:
  %call = tail call float @__enzyme_autodiff(i8* bitcast (void (i32, float, float*, i32, float*, i32)* @wrapper to i8*), i32 %n, float %alpha, float* %x, float* %_x, i32 %incx, float* %y, float* %_y, i32 %incy)
  ret void
}

declare float @__enzyme_autodiff(i8*, i32, float, float*, float*, i32, float*, float*, i32)

define void @inactive_alpha(i32 %n, float %alpha, float* %x, float* %_x, i32 %incx, float* %y, float* %_y, i32 %incy) {
entry:
  %0 = load i32, i32* @enzyme_const, align 4
  tail call void @__enzyme_autodiff1(i8* bitcast (void (i32, float, float*, i32, float*, i32)* @wrapper to i8*), i32 %n, i32 %0, float %alpha, float* %x, float* %_x, i32 %incx, float* %y, float* %_y, i32 %incy)
  ret void
}

declare void @__enzyme_autodiff1(i8*, i32, i32, float, float*, float*, i32, float*, float*, i32)

define void @activeMod(i32 %n, float %alpha, float* %x, float* %_x, i32 %incx, float* %y, float* %_y, i32 %incy) {
entry:
  %call = tail call float @__enzyme_autodiff(i8* bitcast (void (i32, float, float*, i32, float*, i32)* @wrapperMod to i8*), i32 %n, float %alpha, float* %x, float* %_x, i32 %incx, float* %y, float* %_y, i32 %incy)
  ret void
}

;CHECK: define void @active
;CHECK-NEXT: entry
;CHECK-NEXT: call { float } @[[active:.+]](

;CHECK: define void @inactive_alpha
;CHECK-NEXT: entry
;CHECK-NEXT: call void @[[inactive_alpha:.+]](

;CHECK: define void @activeMod
;CHECK-NEXT: entry
;CHECK-NEXT: call { float } @[[activeMod:.+]](

;CHECK:define internal { float } @[[active]](i32 %n, float %alpha, float* %x, float* %"x'", i32 %incx, float* %y, float* %"y'", i32 %incy)
;CHECK-NEXT:entry:
;CHECK-NEXT:  tail call void @cblas_saxpy(i32 %n, float %alpha, float* %x, i32 %incx, float* %y, i32 %incy)
;CHECK-NEXT:  call void @cblas_saxpy(i32 %n, float %alpha, float* %"y'", i32 %incy, float* %"x'", i32 %incx)
;CHECK-NEXT:  %0 = call fast float @cblas_sdot(i32 %n, float* nocapture readonly %"y'", i32 %incy, float* nocapture readonly %x, i32 %incx)
;CHECK-NEXT:  %1 = insertvalue { float } undef, float %0, 0
;CHECK-NEXT:  ret { float } %1
;CHECK-NEXT:}

;CHECK:define internal void @[[inactive_alpha]](i32 %n, float %alpha, float* %x, float* %"x'", i32 %incx, float* %y, float* %"y'", i32 %incy)
;CHECK-NEXT:entry:
;CHECK-NEXT:  tail call void @cblas_saxpy(i32 %n, float %alpha, float* %x, i32 %incx, float* %y, i32 %incy)
;CHECK-NEXT:  call void @cblas_saxpy(i32 %n, float %alpha, float* %"y'", i32 %incy, float* %"x'", i32 %incx)
;CHECK-NEXT:  ret void
;CHECK-NEXT:}

;CHECK:define internal { float } @[[activeMod]](i32 %n, float %alpha, float* %x, float* %"x'", i32 %incx, float* %y, float* %"y'", i32 %incy)
;CHECK-NEXT:entry:
;CHECK-NEXT:  %0 = zext i32 %n to i64
;CHECK-NEXT:  %mallocsize = mul i64 %0, ptrtoint (float* getelementptr (float, float* null, i32 1) to i64)
;CHECK-NEXT:  %malloccall = tail call i8* @malloc(i64 %mallocsize)
;CHECK-NEXT:  %1 = bitcast i8* %malloccall to float*
;CHECK-NEXT:  call void @__enzyme_memcpy_floatda0sa0stride(float* %1, float* %x, i32 %n, i32 %incx)
;CHECK-NEXT:  tail call void @cblas_saxpy(i32 %n, float %alpha, float* %x, i32 %incx, float* %y, i32 %incy)
;CHECK-NEXT:  store float 0.000000e+00, float* %x, align 4
;CHECK-NEXT:  store float 0.000000e+00, float* %"x'", align 4
;CHECK-NEXT:  call void @cblas_saxpy(i32 %n, float %alpha, float* %"y'", i32 %incy, float* %"x'", i32 1)
;CHECK-NEXT:  %2 = call fast float @cblas_sdot(i32 %n, float* nocapture readonly %"y'", i32 %incy, float* nocapture readonly %1, i32 1)
;CHECK-NEXT:  %3 = insertvalue { float } undef, float %2, 0
;CHECK-NEXT:  ret { float } %3
;CHECK-NEXT:}
