;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

define void @wrapper(i32 %m, i32 %n, float %alpha, float* %a, i32 %lda, float* %x, i32 %incx, float %beta, float* %y, i32 %incy) {
entry:
  tail call void @cblas_sgemv(i32 102, i32 111, i32 %m, i32 %n, float %alpha, float* %a, i32 %lda, float* %x, i32 %incx, float %beta, float* %y, i32 %incy)
  ret void
}

declare void @cblas_sgemv(i32, i32, i32, i32, float, float*, i32, float*, i32, float, float*, i32)

define void @wrapperMod(i32 %m, i32 %n, float %alpha, float* %a, i32 %lda, float* %x, i32 %incx, float %beta, float* %y, i32 %incy) {
entry:
  tail call void @cblas_sgemv(i32 102, i32 111, i32 %m, i32 %n, float %alpha, float* %a, i32 %lda, float* %x, i32 %incx, float %beta, float* %y, i32 %incy)
  store float 0.000000e+00, float* %a, align 4
  ret void
}

define void @active(i32 %m, i32 %n, float %alpha, float* %a, float* %_a, i32 %lda, float* %x, float* %_x, i32 %incx, float %beta, float* %y, float* %_y, i32 %incy) {
entry:
  tail call void @__enzyme_autodiff(i8* bitcast (void (i32, i32, float, float*, i32, float*, i32, float, float*, i32)* @wrapper to i8*), i32 %m, i32 %n, float %alpha, float* %a, float* %_a, i32 %lda, float* %x, float* %_x, i32 %incx, float %beta, float* %y, float* %_y, i32 %incy)
  ret void
}

declare void @__enzyme_autodiff(i8*, i32, i32, float, float*, float*, i32, float*, float*, i32, float, float*, float*, i32)

define void @activeMod(i32 %m, i32 %n, float %alpha, float* %a, float* %_a, i32 %lda, float* %x, float* %_x, i32 %incx, float %beta, float* %y, float* %_y, i32 %incy) {
entry:
  tail call void @__enzyme_autodiff(i8* bitcast (void (i32, i32, float, float*, i32, float*, i32, float, float*, i32)* @wrapperMod to i8*), i32 %m, i32 %n, float %alpha, float* %a, float* %_a, i32 %lda, float* %x, float* %_x, i32 %incx, float %beta, float* %y, float* %_y, i32 %incy)
  ret void
}

;CHECK: define void @active
;CHECK-NEXT: entry
;CHECK-NEXT: call { float, float } @[[active:.+]](

;CHECK: define void @activeMod
;CHECK-NEXT: entry
;CHECK-NEXT: call { float, float } @[[activeMod:.+]](

;CHECK:define internal { float, float } @[[active]](i32 %m, i32 %n, float %alpha, float* %a, float* %"a'", i32 %lda, float* %x, float* %"x'", i32 %incx, float %beta, float* %y, float* %"y'", i32 %incy)
;CHECK-NEXT:entry:
;CHECK-NEXT:  %0 = zext i32 %m to i64
;CHECK-NEXT:  %mallocsize = mul i64 %0, ptrtoint (float* getelementptr (float, float* null, i32 1) to i64)
;CHECK-NEXT:  %malloccall = tail call i8* @malloc(i64 %mallocsize)
;CHECK-NEXT:  %1 = bitcast i8* %malloccall to float*
;CHECK-NEXT:  tail call void @cblas_sgemv(i32 102, i32 111, i32 %m, i32 %n, float %alpha, float* %a, i32 %lda, float* %x, i32 %incx, float %beta, float* %y, i32 %incy)
;CHECK-NEXT:  call void @cblas_sscal(i32 %n, float %beta, float* %"y'", i32 %incy)
;CHECK-NEXT:  call void @cblas_sgemv(i32 102, i32 112, i32 %n, i32 %m, float %alpha, float* %a, i32 %lda, float* %"y'", i32 %incy, float 1.000000e+00, float* %"x'", i32 %incx)
;CHECK-NEXT:  call void @cblas_sger(i32 102, i32 %m, i32 %n, float %alpha, float* %"y'", i32 %incy, float* %"x'", i32 %incx, float* %a, i32 %lda)
;CHECK-NEXT:  call void @cblas_sgemv(i32 102, i32 111, i32 %m, i32 %n, float 1.000000e+00, float* %a, i32 %lda, float* %x, i32 %incx, float 0.000000e+00, float* %1, i32 1)
;CHECK-NEXT:  %2 = call fast float @cblas_sdot(i32 %m, float* nocapture readonly %"y'", i32 %incy, float* nocapture readonly %1, i32 1)
;CHECK-NEXT:  %3 = call fast float @cblas_sdot(i32 %m, float* nocapture readonly %"y'", i32 %incy, float* nocapture readonly %y, i32 %incy)
;CHECK-NEXT:  %4 = insertvalue { float, float } undef, float %2, 0
;CHECK-NEXT:  %5 = insertvalue { float, float } %4, float %3, 1
;CHECK-NEXT:  ret { float, float } %5
;CHECK-NEXT:}

;CHECK:define internal { float, float } @[[activeMod]](i32 %m, i32 %n, float %alpha, float* %a, float* %"a'", i32 %lda, float* %x, float* %"x'", i32 %incx, float %beta, float* %y, float* %"y'", i32 %incy)
;CHECK-NEXT:entry:
;CHECK-NEXT:  %0 = zext i32 %n to i64
;CHECK-NEXT:  %mallocsize = mul i64 %0, ptrtoint (float* getelementptr (float, float* null, i32 1) to i64)
;CHECK-NEXT:  %malloccall = tail call i8* @malloc(i64 %mallocsize)
;CHECK-NEXT:  %1 = bitcast i8* %malloccall to float*
;CHECK-NEXT:  call void @__enzyme_memcpy_floatda0sa0stride(float* %1, float* %x, i32 %n, i32 %incx)
;CHECK-NEXT:  %2 = zext i32 %m to i64
;CHECK-NEXT:  %mallocsize1 = mul i64 %2, ptrtoint (float* getelementptr (float, float* null, i32 1) to i64)
;CHECK-NEXT:  %malloccall2 = tail call i8* @malloc(i64 %mallocsize1)
;CHECK-NEXT:  %3 = bitcast i8* %malloccall2 to float*
;CHECK-NEXT:  call void @__enzyme_memcpy_floatda0sa0stride(float* %3, float* %y, i32 %m, i32 %incy)
;CHECK-NEXT:  %4 = mul i32 %m, %n
;CHECK-NEXT:  %5 = zext i32 %4 to i64
;CHECK-NEXT:  %mallocsize3 = mul i64 %5, ptrtoint (float* getelementptr (float, float* null, i32 1) to i64)
;CHECK-NEXT:  %malloccall4 = tail call i8* @malloc(i64 %mallocsize3)
;CHECK-NEXT:  %6 = bitcast i8* %malloccall4 to float*
;CHECK-NEXT:  call void @__enzyme_memcpy_floatda0sa0matrix(float* %6, float* %a, i32 %m, i32 %n, i32 %lda, i32 102)
;CHECK-NEXT:  %7 = zext i32 %m to i64
;CHECK-NEXT:  %mallocsize5 = mul i64 %7, ptrtoint (float* getelementptr (float, float* null, i32 1) to i64)
;CHECK-NEXT:  %malloccall6 = tail call i8* @malloc(i64 %mallocsize5)
;CHECK-NEXT:  %8 = bitcast i8* %malloccall6 to float*
;CHECK-NEXT:  tail call void @cblas_sgemv(i32 102, i32 111, i32 %m, i32 %n, float %alpha, float* %a, i32 %lda, float* %x, i32 %incx, float %beta, float* %y, i32 %incy)
;CHECK-NEXT:  store float 0.000000e+00, float* %a, align 4
;CHECK-NEXT:  store float 0.000000e+00, float* %"a'", align 4
;CHECK-NEXT:  call void @cblas_sscal(i32 %n, float %beta, float* %"y'", i32 1)
;CHECK-NEXT:  call void @cblas_sgemv(i32 102, i32 112, i32 %n, i32 %m, float %alpha, float* %6, i32 %lda, float* %"y'", i32 1, float 1.000000e+00, float* %"x'", i32 1)
;CHECK-NEXT:  call void @cblas_sger(i32 102, i32 %m, i32 %n, float %alpha, float* %"y'", i32 1, float* %"x'", i32 1, float* %6, i32 %lda)
;CHECK-NEXT:  call void @cblas_sgemv(i32 102, i32 111, i32 %m, i32 %n, float 1.000000e+00, float* %6, i32 %lda, float* %1, i32 1, float 0.000000e+00, float* %8, i32 1)
;CHECK-NEXT:  %9 = call fast float @cblas_sdot(i32 %m, float* nocapture readonly %"y'", i32 1, float* nocapture readonly %8, i32 1)
;CHECK-NEXT:  %10 = call fast float @cblas_sdot(i32 %m, float* nocapture readonly %"y'", i32 1, float* nocapture readonly %3, i32 1)
;CHECK-NEXT:  %11 = insertvalue { float, float } undef, float %9, 0
;CHECK-NEXT:  %12 = insertvalue { float, float } %11, float %10, 1
;CHECK-NEXT:  ret { float, float } %12
;CHECK-NEXT:}
