;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

define void @wrapper(i32 %n, float %alpha, float* %x, i32 %incx, float* %y, i32 %incy, float* %a, i32 %lda) {
entry:
  tail call void @cblas_ssyr2(i32 102, i32 122, i32 %n, float %alpha, float* %x, i32 %incx, float* %y, i32 %incy, float* %a, i32 %lda)
  ret void
}

declare void @cblas_ssyr2(i32, i32, i32, float, float*, i32, float*, i32, float*, i32)

define void @active(i32 %n, float %alpha, i32 %lda, float* %x, float* %_x, i32 %incx, float* %y, float* %_y, i32 %incy, float* %a, float* %_a) {
entry:
  tail call void @__enzyme_autodiff(i8* bitcast (void (i32, float, float*, i32, float*, i32, float*, i32)* @wrapper to i8*), i32 %n, float %alpha, float* %x, float* %_x, i32 %incx, float* %y, float* %_y, i32 %incy, float* %a, float* %_a, i32 %lda)
  ret void
}

declare void @__enzyme_autodiff(i8*, i32, float, float*, float*, i32, float*, float*, i32, float*, float*, i32)

;CHECK: define void @active
;CHECK-NEXT: entry
;CHECK-NEXT: call { float } @[[active:.+]](

;CHECK:define internal { float } @[[active]](i32 %n, float %alpha, float* %x, float* %"x'", i32 %incx, float* %y, float* %"y'", i32 %incy, float* %a, float* %"a'", i32 %lda)
;CHECK-NEXT:entry:
;CHECK-NEXT:  %0 = zext i32 %n to i64
;CHECK-NEXT:  %mallocsize = mul i64 %0, ptrtoint (float* getelementptr (float, float* null, i32 1) to i64)
;CHECK-NEXT:  %malloccall = tail call i8* @malloc(i64 %mallocsize)
;CHECK-NEXT:  %1 = bitcast i8* %malloccall to float*
;CHECK-NEXT:  tail call void @cblas_ssyr2(i32 102, i32 122, i32 %n, float %alpha, float* %x, i32 %incx, float* %y, i32 %incy, float* %a, i32 %lda)
;CHECK-NEXT:  %2 = fadd fast float 2.000000e+00, %alpha
;CHECK-NEXT:  call void @cblas_ssymv(i32 102, i32 122, i32 %n, float %2, float* %"a'", i32 %lda, float* %x, i32 %incx, float 1.000000e+00, float* %"y'", i32 %incy)
;CHECK-NEXT:  %3 = fadd fast float 2.000000e+00, %alpha
;CHECK-NEXT:  call void @cblas_ssymv(i32 102, i32 122, i32 %n, float %3, float* %"a'", i32 %lda, float* %y, i32 %incy, float 1.000000e+00, float* %"y'", i32 %incx)
;CHECK-NEXT:  call void @cblas_ssymv(i32 102, i32 122, i32 %n, float 2.000000e+00, float* %"a'", i32 %lda, float* %y, i32 %incy, float 0.000000e+00, float* %1, i32 1)
;CHECK-NEXT:  %4 = call fast float @cblas_sdot(i32 %n, float* nocapture readonly %x, i32 %incy, float* nocapture readonly %1, i32 1)
;CHECK-NEXT:  %5 = insertvalue { float } undef, float %4, 0
;CHECK-NEXT:  ret { float } %5
;CHECK-NEXT:}
