;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

define void @wrapper(i32 %n, float* %a, i32 %lda, float* %x, i32 %incx) {
entry:
  tail call void @cblas_strsv(i32 102, i32 121, i32 111, i32 131, i32 %n, float* %a, i32 %lda, float* %x, i32 %incx)
  ret void
}

declare void @cblas_strsv(i32, i32, i32, i32, i32, float*, i32, float*, i32)

define void @active(i32 %n, float* %a, float* %_a, i32 %lda, float* %x, float* %_x, i32 %incx) {
entry:
  tail call void @__enzyme_autodiff(i8* bitcast (void (i32, float*, i32, float*, i32)* @wrapper to i8*), i32 %n, float* %a, float* %_a, i32 %lda, float* %x, float* %_x, i32 %incx)
  ret void
}

declare void @__enzyme_autodiff(i8*, i32, float*, float*, i32, float*, float*, i32)

;CHECK: define void @active
;CHECK-NEXT: entry
;CHECK-NEXT: call void @[[active:.+]](

;CHECK:define internal void @[[active]](i32 %n, float* %a, float* %"a'", i32 %lda, float* %x, float* %"x'", i32 %incx)
;CHECK-NEXT:entry:
;CHECK-NEXT:  tail call void @cblas_strsv(i32 102, i32 121, i32 111, i32 131, i32 %n, float* %a, i32 %lda, float* %x, i32 %incx)
;CHECK-NEXT:  call void @cblas_strsv(i32 102, i32 121, i32 111, i32 131, i32 %n, float* %a, i32 %lda, float* %"x'", i32 %incx)
;CHECK-NEXT:  call void @cblas_sger(i32 102, i32 %n, i32 %n, float 1.000000e+00, float* %x, i32 %incx, float* %"x'", i32 %incx, float* %a, i32 %lda)
;CHECK-NEXT:  ret void
;CHECK-NEXT:}
