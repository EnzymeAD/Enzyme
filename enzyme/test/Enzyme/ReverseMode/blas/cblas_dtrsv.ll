;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

define void @wrapper(i32 %n, double* %a, i32 %lda, double* %x, i32 %incx) {
entry:
  tail call void @cblas_dtrsv(i32 102, i32 121, i32 111, i32 131, i32 %n, double* %a, i32 %lda, double* %x, i32 %incx)
  ret void
}

declare void @cblas_dtrsv(i32, i32, i32, i32, i32, double*, i32, double*, i32)

define void @active(i32 %n, double* %a, double* %_a, i32 %lda, double* %x, double* %_x, i32 %incx) {
entry:
  tail call void @__enzyme_autodiff(i8* bitcast (void (i32, double*, i32, double*, i32)* @wrapper to i8*), i32 %n, double* %a, double* %_a, i32 %lda, double* %x, double* %_x, i32 %incx)
  ret void
}

declare void @__enzyme_autodiff(i8*, i32, double*, double*, i32, double*, double*, i32)

;CHECK: define void @active
;CHECK-NEXT: entry
;CHECK-NEXT: call void @[[active:.+]](

;CHECK:define internal void @[[active]](i32 %n, double* %a, double* %"a'", i32 %lda, double* %x, double* %"x'", i32 %incx)
;CHECK-NEXT:entry:
;CHECK-NEXT:  tail call void @cblas_dtrsv(i32 102, i32 121, i32 111, i32 131, i32 %n, double* %a, i32 %lda, double* %x, i32 %incx)
;CHECK-NEXT:  call void @cblas_dtrsv(i32 102, i32 121, i32 111, i32 131, i32 %n, double* %a, i32 %lda, double* %"x'", i32 %incx)
;CHECK-NEXT:  call void @cblas_dger(i32 102, i32 %n, i32 %n, double 1.000000e+00, double* %x, i32 %incx, double* %"x'", i32 %incx, double* %a, i32 %lda)
;CHECK-NEXT:  ret void
;CHECK-NEXT:}
