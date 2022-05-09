;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

define void @wrapper(i32 %n, i32 %k, float %alpha, float* %a, i32 %lda, float %beta, float* %c, i32 %ldc) {
entry:
  tail call void @cblas_ssyrk(i32 102, i32 122, i32 111, i32 %n, i32 %k, float %alpha, float* %a, i32 %lda, float %beta, float* %c, i32 %ldc)
  ret void
}

declare void @cblas_ssyrk(i32, i32, i32, i32, i32, float, float*, i32, float, float*, i32)

define void @active(i32 %n, i32 %k, float %alpha, float* %a, float* %_a, i32 %lda, float %beta, float* %c, float* %_c, i32 %ldc) {
entry:
  tail call void @__enzyme_autodiff(i8* bitcast (void (i32, i32, float, float*, i32, float, float*, i32)* @wrapper to i8*), i32 %n, i32 %k, float %alpha, float* %a, float* %_a, i32 %lda, float %beta, float* %c, float* %_c, i32 %ldc)
  ret void
}

declare void @__enzyme_autodiff(i8*, i32, i32, float, float*, float*, i32, float, float*, float*, i32)

;CHECK: define void @active
;CHECK-NEXT: entry
;CHECK-NEXT: call { float, float } @[[active:.+]](

;CHECK:define internal { float, float } @[[active]](i32 %n, i32 %k, float %alpha, float* %a, float* %"a'", i32 %lda, float %beta, float* %c, float* %"c'", i32 %ldc)
;CHECK-NEXT:entry:
;CHECK-NEXT:  tail call void @cblas_ssyrk(i32 102, i32 122, i32 111, i32 %n, i32 %k, float %alpha, float* %a, i32 %lda, float %beta, float* %c, i32 %ldc)
;CHECK-NEXT:  %0 = fmul fast float %alpha, 2.000000e+00
;CHECK-NEXT:  call void @cblas_ssymm(i32 102, i32 141, i32 122, i32 %n, i32 %k, float %0, float* %"c'", i32 %ldc, float* %a, i32 %lda, float 1.000000e+00, float* %"a'", i32 %lda)
;CHECK-NEXT:  call void @__enzyme_memcpy_floatmatrix_scal(i32 102, i32 %n, i32 %n, float %beta, float* %"c'", i32 %ldc)
;CHECK-NEXT:  ret { float, float } zeroinitializer
;CHECK-NEXT:}
