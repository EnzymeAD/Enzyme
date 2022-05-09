;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

define void @wrapper(i32 %n, i32 %k, double %alpha, double* %a, i32 %lda, double %beta, double* %c, i32 %ldc) {
entry:
  tail call void @cblas_dsyrk(i32 102, i32 122, i32 111, i32 %n, i32 %k, double %alpha, double* %a, i32 %lda, double %beta, double* %c, i32 %ldc)
  ret void
}

declare void @cblas_dsyrk(i32, i32, i32, i32, i32, double, double*, i32, double, double*, i32)

define void @active(i32 %n, i32 %k, double %alpha, double* %a, double* %_a, i32 %lda, double %beta, double* %c, double* %_c, i32 %ldc) {
entry:
  tail call void @__enzyme_autodiff(i8* bitcast (void (i32, i32, double, double*, i32, double, double*, i32)* @wrapper to i8*), i32 %n, i32 %k, double %alpha, double* %a, double* %_a, i32 %lda, double %beta, double* %c, double* %_c, i32 %ldc)
  ret void
}

declare void @__enzyme_autodiff(i8*, i32, i32, double, double*, double*, i32, double, double*, double*, i32)

;CHECK: define void @active
;CHECK-NEXT: entry
;CHECK-NEXT: call { double, double } @[[active:.+]](

;CHECK:define internal { double, double } @[[active]](i32 %n, i32 %k, double %alpha, double* %a, double* %"a'", i32 %lda, double %beta, double* %c, double* %"c'", i32 %ldc)
;CHECK-NEXT:entry:
;CHECK-NEXT:  tail call void @cblas_dsyrk(i32 102, i32 122, i32 111, i32 %n, i32 %k, double %alpha, double* %a, i32 %lda, double %beta, double* %c, i32 %ldc)
;CHECK-NEXT:  %0 = fmul fast double %alpha, 2.000000e+00
;CHECK-NEXT:  call void @cblas_dsymm(i32 102, i32 141, i32 122, i32 %n, i32 %k, double %0, double* %"c'", i32 %ldc, double* %a, i32 %lda, double 1.000000e+00, double* %"a'", i32 %lda)
;CHECK-NEXT:  call void @__enzyme_memcpy_doublematrix_scal(i32 102, i32 %n, i32 %n, double %beta, double* %"c'", i32 %ldc)
;CHECK-NEXT:  ret { double, double } zeroinitializer
;CHECK-NEXT:}
