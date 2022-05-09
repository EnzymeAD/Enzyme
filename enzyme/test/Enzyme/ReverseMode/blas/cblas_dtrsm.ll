;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

define void @wrapper(i32 %m, i32 %n, double %alpha, double* %a, i32 %lda, double* %b, i32 %ldb) {
entry:
  tail call void @cblas_dtrsm(i32 102, i32 141, i32 121, i32 111, i32 132, i32 %m, i32 %n, double %alpha, double* %a, i32 %lda, double* %b, i32 %ldb)
  ret void
}

declare void @cblas_dtrsm(i32, i32, i32, i32, i32, i32, i32, double, double*, i32, double*, i32)

define void @active(i32 %m, i32 %n, double %alpha, double* %a, double* %_a, i32 %lda, double* %b, double* %_b, i32 %ldb) {
entry:
  tail call void @__enzyme_autodiff(i8* bitcast (void (i32, i32, double, double*, i32, double*, i32)* @wrapper to i8*), i32 %m, i32 %n, double %alpha, double* %a, double* %_a, i32 %lda, double* %b, double* %_b, i32 %ldb)
  ret void
}

declare void @__enzyme_autodiff(i8*, i32, i32, double, double*, double*, i32, double*, double*, i32)

;CHECK: define void @active
;CHECK-NEXT: entry
;CHECK-NEXT: call { double } @[[active:.+]](

;CHECK:define internal { double } @[[active]](i32 %m, i32 %n, double %alpha, double* %a, double* %"a'", i32 %lda, double* %b, double* %"b'", i32 %ldb)
;CHECK-NEXT:entry:
;CHECK-NEXT:  tail call void @cblas_dtrsm(i32 102, i32 141, i32 121, i32 111, i32 132, i32 %m, i32 %n, double %alpha, double* %a, i32 %lda, double* %b, i32 %ldb)
;CHECK-NEXT:  call void @cblas_dgemm(i32 102, i32 111, i32 112, i32 %n, i32 %n, i32 %n, double 1.000000e+00, double* %"b'", i32 %ldb, double* %b, i32 %ldb, double 1.000000e+00, double* %"a'", i32 %lda)
;CHECK-NEXT:  call void @cblas_dtrsm(i32 102, i32 141, i32 121, i32 111, i32 132, i32 %n, i32 %n, double %alpha, double* %a, i32 %lda, double* %b, i32 %ldb)
;CHECK-NEXT:  ret { double } zeroinitializer
;CHECK-NEXT:}
