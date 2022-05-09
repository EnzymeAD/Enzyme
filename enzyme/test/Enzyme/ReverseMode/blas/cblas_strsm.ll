;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

define void @wrapper(i32 %m, i32 %n, float %alpha, float* %a, i32 %lda, float* %b, i32 %ldb) {
entry:
  tail call void @cblas_strsm(i32 102, i32 141, i32 121, i32 111, i32 132, i32 %m, i32 %n, float %alpha, float* %a, i32 %lda, float* %b, i32 %ldb)
  ret void
}

declare void @cblas_strsm(i32, i32, i32, i32, i32, i32, i32, float, float*, i32, float*, i32)

define void @active(i32 %m, i32 %n, float %alpha, float* %a, float* %_a, i32 %lda, float* %b, float* %_b, i32 %ldb) {
entry:
  tail call void @__enzyme_autodiff(i8* bitcast (void (i32, i32, float, float*, i32, float*, i32)* @wrapper to i8*), i32 %m, i32 %n, float %alpha, float* %a, float* %_a, i32 %lda, float* %b, float* %_b, i32 %ldb)
  ret void
}

declare void @__enzyme_autodiff(i8*, i32, i32, float, float*, float*, i32, float*, float*, i32)

;CHECK: define void @active
;CHECK-NEXT: entry
;CHECK-NEXT: call { float } @[[active:.+]](

;CHECK:define internal { float } @[[active]](i32 %m, i32 %n, float %alpha, float* %a, float* %"a'", i32 %lda, float* %b, float* %"b'", i32 %ldb)
;CHECK-NEXT:entry:
;CHECK-NEXT:  tail call void @cblas_strsm(i32 102, i32 141, i32 121, i32 111, i32 132, i32 %m, i32 %n, float %alpha, float* %a, i32 %lda, float* %b, i32 %ldb)
;CHECK-NEXT:  call void @cblas_sgemm(i32 102, i32 111, i32 112, i32 %n, i32 %n, i32 %n, float 1.000000e+00, float* %"b'", i32 %ldb, float* %b, i32 %ldb, float 1.000000e+00, float* %"a'", i32 %lda)
;CHECK-NEXT:  call void @cblas_strsm(i32 102, i32 141, i32 121, i32 111, i32 132, i32 %n, i32 %n, float %alpha, float* %a, i32 %lda, float* %b, i32 %ldb)
;CHECK-NEXT:  ret { float } zeroinitializer
;CHECK-NEXT:}
