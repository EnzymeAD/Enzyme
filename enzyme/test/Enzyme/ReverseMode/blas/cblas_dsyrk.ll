;RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S -enzyme-detect-readthrow=0 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @__enzyme_autodiff(...)

declare void @cblas_dsyrk(i32, i32, i32, i32, i32, double, double*, i32, double, double*, i32)

define void @f(i32 %layout, i32 %uplo, i32 %trans, i32 %n, i32 %k, double %alpha, double* %A, i32 %lda, double %beta, double* %C) {
entry:
  call void @cblas_dsyrk(i32 %layout, i32 %uplo, i32 %trans, i32 %n, i32 %k, double %alpha, double* %A, i32 %lda, double %beta, double* %C, i32 %n)
  ret void
}

define void @active(i32 %layout, i32 %uplo, i32 %trans, i32 %n, i32 %k, double %alpha, double* %A, double* %dA, i32 %lda, double %beta, double* %C, double* %dC) {
entry:
  call void (...) @__enzyme_autodiff(void (i32, i32, i32, i32, i32, double, double*, i32, double, double*)* @f, i32 %layout, i32 %uplo, i32 %trans, i32 %n, i32 %k, metadata !"enzyme_const", double %alpha, double* %A, double* %dA, i32 %lda, metadata !"enzyme_const", double %beta, double* %C, double* %dC)
  ret void
}

; CHECK: declare void @cblas_dsyrk(i32{{.*}}, i32 "enzyme_inactive", i32 "enzyme_inactive", i32 "enzyme_inactive", i32 "enzyme_inactive", double, double* nocapture readonly, i32 "enzyme_inactive", double, double* nocapture, i32 "enzyme_inactive")

; CHECK: define internal void @[[active:.+]](i32 %layout, i32 %uplo, i32 %trans, i32 %n, i32 %k, double %alpha, double* %A, double* %"A'", i32 %lda, double %beta, double* %C, double* %"C'")
; CHECK:   call void @cblas_dsymm(i32 %layout, i8 {{.*}}, i32 %uplo, i32 {{.*}}, i32 {{.*}}, double %alpha, double* %"C'", i32 %n, double* %A, i32 %lda, double 1.000000e+00, double* %"A'", i32 %lda)

