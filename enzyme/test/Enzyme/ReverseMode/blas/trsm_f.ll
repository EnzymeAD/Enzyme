; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -S -enzyme-detect-readthrow=0 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @dtrsm_64_(i8*, i8*, i8*, i8*, i64*, i64*, double*, double*, i64*, double*, i64*, i64, i64, i64, i64)

define void @f(double* %A, double* %B) {
entry:
  %side = alloca i8, align 1
  %uplo = alloca i8, align 1
  %transa = alloca i8, align 1
  %diag = alloca i8, align 1
  %m = alloca i64, align 8
  %n = alloca i64, align 8
  %alpha = alloca double, align 8
  %lda = alloca i64, align 8
  %ldb = alloca i64, align 8
  store i8 76, i8* %side, align 1
  store i8 85, i8* %uplo, align 1
  store i8 78, i8* %transa, align 1
  store i8 78, i8* %diag, align 1
  store i64 4, i64* %m, align 8
  store i64 3, i64* %n, align 8
  store double 2.000000e+00, double* %alpha, align 8
  store i64 4, i64* %lda, align 8
  store i64 4, i64* %ldb, align 8
  call void @dtrsm_64_(i8* %side, i8* %uplo, i8* %transa, i8* %diag, i64* %m, i64* %n, double* %alpha, double* %A, i64* %lda, double* %B, i64* %ldb, i64 1, i64 1, i64 1, i64 1)
  ret void
}

declare void @__enzyme_autodiff(...)

define void @active(double* %A, double* %dA, double* %B, double* %dB) {
entry:
  call void (...) @__enzyme_autodiff(void (double*, double*)* @f, metadata !"enzyme_dup", double* %A, double* %dA, metadata !"enzyme_dup", double* %B, double* %dB)
  ret void
}

; CHECK: define internal void @diffef(double* %A, double* %"A'", double* %B, double* %"B'")
; CHECK: call void @dtrsm_64_
; CHECK: invertentry:
; CHECK: call void @dlacpy_64_
; CHECK: call void @dtrsm_64_
; CHECK: call void @dlacpy_64_
; CHECK: call void @dgemm_64_
; CHECK: call void @dcopy_64_
; CHECK: call void @dlacpy_64_
; CHECK: call void @dtrsm_64_
