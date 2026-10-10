; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -S -enzyme-detect-readthrow=0 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @dtrsv_64_(i8*, i8*, i8*, i64*, double*, i64*, double*, i64*, i64, i64, i64)

define void @f(double* %A, double* %x) {
entry:
  %uplo = alloca i8, align 1
  %trans = alloca i8, align 1
  %diag = alloca i8, align 1
  %n = alloca i64, align 8
  %lda = alloca i64, align 8
  %incx = alloca i64, align 8
  store i8 85, i8* %uplo, align 1
  store i8 78, i8* %trans, align 1
  store i8 78, i8* %diag, align 1
  store i64 4, i64* %n, align 8
  store i64 4, i64* %lda, align 8
  store i64 1, i64* %incx, align 8
  call void @dtrsv_64_(i8* %uplo, i8* %trans, i8* %diag, i64* %n, double* %A, i64* %lda, double* %x, i64* %incx, i64 1, i64 1, i64 1)
  ret void
}

declare void @__enzyme_autodiff(...)

define void @active(double* %A, double* %dA, double* %x, double* %dx) {
entry:
  call void (...) @__enzyme_autodiff(void (double*, double*)* @f, metadata !"enzyme_dup", double* %A, double* %dA, metadata !"enzyme_dup", double* %x, double* %dx)
  ret void
}

; CHECK: define internal void @diffef(double* %A, double* %"A'", double* %x, double* %"x'")
; CHECK: call void @dtrsv_64_
; CHECK: invertentry:
; CHECK: call void @dtrsv_64_
; CHECK: call void @dlacpy_64_
; CHECK: call void @dger_64_
; CHECK: call void @dcopy_64_
; CHECK: call void @dlacpy_64_
