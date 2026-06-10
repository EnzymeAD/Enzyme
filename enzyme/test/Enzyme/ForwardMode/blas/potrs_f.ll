; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -S -enzyme-detect-readthrow=0 | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -enzyme-detect-readthrow=0 -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare dso_local void @__enzyme_fwddiff(...)

declare void @dpotrs_64_(i8*, i64*, i64*, double*, i64*, double*, i64*, i64*, i64)

define void @f(double* %A, double* %B) {
entry:
  %uplo = alloca i8, align 1
  %n = alloca i64, align 8
  %nrhs = alloca i64, align 8
  %lda = alloca i64, align 8
  %ldb = alloca i64, align 8
  %info = alloca i64, align 8
  store i8 85, i8* %uplo, align 1
  store i64 4, i64* %n, align 8
  store i64 3, i64* %nrhs, align 8
  store i64 4, i64* %lda, align 8
  store i64 4, i64* %ldb, align 8
  call void @dpotrs_64_(i8* %uplo, i64* %n, i64* %nrhs, double* %A, i64* %lda, double* %B, i64* %ldb, i64* %info, i64 1)
  ret void
}

define void @active(double* %A, double* %dA, double* %B, double* %dB) {
entry:
  call void (...) @__enzyme_fwddiff(void (double*, double*)* @f, metadata !"enzyme_dup", double* %A, double* %dA, metadata !"enzyme_dup", double* %B, double* %dB)
  ret void
}

; CHECK-LABEL: define internal void @fwddiffef(
; CHECK: call void @dpotrs_64_
; CHECK: call void @dlacpy_64_
; CHECK: call void @dtrmm_64_
; CHECK: call void @dtrmm_64_
; CHECK: call void @dlascl_64_
; CHECK: call void @dlacpy_64_
; CHECK: call void @dtrmm_64_
; CHECK: call void @dtrmm_64_
; CHECK: call void @dlascl_64_
; CHECK: call void @dpotrs_64_
; CHECK: ret void
