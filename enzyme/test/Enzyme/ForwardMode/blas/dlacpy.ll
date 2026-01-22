; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -S -enzyme-detect-readthrow=0 | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -enzyme-detect-readthrow=0 -S -enzyme-detect-readthrow=0 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare dso_local void @__enzyme_fwddiff(...)

declare void @dlacpy_64_(i8* %uplo, i64* %m, i64* %n, double* %a, i64* %lda, double* %b, i64* %ldb, i64)

define void @f(i8* %uplo, i64* %m, i64* %n, double* %a, i64* %lda, double* %b, i64* %ldb) {
entry:
  call void @dlacpy_64_(i8* %uplo, i64* %m, i64* %n, double* %a, i64* %lda, double* %b, i64* %ldb, i64 1)
  ret void
}

define void @active(i8* %uplo, i64* %m, i64* %n, double* %a, double* %da, i64* %lda, double* %b, double* %db, i64* %ldb) {
entry:
  call void (...) @__enzyme_fwddiff(void (i8*, i64*, i64*, double*, i64*, double*, i64*)* @f, metadata !"enzyme_const", i8* %uplo, metadata !"enzyme_const",  i64* %m,  metadata !"enzyme_const", i64* %n,  metadata !"enzyme_dup", double* %a, double* %da, metadata !"enzyme_const",  i64* %lda, metadata !"enzyme_dup",  double* %b, double* %db, metadata !"enzyme_const",  i64* %ldb)
  ret void
}

; CHECK: define internal void @fwddiffef(i8* %uplo, i64* %m, i64* %n, double* %a, double* %"a'", i64* %lda, double* %b, double* %"b'", i64* %ldb)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %byref.int.one = alloca i64, align 8
; CHECK-NEXT:   store i64 1, i64* %byref.int.one, align 8
; CHECK-NEXT:   call void @dlacpy_64_(i8* %uplo, i64* %m, i64* %n, double* %"a'", i64* %lda, double* %"b'", i64* %ldb, i64 1)
; CHECK-NEXT:   call void @dlacpy_64_(i8* %uplo, i64* %m, i64* %n, double* %a, i64* %lda, double* %b, i64* %ldb, i64 1)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; TODO if the dst is active, this should emit a 0, currently it does not!

define void @activeA(i8* %uplo, i64* %m, i64* %n, double* %a, double* %da, i64* %lda, double* %b, double* %db, i64* %ldb) {
entry:
  call void (...) @__enzyme_fwddiff(void (i8*, i64*, i64*, double*, i64*, double*, i64*)* @f, metadata !"enzyme_const", i8* %uplo, metadata !"enzyme_const",  i64* %m,  metadata !"enzyme_const", i64* %n,  metadata !"enzyme_dup", double* %a, double* %da, metadata !"enzyme_const",  i64* %lda, metadata !"enzyme_const",  double* %b,  metadata !"enzyme_const",  i64* %ldb)
  ret void
}


define void @activeB(i8* %uplo, i64* %m, i64* %n, double* %a, double* %da, i64* %lda, double* %b, double* %db, i64* %ldb) {
entry:
  call void (...) @__enzyme_fwddiff(void (i8*, i64*, i64*, double*, i64*, double*, i64*)* @f, metadata !"enzyme_const", i8* %uplo, metadata !"enzyme_const",  i64* %m,  metadata !"enzyme_const", i64* %n,  metadata !"enzyme_const", double* %a, metadata !"enzyme_const",  i64* %lda, metadata !"enzyme_dup",  double* %b, double* %db, metadata !"enzyme_const",  i64* %ldb)
  ret void
}
