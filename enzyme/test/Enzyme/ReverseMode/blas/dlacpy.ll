; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -S -enzyme-detect-readthrow=0 | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -enzyme-detect-readthrow=0 -S -enzyme-detect-readthrow=0 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare dso_local void @__enzyme_autodiff(...)

declare void @dlacpy_64_(i8* %uplo, i64* %m, i64* %n, double* %a, i64* %lda, double* %b, i64* %ldb, i64)

define void @f(i8* %uplo, i64* %m, i64* %n, double* %a, i64* %lda, double* %b, i64* %ldb) {
entry:
  call void @dlacpy_64_(i8* %uplo, i64* %m, i64* %n, double* %a, i64* %lda, double* %b, i64* %ldb, i64 1)
  ret void
}

define void @active(i8* %uplo, i64* %m, i64* %n, double* %a, double* %da, i64* %lda, double* %b, double* %db, i64* %ldb) {
entry:
  call void (...) @__enzyme_autodiff(void (i8*, i64*, i64*, double*, i64*, double*, i64*)* @f, metadata !"enzyme_const", i8* %uplo, metadata !"enzyme_const",  i64* %m,  metadata !"enzyme_const", i64* %n,  metadata !"enzyme_dup", double* %a, double* %da, metadata !"enzyme_const",  i64* %lda, metadata !"enzyme_dup",  double* %b, double* %db, metadata !"enzyme_const",  i64* %ldb)
  ret void
}

; CHECK: define internal void @diffef(i8* %uplo, i64* %m, i64* %n, double* %a, double* %"a'", i64* %lda, double* %b, double* %"b'", i64* %ldb)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %ret = alloca double, align 8
; CHECK-NEXT:   %byref.int.one = alloca i64, align 8
; CHECK-NEXT:   call void @dlacpy_64_(i8* %uplo, i64* %m, i64* %n, double* %a, i64* %lda, double* %b, i64* %ldb, i64 1)
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %entry
; CHECK-NEXT:   store i64 1, i64* %byref.int.one, align 8
; CHECK-NEXT:   %0 = load i8, i8* %uplo, align 1
; CHECK-NEXT:   %1 = load i64, i64* %m, align 8
; CHECK-NEXT:   %2 = load i64, i64* %n, align 8
; CHECK-NEXT:   %3 = load i64, i64* %lda, align 8
; CHECK-NEXT:   %4 = load i64, i64* %ldb, align 8
; CHECK-NEXT:   call void @llvm.experimental.noalias.scope.decl(metadata !0)
; CHECK-NEXT:   call void @llvm.experimental.noalias.scope.decl(metadata !3)
; CHECK:        br i1 {{.*}}, label %__enzyme_dmemcpy_double_mat_8_8_64_zero.exit, label %swtch.i

; CHECK: swtch.i:                                          ; preds = %invertentry
; CHECK-NEXT:   switch i8 %0, label %Ginit.idx.i [
; CHECK-NEXT:     i8 85, label %Uinit.idx.i
; CHECK-NEXT:     i8 76, label %Linit.idx.i
; CHECK-NEXT:   ]

; CHECK: Gfor.body.i:                                      ; preds = %Gfor.body.i, %Ginit.idx.i
; CHECK:        load double
; CHECK:        fadd double
; CHECK:        store double
; CHECK: Ufor.body.i:                                      ; preds = %Ufor.body.i, %Uinit.idx.i
; CHECK:        load double
; CHECK:        fadd double
; CHECK:        store double
; CHECK: Lfor.body.i:                                      ; preds = %Lfor.body.i, %Linit.idx.i
; CHECK:        load double
; CHECK:        fadd double
; CHECK:        store double
; CHECK: __enzyme_dmemcpy_double_mat_8_8_64_zero.exit:     ; preds =
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

define void @activeA(i8* %uplo, i64* %m, i64* %n, double* %a, double* %da, i64* %lda, double* %b, double* %db, i64* %ldb) {
entry:
  call void (...) @__enzyme_autodiff(void (i8*, i64*, i64*, double*, i64*, double*, i64*)* @f, metadata !"enzyme_const", i8* %uplo, metadata !"enzyme_const",  i64* %m,  metadata !"enzyme_const", i64* %n,  metadata !"enzyme_dup", double* %a, double* %da, metadata !"enzyme_const",  i64* %lda, metadata !"enzyme_const",  double* %b,  metadata !"enzyme_const",  i64* %ldb)
  ret void
}

; CHECK: define internal void @diffef.1(i8* %uplo, i64* %m, i64* %n, double* %a, double* %"a'", i64* %lda, double* %b, i64* %ldb)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %ret = alloca double, align 8
; CHECK-NEXT:   %byref.int.one = alloca i64, align 8
; CHECK-NEXT:   call void @dlacpy_64_(i8* %uplo, i64* %m, i64* %n, double* %a, i64* %lda, double* %b, i64* %ldb, i64 1)
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %entry
; CHECK-NEXT:   store i64 1, i64* %byref.int.one, align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }


define void @activeB(i8* %uplo, i64* %m, i64* %n, double* %a, double* %da, i64* %lda, double* %b, double* %db, i64* %ldb) {
entry:
  call void (...) @__enzyme_autodiff(void (i8*, i64*, i64*, double*, i64*, double*, i64*)* @f, metadata !"enzyme_const", i8* %uplo, metadata !"enzyme_const",  i64* %m,  metadata !"enzyme_const", i64* %n,  metadata !"enzyme_const", double* %a, metadata !"enzyme_const",  i64* %lda, metadata !"enzyme_dup",  double* %b, double* %db, metadata !"enzyme_const",  i64* %ldb)
  ret void
}

; CHECK: define internal void @diffef.2(i8* %uplo, i64* %m, i64* %n, double* %a, i64* %lda, double* %b, double* %"b'", i64* %ldb)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %ret = alloca double, align 8
; CHECK-NEXT:   %byref.int.one = alloca i64, align 8
; CHECK-NEXT:   %byref.constant.int.0 = alloca i64, align 8
; CHECK-NEXT:   %byref.constant.int.01 = alloca i64, align 8
; CHECK-NEXT:   %byref.constant.fp.1.0 = alloca double, align 8
; CHECK-NEXT:   %byref.constant.fp.0.0 = alloca double, align 8
; CHECK-NEXT:   %0 = alloca i64, align 8
; CHECK-NEXT:   call void @dlacpy_64_(i8* %uplo, i64* %m, i64* %n, double* %a, i64* %lda, double* %b, i64* %ldb, i64 1)
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %entry
; CHECK-NEXT:   store i64 1, i64* %byref.int.one, align 8
; CHECK-NEXT:   store i64 0, i64* %byref.constant.int.0, align 8
; CHECK-NEXT:   store i64 0, i64* %byref.constant.int.01, align 8
; CHECK-NEXT:   store double 1.000000e+00, double* %byref.constant.fp.1.0, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %byref.constant.fp.0.0, align 8
; CHECK-NEXT:   call void @dlascl_64_(i8* %uplo, i64* %byref.constant.int.0, i64* %byref.constant.int.01, double* %byref.constant.fp.1.0, double* %byref.constant.fp.0.0, i64* %m, i64* %n, double* %"b'", i64* %ldb, i64* %0, i64 1)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
