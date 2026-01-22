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
; CHECK-NEXT:   %5 = icmp eq i64 %1, 0
; CHECK-NEXT:   %6 = icmp eq i64 %2, 0
; CHECK-NEXT:   %7 = or i1 %5, %6
; CHECK-NEXT:   br i1 %7, label %__enzyme_dmemcpy_double_mat_64_zero.exit, label %swtch.i

; CHECK: swtch.i:                                          ; preds = %invertentry
; CHECK-NEXT:   switch i8 %0, label %Ginit.idx.i [
; CHECK-NEXT:     i8 85, label %Uinit.idx.i
; CHECK-NEXT:     i8 76, label %Linit.idx.i
; CHECK-NEXT:   ]

; CHECK: Ginit.idx.i:                                      ; preds = %Ginit.end.i, %swtch.i
; CHECK-NEXT:   %Gj.i = phi i64 [ 0, %swtch.i ], [ %Gj.next.i, %Ginit.end.i ]
; CHECK-NEXT:   br label %Gfor.body.i

; CHECK: Uinit.idx.i:                                      ; preds = %Uinit.end.i, %swtch.i
; CHECK-NEXT:   %Uj.i = phi i64 [ 0, %swtch.i ], [ %Uj.next.i, %Uinit.end.i ]
; CHECK-NEXT:   %8 = add nuw nsw i64 %Uj.i, 1
; CHECK-NEXT:   %9 = icmp ult i64 %8, %1
; CHECK-NEXT:   %10 = select i1 %9, i64 %8, i64 %1
; CHECK-NEXT:   br label %Ufor.body.i

; CHECK: Linit.idx.i:                                      ; preds = %Linit.end.i, %swtch.i
; CHECK-NEXT:   %Lj.i = phi i64 [ 0, %swtch.i ], [ %Lj.next.i, %Linit.end.i ]
; CHECK-NEXT:   br label %Lfor.body.i

; CHECK: Gfor.body.i:                                      ; preds = %Gfor.body.i, %Ginit.idx.i
; CHECK-NEXT:   %Gi.i = phi i64 [ 0, %Ginit.idx.i ], [ %Gi.next.i, %Gfor.body.i ]
; CHECK-NEXT:   %11 = mul nuw nsw i64 %Gj.i, %4
; CHECK-NEXT:   %12 = add nuw nsw i64 %Gi.i, %11
; CHECK-NEXT:   %Gsrc.i.i = getelementptr inbounds double, double* %"b'", i64 %12
; CHECK-NEXT:   %13 = mul nuw nsw i64 %Gj.i, %3
; CHECK-NEXT:   %14 = add nuw nsw i64 %Gi.i, %13
; CHECK-NEXT:   %Gdst.i.i = getelementptr inbounds double, double* %"a'", i64 %14
; CHECK-NEXT:   %Gsrc.i.l.i = load double, double* %Gsrc.i.i, align 8, !alias.scope !3, !noalias !0
; CHECK-NEXT:   %Gdst.i.l.i = load double, double* %Gdst.i.i, align 8, !alias.scope !0, !noalias !3
; CHECK-NEXT:   %15 = fadd double %Gsrc.i.l.i, %Gdst.i.l.i
; CHECK-NEXT:   store double %15, double* %Gdst.i.i, align 8, !alias.scope !0, !noalias !3
; CHECK-NEXT:   store double 0.000000e+00, double* %Gsrc.i.i, align 8, !alias.scope !3, !noalias !0
; CHECK-NEXT:   %Gi.next.i = add nuw nsw i64 %Gi.i, 1
; CHECK-NEXT:   %16 = icmp eq i64 %Gi.next.i, %1
; CHECK-NEXT:   br i1 %16, label %Ginit.end.i, label %Gfor.body.i

; CHECK: Ginit.end.i:                                      ; preds = %Gfor.body.i
; CHECK-NEXT:   %Gj.next.i = add nuw nsw i64 %Gj.i, 1
; CHECK-NEXT:   %17 = icmp eq i64 %Gj.next.i, %2
; CHECK-NEXT:   br i1 %17, label %__enzyme_dmemcpy_double_mat_64_zero.exit, label %Ginit.idx.i

; CHECK: Ufor.body.i:                                      ; preds = %Ufor.body.i, %Uinit.idx.i
; CHECK-NEXT:   %Ui.i = phi i64 [ 0, %Uinit.idx.i ], [ %Ui.next.i, %Ufor.body.i ]
; CHECK-NEXT:   %18 = mul nuw nsw i64 %Uj.i, %4
; CHECK-NEXT:   %19 = add nuw nsw i64 %Ui.i, %18
; CHECK-NEXT:   %Usrc.i.i = getelementptr inbounds double, double* %"b'", i64 %19
; CHECK-NEXT:   %20 = mul nuw nsw i64 %Uj.i, %3
; CHECK-NEXT:   %21 = add nuw nsw i64 %Ui.i, %20
; CHECK-NEXT:   %Udst.i.i = getelementptr inbounds double, double* %"a'", i64 %21
; CHECK-NEXT:   %Usrc.i.l.i = load double, double* %Usrc.i.i, align 8, !alias.scope !3, !noalias !0
; CHECK-NEXT:   %Udst.i.l.i = load double, double* %Udst.i.i, align 8, !alias.scope !0, !noalias !3
; CHECK-NEXT:   %22 = fadd double %Usrc.i.l.i, %Udst.i.l.i
; CHECK-NEXT:   store double %22, double* %Udst.i.i, align 8, !alias.scope !0, !noalias !3
; CHECK-NEXT:   store double 0.000000e+00, double* %Usrc.i.i, align 8, !alias.scope !3, !noalias !0
; CHECK-NEXT:   %Ui.next.i = add nuw nsw i64 %Ui.i, 1
; CHECK-NEXT:   %23 = icmp eq i64 %Ui.next.i, %10
; CHECK-NEXT:   br i1 %23, label %Uinit.end.i, label %Ufor.body.i

; CHECK: Uinit.end.i:                                      ; preds = %Ufor.body.i
; CHECK-NEXT:   %Uj.next.i = add nuw nsw i64 %Uj.i, 1
; CHECK-NEXT:   %24 = icmp eq i64 %Uj.next.i, %2
; CHECK-NEXT:   br i1 %24, label %__enzyme_dmemcpy_double_mat_64_zero.exit, label %Uinit.idx.i

; CHECK: Lfor.body.i:                                      ; preds = %Lfor.body.i, %Linit.idx.i
; CHECK-NEXT:   %Li.i = phi i64 [ %Lj.i, %Linit.idx.i ], [ %Li.next.i, %Lfor.body.i ]
; CHECK-NEXT:   %25 = mul nuw nsw i64 %Lj.i, %4
; CHECK-NEXT:   %26 = add nuw nsw i64 %Li.i, %25
; CHECK-NEXT:   %Lsrc.i.i = getelementptr inbounds double, double* %"b'", i64 %26
; CHECK-NEXT:   %27 = mul nuw nsw i64 %Lj.i, %3
; CHECK-NEXT:   %28 = add nuw nsw i64 %Li.i, %27
; CHECK-NEXT:   %Ldst.i.i = getelementptr inbounds double, double* %"a'", i64 %28
; CHECK-NEXT:   %Lsrc.i.l.i = load double, double* %Lsrc.i.i, align 8, !alias.scope !3, !noalias !0
; CHECK-NEXT:   %Ldst.i.l.i = load double, double* %Ldst.i.i, align 8, !alias.scope !0, !noalias !3
; CHECK-NEXT:   %29 = fadd double %Lsrc.i.l.i, %Ldst.i.l.i
; CHECK-NEXT:   store double %29, double* %Ldst.i.i, align 8, !alias.scope !0, !noalias !3
; CHECK-NEXT:   store double 0.000000e+00, double* %Lsrc.i.i, align 8, !alias.scope !3, !noalias !0
; CHECK-NEXT:   %Li.next.i = add nuw nsw i64 %Li.i, 1
; CHECK-NEXT:   %30 = icmp eq i64 %Li.next.i, %1
; CHECK-NEXT:   br i1 %30, label %Linit.end.i, label %Lfor.body.i

; CHECK: Linit.end.i:                                      ; preds = %Lfor.body.i
; CHECK-NEXT:   %Lj.next.i = add nuw nsw i64 %Lj.i, 1
; CHECK-NEXT:   %31 = icmp eq i64 %Lj.next.i, %2
; CHECK-NEXT:   br i1 %31, label %__enzyme_dmemcpy_double_mat_64_zero.exit, label %Linit.idx.i

; CHECK: __enzyme_dmemcpy_double_mat_64_zero.exit:         ; preds = %invertentry, %Ginit.end.i, %Uinit.end.i, %Linit.end.i
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

