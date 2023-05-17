;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare dso_local void @__enzyme_autodiff(...)

declare double @cblas_ddot(i32, double*, i32, double*, i32)

define void @active(i32 %len, double* noalias %m, double* %dm, i32 %incm, double* noalias %n, double* %dn, i32 %incn) {
entry:
  call void (...) @__enzyme_autodiff(double (i32, double*, i32, double*, i32)* @f, metadata !"enzyme_width", i64 2, i32 %len, double* noalias %m, double* %dm, double* %dm, i32 %incm, double* noalias %n, double* %dn, double* %dn, i32 %incn)
  ret void
}

define void @inactiveFirst(i32 %len, double* noalias %m, i32 %incm, double* noalias %n, double* %dn, i32 %incn) {
entry:
  call void (...) @__enzyme_autodiff(double (i32, double*, i32, double*, i32)* @f, metadata !"enzyme_width", i64 2,  i32 %len, metadata !"enzyme_const", double* noalias %m, i32 %incm, double* noalias %n, double* %dn, double* %dn, i32 %incn)
  ret void
}

define void @inactiveSecond(i32 %len, double* noalias %m, double* noalias %dm, i32 %incm, double* noalias %n, i32 %incn) {
entry:
  call void (...) @__enzyme_autodiff(double (i32, double*, i32, double*, i32)* @f, metadata !"enzyme_width", i64 2, i32 %len, double* noalias %m, double* noalias %dm, double* %dm, i32 %incm, metadata !"enzyme_const", double* noalias %n, i32 %incn)
  ret void
}

define void @activeMod(i32 %len, double* noalias %m, double* %dm, i32 %incm, double* noalias %n, double* %dn, i32 %incn) {
entry:
  call void (...) @__enzyme_autodiff(double (i32, double*, i32, double*, i32)* @modf, metadata !"enzyme_width", i64 2, i32 %len, double* noalias %m, double* %dm, double* %dm, i32 %incm, double* noalias %n, double* %dn, double* %dn, i32 %incn)
  ret void
}

define void @inactiveModFirst(i32 %len, double* noalias %m, i32 %incm, double* noalias %n, double* %dn, i32 %incn) {
entry:
  call void (...) @__enzyme_autodiff(double (i32, double*, i32, double*, i32)* @modf, metadata !"enzyme_width", i64 2,  i32 %len, metadata !"enzyme_const", double* noalias %m, i32 %incm, double* noalias %n, double* %dn, double* %dn, i32 %incn)
  ret void
}

define void @inactiveModSecond(i32 %len, double* noalias %m, double* noalias %dm, i32 %incm, double* noalias %n, i32 %incn) {
entry:
  call void (...) @__enzyme_autodiff(double (i32, double*, i32, double*, i32)* @modf, metadata !"enzyme_width", i64 2, i32 %len, double* noalias %m, double* noalias %dm, double* %dm, i32 %incm, metadata !"enzyme_const", double* noalias %n, i32 %incn)
  ret void
}

define double @f(i32 %len, double* noalias %m, i32 %incm, double* noalias %n, i32 %incn) {
entry:
  %call = call double @cblas_ddot(i32 %len, double* %m, i32 %incm, double* %n, i32 %incn)
  ret double %call
}

define double @modf(i32 %len, double* noalias %m, i32 %incm, double* noalias %n, i32 %incn) {
entry:
  %call = call double @f(i32 %len, double* %m, i32 %incm, double* %n, i32 %incn)
  store double 0.000000e+00, double* %m
  store double 0.000000e+00, double* %n
  ret double %call
}


; CHECK: define void @active
; CHECK-NEXT: entry
; CHECK: call void @[[active:.+]](

; CHECK: define void @inactiveFirst
; CHECK-NEXT: entry
; CHECK: call void @[[inactiveFirst:.+]](

; CHECK: define void @inactiveSecond
; CHECK-NEXT: entry
; CHECK: call void @[[inactiveSecond:.+]](


; CHECK: define void @activeMod
; CHECK-NEXT: entry
; CHECK: call void @[[activeMod:.+]](

; CHECK: define void @inactiveModFirst
; CHECK-NEXT: entry
; CHECK: call void @[[inactiveModFirst:.+]](

; CHECK: define void @inactiveModSecond
; CHECK-NEXT: entry
; CHECK: call void @[[inactiveModSecond:.+]](


; CHECK: define internal void @[[active]](i32 %len, double* noalias %m, [2 x double*] %"m'", i32 %incm, double* noalias %n, [2 x double*] %"n'", i32 %incn, [2 x double] %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue [2 x double*] %"m'", 0
; CHECK-NEXT:   %1 = extractvalue [2 x double*] %"n'", 0
; CHECK-NEXT:   %2 = extractvalue [2 x double] %differeturn, 0
; CHECK-NEXT:   call void @cblas_daxpy(i32 %len, double %2, double* %n, i32 %incn, double* %0, i32 %incm)
; CHECK-NEXT:   call void @cblas_daxpy(i32 %len, double %2, double* %m, i32 %incm, double* %1, i32 %incn)
; CHECK-NEXT:   %3 = extractvalue [2 x double*] %"m'", 1
; CHECK-NEXT:   %4 = extractvalue [2 x double*] %"n'", 1
; CHECK-NEXT:   %5 = extractvalue [2 x double] %differeturn, 1
; CHECK-NEXT:   call void @cblas_daxpy(i32 %len, double %5, double* %n, i32 %incn, double* %3, i32 %incm)
; CHECK-NEXT:   call void @cblas_daxpy(i32 %len, double %5, double* %m, i32 %incm, double* %4, i32 %incn)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @[[inactiveFirst]](i32 %len, double* noalias %m, i32 %incm, double* noalias %n, [2 x double*] %"n'", i32 %incn, [2 x double] %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue [2 x double*] %"n'", 0
; CHECK-NEXT:   %1 = extractvalue [2 x double] %differeturn, 0
; CHECK-NEXT:   call void @cblas_daxpy(i32 %len, double %1, double* %m, i32 %incm, double* %0, i32 %incn)
; CHECK-NEXT:   %2 = extractvalue [2 x double*] %"n'", 1
; CHECK-NEXT:   %3 = extractvalue [2 x double] %differeturn, 1
; CHECK-NEXT:   call void @cblas_daxpy(i32 %len, double %3, double* %m, i32 %incm, double* %2, i32 %incn)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @[[inactiveSecond]](i32 %len, double* noalias %m, [2 x double*] %"m'", i32 %incm, double* noalias %n, i32 %incn, [2 x double] %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue [2 x double*] %"m'", 0
; CHECK-NEXT:   %1 = extractvalue [2 x double] %differeturn, 0
; CHECK-NEXT:   call void @cblas_daxpy(i32 %len, double %1, double* %n, i32 %incn, double* %0, i32 %incm)
; CHECK-NEXT:   %2 = extractvalue [2 x double*] %"m'", 1
; CHECK-NEXT:   %3 = extractvalue [2 x double] %differeturn, 1
; CHECK-NEXT:   call void @cblas_daxpy(i32 %len, double %3, double* %n, i32 %incn, double* %2, i32 %incm)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @[[activeMod]](i32 %len, double* noalias %m, [2 x double*] %"m'", i32 %incm, double* noalias %n, [2 x double*] %"n'", i32 %incn, [2 x double] %differeturn)
; CHECK-NEXT: entry:
; CHECK:        %call_augmented = call { double*, double* } @[[augMod:.+]](i32 %len, double* %m, [2 x double*] %"m'", i32 %incm, double* %n, [2 x double*] %"n'", i32 %incn)
; CHECK:        call void @[[revMod:.+]](i32 %len, double* %m, [2 x double*] %"m'", i32 %incm, double* %n, [2 x double*] %"n'", i32 %incn, [2 x double] %differeturn, { double*, double* } %call_augmented)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal { double*, double* } @[[augMod]](i32 %len, double* noalias %m, [2 x double*] %"m'", i32 %incm, double* noalias %n, [2 x double*] %"n'", i32 %incn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %mallocsize = mul nuw nsw i32 %len, 8
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i32 %mallocsize)
; CHECK-NEXT:   %0 = bitcast i8* %malloccall to double*
; CHECK-NEXT:   %1 = icmp eq i32 %len, 0
; CHECK-NEXT:   br i1 %1, label %__enzyme_memcpy_double_32_da0sa0stride.exit, label %init.idx.i

; CHECK: init.idx.i:                                       ; preds = %entry
; CHECK-NEXT:   %a.i = sub nsw i32 1, %len
; CHECK-NEXT:   %negidx.i = mul nsw i32 %a.i, %incm
; CHECK-NEXT:   %is.neg.i = icmp slt i32 %incm, 0
; CHECK-NEXT:   %startidx.i = select i1 %is.neg.i, i32 %negidx.i, i32 0
; CHECK-NEXT:   br label %for.body.i

; CHECK: for.body.i:                                       ; preds = %for.body.i, %init.idx.i
; CHECK-NEXT:   %idx.i = phi i32 [ 0, %init.idx.i ], [ %idx.next.i, %for.body.i ]
; CHECK-NEXT:   %sidx.i = phi i32 [ %startidx.i, %init.idx.i ], [ %sidx.next.i, %for.body.i ]
; CHECK-NEXT:   %dst.i.i = getelementptr inbounds double, double* %0, i32 %idx.i
; CHECK-NEXT:   %src.i.i = getelementptr inbounds double, double* %m, i32 %sidx.i
; CHECK-NEXT:   %src.i.l.i = load double, double* %src.i.i
; CHECK-NEXT:   store double %src.i.l.i, double* %dst.i.i
; CHECK-NEXT:   %idx.next.i = add nsw i32 %idx.i, 1
; CHECK-NEXT:   %sidx.next.i = add nsw i32 %sidx.i, %incm
; CHECK-NEXT:   %2 = icmp eq i32 %len, %idx.next.i
; CHECK-NEXT:   br i1 %2, label %__enzyme_memcpy_double_32_da0sa0stride.exit, label %for.body.i

; CHECK: __enzyme_memcpy_double_32_da0sa0stride.exit:      ; preds = %entry, %for.body.i
; CHECK-NEXT:   %mallocsize1 = mul nuw nsw i32 %len, 8
; CHECK-NEXT:   %malloccall2 = tail call noalias nonnull i8* @malloc(i32 %mallocsize1)
; CHECK-NEXT:   %3 = bitcast i8* %malloccall2 to double*
; CHECK-NEXT:   %4 = icmp eq i32 %len, 0
; CHECK-NEXT:   br i1 %4, label %__enzyme_memcpy_double_32_da0sa0stride.exit14, label %init.idx.i5

; CHECK: init.idx.i5:                                      ; preds = %__enzyme_memcpy_double_32_da0sa0stride.exit
; CHECK-NEXT:   %a.i1 = sub nsw i32 1, %len
; CHECK-NEXT:   %negidx.i2 = mul nsw i32 %a.i1, %incn
; CHECK-NEXT:   %is.neg.i3 = icmp slt i32 %incn, 0
; CHECK-NEXT:   %startidx.i4 = select i1 %is.neg.i3, i32 %negidx.i2, i32 0
; CHECK-NEXT:   br label %for.body.i13

; CHECK: for.body.i13:                                     ; preds = %for.body.i13, %init.idx.i5
; CHECK-NEXT:   %idx.i6 = phi i32 [ 0, %init.idx.i5 ], [ %idx.next.i11, %for.body.i13 ]
; CHECK-NEXT:   %sidx.i7 = phi i32 [ %startidx.i4, %init.idx.i5 ], [ %sidx.next.i12, %for.body.i13 ]
; CHECK-NEXT:   %dst.i.i8 = getelementptr inbounds double, double* %3, i32 %idx.i6
; CHECK-NEXT:   %src.i.i9 = getelementptr inbounds double, double* %n, i32 %sidx.i7
; CHECK-NEXT:   %src.i.l.i10 = load double, double* %src.i.i9
; CHECK-NEXT:   store double %src.i.l.i10, double* %dst.i.i8
; CHECK-NEXT:   %idx.next.i11 = add nsw i32 %idx.i6, 1
; CHECK-NEXT:   %sidx.next.i12 = add nsw i32 %sidx.i7, %incn
; CHECK-NEXT:   %5 = icmp eq i32 %len, %idx.next.i11
; CHECK-NEXT:   br i1 %5, label %__enzyme_memcpy_double_32_da0sa0stride.exit14, label %for.body.i13

; CHECK: __enzyme_memcpy_double_32_da0sa0stride.exit14:    ; preds = %__enzyme_memcpy_double_32_da0sa0stride.exit, %for.body.i13
; CHECK-NEXT:   %6 = insertvalue { double*, double* } undef, double* %0, 0
; CHECK-NEXT:   %7 = insertvalue { double*, double* } %6, double* %3, 1
; CHECK-NEXT:   ret { double*, double* } %7
; CHECK-NEXT: }

; CHECK: define internal void @[[inactiveModFirst]](i32 %len, double* noalias %m, i32 %incm, double* noalias %n, [2 x double*] %"n'", i32 %incn, [2 x double] %differeturn)
; CHECK-NEXT: entry:
; CHECK:        %call_augmented = call double* @[[augModFirst:.+]](i32 %len, double* %m, i32 %incm, double* %n, [2 x double*] %"n'", i32 %incn)
; CHECK:        call void @[[revModFirst:.+]](i32 %len, double* %m, i32 %incm, double* %n, [2 x double*] %"n'", i32 %incn, [2 x double] %differeturn, double* %call_augmented)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal double* @[[augModFirst]](i32 %len, double* noalias %m, i32 %incm, double* noalias %n, [2 x double*] %"n'", i32 %incn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %mallocsize = mul nuw nsw i32 %len, 8
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i32 %mallocsize)
; CHECK-NEXT:   %0 = bitcast i8* %malloccall to double*
; CHECK-NEXT:   %1 = icmp eq i32 %len, 0
; CHECK-NEXT:   br i1 %1, label %__enzyme_memcpy_double_32_da0sa0stride.exit, label %init.idx.i

; CHECK: init.idx.i:                                       ; preds = %entry
; CHECK-NEXT:   %a.i = sub nsw i32 1, %len
; CHECK-NEXT:   %negidx.i = mul nsw i32 %a.i, %incm
; CHECK-NEXT:   %is.neg.i = icmp slt i32 %incm, 0
; CHECK-NEXT:   %startidx.i = select i1 %is.neg.i, i32 %negidx.i, i32 0
; CHECK-NEXT:   br label %for.body.i

; CHECK: for.body.i:                                       ; preds = %for.body.i, %init.idx.i
; CHECK-NEXT:   %idx.i = phi i32 [ 0, %init.idx.i ], [ %idx.next.i, %for.body.i ]
; CHECK-NEXT:   %sidx.i = phi i32 [ %startidx.i, %init.idx.i ], [ %sidx.next.i, %for.body.i ]
; CHECK-NEXT:   %dst.i.i = getelementptr inbounds double, double* %0, i32 %idx.i
; CHECK-NEXT:   %src.i.i = getelementptr inbounds double, double* %m, i32 %sidx.i
; CHECK-NEXT:   %src.i.l.i = load double, double* %src.i.i
; CHECK-NEXT:   store double %src.i.l.i, double* %dst.i.i
; CHECK-NEXT:   %idx.next.i = add nsw i32 %idx.i, 1
; CHECK-NEXT:   %sidx.next.i = add nsw i32 %sidx.i, %incm
; CHECK-NEXT:   %2 = icmp eq i32 %len, %idx.next.i
; CHECK-NEXT:   br i1 %2, label %__enzyme_memcpy_double_32_da0sa0stride.exit, label %for.body.i

; CHECK: __enzyme_memcpy_double_32_da0sa0stride.exit:      ; preds = %entry, %for.body.i
; CHECK-NEXT:   ret double* %0
; CHECK-NEXT: }

; CHECK: define internal void @[[revModFirst]](i32 %len, double* noalias %m, i32 %incm, double* noalias %n, [2 x double*] %"n'", i32 %incn, [2 x double] %differeturn, double*
; CHECK-NEXT: entry:
; CHECK-NEXT:   %1 = extractvalue [2 x double*] %"n'", 0
; CHECK-NEXT:   %2 = extractvalue [2 x double] %differeturn, 0
; CHECK-NEXT:   call void @cblas_daxpy(i32 %len, double %2, double* %0, i32 1, double* %1, i32 %incn)
; CHECK-NEXT:   %3 = extractvalue [2 x double*] %"n'", 1
; CHECK-NEXT:   %4 = extractvalue [2 x double] %differeturn, 1
; CHECK-NEXT:   call void @cblas_daxpy(i32 %len, double %4, double* %0, i32 1, double* %3, i32 %incn)
; CHECK-NEXT:   %5 = bitcast double* %0 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %5)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @[[inactiveModSecond]](i32 %len, double* noalias %m, [2 x double*] %"m'", i32 %incm, double* noalias %n, i32 %incn, [2 x double] %differeturn)
; CHECK-NEXT: entry:
; CHECK:        %call_augmented = call double* @[[augModSecond:.+]](i32 %len, double* %m, [2 x double*] %"m'", i32 %incm, double* %n, i32 %incn)
; CHECK:        call void @[[revModSecond:.+]](i32 %len, double* %m, [2 x double*] %"m'", i32 %incm, double* %n, i32 %incn, [2 x double] %differeturn, double* %call_augmented)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal double* @[[augModSecond]](i32 %len, double* noalias %m, [2 x double*] %"m'", i32 %incm, double* noalias %n, i32 %incn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %mallocsize = mul nuw nsw i32 %len, 8
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i32 %mallocsize)
; CHECK-NEXT:   %0 = bitcast i8* %malloccall to double*
; CHECK-NEXT:   %1 = icmp eq i32 %len, 0
; CHECK-NEXT:   br i1 %1, label %__enzyme_memcpy_double_32_da0sa0stride.exit, label %init.idx.i

; CHECK: init.idx.i:                                       ; preds = %entry
; CHECK-NEXT:   %a.i = sub nsw i32 1, %len
; CHECK-NEXT:   %negidx.i = mul nsw i32 %a.i, %incn
; CHECK-NEXT:   %is.neg.i = icmp slt i32 %incn, 0
; CHECK-NEXT:   %startidx.i = select i1 %is.neg.i, i32 %negidx.i, i32 0
; CHECK-NEXT:   br label %for.body.i
 
; CHECK: for.body.i:                                       ; preds = %for.body.i, %init.idx.i
; CHECK-NEXT:   %idx.i = phi i32 [ 0, %init.idx.i ], [ %idx.next.i, %for.body.i ]
; CHECK-NEXT:   %sidx.i = phi i32 [ %startidx.i, %init.idx.i ], [ %sidx.next.i, %for.body.i ]
; CHECK-NEXT:   %dst.i.i = getelementptr inbounds double, double* %0, i32 %idx.i
; CHECK-NEXT:   %src.i.i = getelementptr inbounds double, double* %n, i32 %sidx.i
; CHECK-NEXT:   %src.i.l.i = load double, double* %src.i.i
; CHECK-NEXT:   store double %src.i.l.i, double* %dst.i.i
; CHECK-NEXT:   %idx.next.i = add nsw i32 %idx.i, 1
; CHECK-NEXT:   %sidx.next.i = add nsw i32 %sidx.i, %incn
; CHECK-NEXT:   %2 = icmp eq i32 %len, %idx.next.i
; CHECK-NEXT:   br i1 %2, label %__enzyme_memcpy_double_32_da0sa0stride.exit, label %for.body.i

; CHECK: __enzyme_memcpy_double_32_da0sa0stride.exit:      ; preds = %entry, %for.body.i
; CHECK-NEXT:   ret double* %0
; CHECK-NEXT: }


; CHECK: define internal void @[[revModSecond]](i32 %len, double* noalias %m, [2 x double*] %"m'", i32 %incm, double* noalias %n, i32 %incn, [2 x double] %differeturn, double*
; CHECK-NEXT: entry:
; CHECK-NEXT:   %1 = extractvalue [2 x double*] %"m'", 0
; CHECK-NEXT:   %2 = extractvalue [2 x double] %differeturn, 0
; CHECK-NEXT:   call void @cblas_daxpy(i32 %len, double %2, double* %0, i32 1, double* %1, i32 %incm)
; CHECK-NEXT:   %3 = extractvalue [2 x double*] %"m'", 1
; CHECK-NEXT:   %4 = extractvalue [2 x double] %differeturn, 1
; CHECK-NEXT:   call void @cblas_daxpy(i32 %len, double %4, double* %0, i32 1, double* %3, i32 %incm)
; CHECK-NEXT:   %5 = bitcast double* %0 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %5)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

