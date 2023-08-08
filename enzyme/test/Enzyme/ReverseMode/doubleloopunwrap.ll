; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg)" -S | FileCheck %s

source_filename = "uniq.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$_Z18my__introsort_loopIPdiEvT_S1_T0_ = comdat any

@enzyme_dup = dso_local local_unnamed_addr global i32 0, align 4
@enzyme_dupnoneed = dso_local local_unnamed_addr global i32 0, align 4
@enzyme_out = dso_local local_unnamed_addr global i32 0, align 4

; Function Attrs: uwtable mustprogress
define dso_local void @_Z2f2Pd(double* %__first, i64 %n) {
entry:
  br label %while.body

while.body:                                       ; preds = %_Z29my__unguarded_partition_pivotIPdET_S1_S1_.exit, %entry
  %sub.ptr.sub16 = phi i64 [ %sub.ptr.lhs.cast, %_Z29my__unguarded_partition_pivotIPdET_S1_S1_.exit ], [ 16, %entry ]
  %add.ptr.i = getelementptr inbounds double, double* %__first, i64 %sub.ptr.sub16
  %i2 = load double, double* %add.ptr.i, align 8, !tbaa !2
  store double %i2, double* %__first, align 8, !tbaa !2
  br label %while.cond.i.i

while.cond.i.i:                                   ; preds = %while.cond.i.i, %if.end
  %idx = phi i64 [ 0, %while.body ], [ %idx_inc, %while.cond.i.i ]
  %idx_inc = add i64 %idx, 1
  %__first.addr.0.i.i = sub i64 14, %idx
  %cmp.i.i.i = icmp ult i64 %idx, %n
  br i1 %cmp.i.i.i, label %while.cond.i.i, label %_Z29my__unguarded_partition_pivotIPdET_S1_S1_.exit, !llvm.loop !6

_Z29my__unguarded_partition_pivotIPdET_S1_S1_.exit: ; preds = %while.cond.i.i
  %sub.ptr.lhs.cast = add i64 %__first.addr.0.i.i, 1
  %cmp = icmp ne i64 %idx, 5
  br i1 %cmp, label %while.body, label %while.end, !llvm.loop !8

while.end:                                        ; preds = %_Z29my__unguarded_partition_pivotIPdET_S1_S1_.exit, %if.then, %entry
  ret void
}

; Function Attrs: norecurse uwtable mustprogress
define dso_local i32 @main() {
entry:
  %t = alloca [5 x double], align 16
  %dt = alloca [5 x double], align 16
  %i = bitcast [5 x double]* %t to i8*
  %i1 = bitcast [5 x double]* %dt to i8*
  %i2 = load i32, i32* @enzyme_dup, align 4, !tbaa !9
  %arraydecay = getelementptr inbounds [5 x double], [5 x double]* %t, i64 0, i64 0
  %arraydecay1 = getelementptr inbounds [5 x double], [5 x double]* %dt, i64 0, i64 0
  call void @_Z17__enzyme_autodiffIvJiPdS0_EET_PvDpT0_(i8* bitcast (void (double*, i64)* @_Z2f2Pd to i8*), i32 %i2, double* nonnull %arraydecay, double* nonnull %arraydecay1, i64 4)
  ret i32 0
}

declare dso_local void @_Z17__enzyme_autodiffIvJiPdS0_EET_PvDpT0_(i8*, i32, double*, double*, i64)

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 12.0.1 (git@github.com:llvm/llvm-project 4973ce53ca8abfc14233a3d8b3045673e0e8543c)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
!8 = distinct !{!8, !7}
!9 = !{!10, !10, i64 0}
!10 = !{!"int", !4, i64 0}


; CHECK: define internal void @diffe_Z2f2Pd(double* %__first, double* %"__first'", i64 %n)
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %while.body

; CHECK: while.body:                                       ; preds = %_Z29my__unguarded_partition_pivotIPdET_S1_S1_.exit, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %_Z29my__unguarded_partition_pivotIPdET_S1_S1_.exit ], [ 0, %entry ]
; CHECK-NEXT:   %sub.ptr.sub16 = phi i64 [ %sub.ptr.lhs.cast, %_Z29my__unguarded_partition_pivotIPdET_S1_S1_.exit ], [ 16, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %"add.ptr.i'ipg" = getelementptr inbounds double, double* %"__first'", i64 %sub.ptr.sub16
; CHECK-NEXT:   %add.ptr.i = getelementptr inbounds double, double* %__first, i64 %sub.ptr.sub16
; CHECK-NEXT:   %i2 = load double, double* %add.ptr.i, align 8, !tbaa !2
; CHECK-NEXT:   store double %i2, double* %__first, align 8, !tbaa !2
; CHECK-NEXT:   br label %while.cond.i.i

; CHECK: while.cond.i.i:                                   ; preds = %while.cond.i.i, %while.body
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %while.cond.i.i ], [ 0, %while.body ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %__first.addr.0.i.i = sub i64 14, %iv1
; CHECK-NEXT:   %cmp.i.i.i = icmp ne i64 %iv1, %n
; CHECK-NEXT:   br i1 %cmp.i.i.i, label %while.cond.i.i, label %_Z29my__unguarded_partition_pivotIPdET_S1_S1_.exit, !llvm.loop !6

; CHECK: _Z29my__unguarded_partition_pivotIPdET_S1_S1_.exit: ; preds = %while.cond.i.i
; CHECK-NEXT:   %"__first.addr.0.i.i!manual_lcssa" = phi i64 [ %__first.addr.0.i.i, %while.cond.i.i ]
; CHECK-NEXT:   %sub.ptr.lhs.cast = add i64 %__first.addr.0.i.i, 1
; CHECK-NEXT:   %cmp = icmp ne i64 %iv1, 5
; CHECK-NEXT:   br i1 %cmp, label %while.body, label %while.end, !llvm.loop !8

; CHECK: while.end:                                        ; preds = %_Z29my__unguarded_partition_pivotIPdET_S1_S1_.exit
; CHECK-NEXT:   %0 = phi i64 [ %iv, %_Z29my__unguarded_partition_pivotIPdET_S1_S1_.exit ]
; CHECK-NEXT:   br label %invertwhile.end

; CHECK: invertentry:                                      ; preds = %invertwhile.body_phimerge
; CHECK-NEXT:   ret void

; CHECK: invertwhile.body:                                 ; preds = %invertwhile.cond.i.i
; CHECK-NEXT:   %1 = load double, double* %"__first'", align 8, !tbaa !2
; CHECK-NEXT:   store double 0.000000e+00, double* %"__first'", align 8, !tbaa !2
; CHECK-NEXT:   %2 = fadd fast double 0.000000e+00, %1
; CHECK-NEXT:   %3 = icmp ne i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %3, label %invertwhile.body_phirc, label %invertwhile.body_phirc1

; CHECK: invertwhile.body_phirc:                           ; preds = %invertwhile.body
; CHECK-NEXT:   %4 = sub nuw i64 %"iv'ac.0", 1
; CHECK-NEXT:   %__first.addr.0.i.i_unwrap = sub i64 14, %n
; CHECK-NEXT:   %sub.ptr.lhs.cast_unwrap = add i64 %__first.addr.0.i.i_unwrap, 1
; CHECK-NEXT:   br label %invertwhile.body_phimerge

; CHECK: invertwhile.body_phirc1:                          ; preds = %invertwhile.body
; CHECK-NEXT:   br label %invertwhile.body_phimerge

; CHECK: invertwhile.body_phimerge:                        ; preds = %invertwhile.body_phirc1, %invertwhile.body_phirc
; CHECK-NEXT:   %5 = phi i64 [ %sub.ptr.lhs.cast_unwrap, %invertwhile.body_phirc ], [ 16, %invertwhile.body_phirc1 ]
; CHECK-NEXT:   %"add.ptr.i'ipg_unwrap" = getelementptr inbounds double, double* %"__first'", i64 %5
; CHECK-NEXT:   %6 = load double, double* %"add.ptr.i'ipg_unwrap", align 8, !tbaa !2
; CHECK-NEXT:   %7 = fadd fast double %6, %2
; CHECK-NEXT:   store double %7, double* %"add.ptr.i'ipg_unwrap", align 8, !tbaa !2
; CHECK-NEXT:   %8 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   %9 = xor i1 %8, true
; CHECK-NEXT:   br i1 %8, label %invertentry, label %incinvertwhile.body

; CHECK: incinvertwhile.body:                              ; preds = %invertwhile.body_phimerge
; CHECK-NEXT:   %10 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invert_Z29my__unguarded_partition_pivotIPdET_S1_S1_.exit

; CHECK: invertwhile.cond.i.i:                             ; preds = %mergeinvertwhile.cond.i.i__Z29my__unguarded_partition_pivotIPdET_S1_S1_.exit, %incinvertwhile.cond.i.i
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ %n, %mergeinvertwhile.cond.i.i__Z29my__unguarded_partition_pivotIPdET_S1_S1_.exit ], [ %13, %incinvertwhile.cond.i.i ]
; CHECK-NEXT:   %11 = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   %12 = xor i1 %11, true
; CHECK-NEXT:   br i1 %11, label %invertwhile.body, label %incinvertwhile.cond.i.i

; CHECK: incinvertwhile.cond.i.i:                          ; preds = %invertwhile.cond.i.i
; CHECK-NEXT:   %13 = add nsw i64 %"iv1'ac.0", -1
; CHECK-NEXT:   br label %invertwhile.cond.i.i

; CHECK: invert_Z29my__unguarded_partition_pivotIPdET_S1_S1_.exit: ; preds = %mergeinvertwhile.body_while.end, %incinvertwhile.body
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %0, %mergeinvertwhile.body_while.end ], [ %10, %incinvertwhile.body ]
; CHECK-NEXT:   br label %mergeinvertwhile.cond.i.i__Z29my__unguarded_partition_pivotIPdET_S1_S1_.exit

; CHECK: mergeinvertwhile.cond.i.i__Z29my__unguarded_partition_pivotIPdET_S1_S1_.exit: ; preds = %invert_Z29my__unguarded_partition_pivotIPdET_S1_S1_.exit
; CHECK-NEXT:   br label %invertwhile.cond.i.i

; CHECK: invertwhile.end:                                  ; preds = %while.end
; CHECK-NEXT:   br label %mergeinvertwhile.body_while.end

; CHECK: mergeinvertwhile.body_while.end:                  ; preds = %invertwhile.end
; CHECK-NEXT:   br label %invert_Z29my__unguarded_partition_pivotIPdET_S1_S1_.exit
; CHECK-NEXT: }
