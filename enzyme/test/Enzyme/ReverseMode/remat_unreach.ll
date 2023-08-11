; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,adce,loop(loop-deletion),correlated-propagation,%simplifycfg)" -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@enzyme_const = dso_local local_unnamed_addr global i32 0, align 4

declare nonnull i8* @malloc(i64) 

declare void @free(i8*) 
declare void @user(i8*)

define double @_Z15integrate_imagedi(double %arg, i1 %cmp) {
bb:
  br label %bb8

bb8:                                              ; preds = %bb19, %bb
  %i9 = phi double [ 0.000000e+00, %bb ], [ %i13, %bb12 ]
  %i10 = tail call noalias nonnull i8* @malloc(i64 80)
  %i11 = bitcast i8* %i10 to double*
  br i1 %cmp, label %oob, label %bb14

oob:
  call void @user(i8* %i10)
  unreachable

bb14:                                             ; preds = %bb14, %bb8
  %i15 = phi i64 [ %i17, %bb14 ], [ 0, %bb8 ]
  %i16 = getelementptr inbounds double, double* %i11, i64 %i15
  store double %arg, double* %i16, align 8
  %i17 = add nuw nsw i64 %i15, 1
  %i18 = icmp eq i64 %i17, 8
  br i1 %i18, label %bb12, label %bb14

bb12:                                             ; preds = %bb14
  %i13 = load double, double* %i11, align 8
  tail call void @free(i8* nonnull %i10) 
  %i21 = fsub double %i13, %i9
  %i22 = fcmp ogt double %i21, 1.000000e-04
  br i1 %i22, label %bb8, label %bb23

bb23:                                             ; preds = %bb19
  ret double %i13
}

define dso_local double @_Z3dondd(double %arg, double %arg1) {
bb:
  %i2 = tail call double (...) @_Z17__enzyme_autodiffPFddiEz(double (double, i1)* nonnull @_Z15integrate_imagedi, double %arg, i1 false)
  ret double %i2
}

declare dso_local double @_Z17__enzyme_autodiffPFddiEz(...)


; CHECK: define internal { double } @diffe_Z15integrate_imagedi(double %arg, i1 %cmp, double %differeturn)
; CHECK-NEXT: bb:
; CHECK-NEXT:   br label %bb8

; CHECK: bb8: 
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %bb12 ], [ 0, %bb ]
; CHECK-NEXT:   %i9 = phi double [ 0.000000e+00, %bb ], [ %i13, %bb12 ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %i10 = tail call noalias nonnull i8* @malloc(i64 80)
; CHECK-NEXT:   %i11 = bitcast i8* %i10 to double*
; CHECK-NEXT:   br i1 %cmp, label %oob, label %bb14

; CHECK: oob:                                              ; preds = %bb8
; CHECK-NEXT:   call void @user(i8* nonnull %i10)
; CHECK-NEXT:   unreachable

; CHECK: bb14:
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %bb14 ], [ 0, %bb8 ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %i16 = getelementptr inbounds double, double* %i11, i64 %iv1
; CHECK-NEXT:   store double %arg, double* %i16, align 8
; CHECK-NEXT:   %i18 = icmp eq i64 %iv.next2, 8
; CHECK-NEXT:   br i1 %i18, label %bb12, label %bb14

; CHECK: bb12:
; CHECK-NEXT:   %i13 = load double, double* %i11, align 8
; CHECK-NEXT:   tail call void @free(i8* nonnull %i10)
; CHECK-NEXT:   %i21 = fsub double %i13, %i9
; CHECK-NEXT:   %i22 = fcmp ogt double %i21, 1.000000e-04
; CHECK-NEXT:   br i1 %i22, label %bb8, label %remat_enter

; CHECK: invertbb:
; CHECK-NEXT:   %0 = insertvalue { double } undef, double %4, 0
; CHECK-NEXT:   ret { double } %0

; CHECK: incinvertbb8:  
; CHECK-NEXT:   %1 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %remat_enter

; CHECK: invertbb14.preheader:  
; CHECK-NEXT:   call void @free(i8* nonnull %"i10'mi")
; CHECK-NEXT:   call void @free(i8* nonnull %remat_i10)
; CHECK-NEXT:   %2 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %2, label %invertbb, label %incinvertbb8

; CHECK: invertbb14:  
; CHECK-NEXT:   %"arg'de.0" = phi double [ %"arg'de.1", %remat_bb8_bb12_phimerge ], [ %4, %incinvertbb14 ]
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ 7, %remat_bb8_bb12_phimerge ], [ %6, %incinvertbb14 ]
; CHECK-NEXT:   %"i11'ipc_unwrap" = bitcast i8* %"i10'mi" to double*
; CHECK-NEXT:   %"i16'ipg_unwrap" = getelementptr inbounds double, double* %"i11'ipc_unwrap", i64 %"iv1'ac.0"
; CHECK-NEXT:   %3 = load double, double* %"i16'ipg_unwrap", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"i16'ipg_unwrap"
; CHECK-NEXT:   %4 = fadd fast double %"arg'de.0", %3
; CHECK-NEXT:   %5 = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   br i1 %5, label %invertbb14.preheader, label %incinvertbb14

; CHECK: incinvertbb14:                                    ; preds = %invertbb14
; CHECK-NEXT:   %6 = add nsw i64 %"iv1'ac.0", -1
; CHECK-NEXT:   br label %invertbb14

; CHECK: remat_enter: 
; CHECK-NEXT:   %"arg'de.1" = phi double [ %4, %incinvertbb8 ], [ 0.000000e+00, %bb12 ]
; CHECK-NEXT:   %"i13'de.0" = phi double [ 0.000000e+00, %incinvertbb8 ], [ %differeturn, %bb12 ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %1, %incinvertbb8 ], [ %iv, %bb12 ]
; CHECK-NEXT:   %remat_i10 = call noalias nonnull i8* @malloc(i64 80)
; CHECK-NEXT:   %"i10'mi" = call noalias nonnull i8* @malloc(i64 80)
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull dereferenceable(80) dereferenceable_or_null(80) %"i10'mi", i8 0, i64 80, i1 false)
; CHECK-NEXT:   br label %remat_bb8_bb14

; CHECK: remat_bb8_bb14:
; CHECK-NEXT:   %fiv = phi i64 [ 0, %remat_enter ], [ %7, %remat_bb8_bb14 ]
; CHECK-NEXT:   %7 = add i64 %fiv, 1
; CHECK-NEXT:   %i11_unwrap = bitcast i8* %remat_i10 to double*
; CHECK-NEXT:   %i16_unwrap = getelementptr inbounds double, double* %i11_unwrap, i64 %fiv
; CHECK-NEXT:   store double %arg, double* %i16_unwrap, align 8
; CHECK-NEXT:   %i18_unwrap = icmp eq i64 %7, 8
; CHECK-NEXT:   br i1 %i18_unwrap, label %remat_bb8_bb12_phimerge, label %remat_bb8_bb14

; CHECK: remat_bb8_bb12_phimerge: 
; CHECK-NEXT:   %"i11'ipc_unwrap5" = bitcast i8* %"i10'mi" to double*
; CHECK-NEXT:   %8 = load double, double* %"i11'ipc_unwrap5"
; CHECK-NEXT:   %9 = fadd fast double %8, %"i13'de.0"
; CHECK-NEXT:   store double %9, double* %"i11'ipc_unwrap5"
; CHECK-NEXT:   br label %invertbb14
; CHECK-NEXT: }
