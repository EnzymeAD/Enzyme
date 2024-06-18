; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -loop-deletion -correlated-propagation -adce -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,loop(loop-deletion),correlated-propagation,adce,%simplifycfg)" -S | FileCheck %s

declare void @__enzyme_autodiff(...)

define void @dsquare(double* %arg, double* %arg1) {
bb:
  tail call void (...) @__enzyme_autodiff(double (double*, i64, i64)* nonnull @sum, metadata !"enzyme_dup", double* %arg, double* %arg1, i64 1, i64 10)
  ret void
}

; Function Attrs: nofree noinline
define internal fastcc double @sum(double* nocapture readonly %arg, i64 "enzyme_inactive" "enzyme_type"="{[-1]:Integer}" %arg1, i64 "enzyme_inactive" "enzyme_type"="{[-1]:Integer}" %arg2) {
bb:
  %i8 = icmp eq i64 %arg2, %arg1
  br i1 %i8, label %bb11, label %bb17

bb11:                                             ; preds = %bb
  %i12 = add i64 %arg2, -1
  %i15 = getelementptr inbounds double, double* %arg, i64 %i12
  %i16 = load double, double* %i15, align 8
  br label %bb9

bb17:                                             ; preds = %bb
  %i18 = sub i64 %arg2, %arg1
  %i45 = ashr i64 %i18, 1
  %i46 = add i64 %i45, %arg1
  %i47 = call fastcc double @sum(double* %arg, i64 signext %arg1, i64 signext %i46)
  %i48 = add i64 %i46, 1
  %i49 = call fastcc double @sum(double* %arg, i64 signext %i48, i64 signext %arg2)
  %i50 = fadd double %i47, %i49
  br label %bb9

bb9:                                              ; preds = %bb44, %bb35, %bb20, %bb11
  %i10 = phi double [ %i16, %bb11 ], [ %i50, %bb17 ]
  ret double %i10
}


; CHECK: define internal fastcc void @diffesum(double* nocapture readonly %arg, double* nocapture %"arg'", i64 "enzyme_inactive" "enzyme_type"="{[-1]:Integer}" %arg1, i64 "enzyme_inactive" "enzyme_type"="{[-1]:Integer}" %arg2, double %differeturn)
; CHECK-NEXT: bb:
; CHECK-NEXT:   %i8 = icmp eq i64 %arg2, %arg1
; CHECK-NEXT:   br i1 %i8, label %invertbb9, label %bb17

; CHECK: bb17:                                             ; preds = %bb
; CHECK-NEXT:   %i18 = sub i64 %arg2, %arg1
; CHECK-NEXT:   %i45 = ashr i64 %i18, 1
; CHECK-NEXT:   %i46 = add i64 %i45, %arg1
; CHECK-NEXT:   call fastcc void @augmented_sum(double* %arg, double* %"arg'", i64 signext %arg1, i64 signext %i46)
; CHECK-NEXT:   br label %invertbb9

; CHECK: invertbb:                                         ; preds = %invertbb17, %invertbb11
; CHECK-NEXT:   ret void

; CHECK: invertbb11:                                       ; preds = %invertbb9
; CHECK-NEXT:   %i12_unwrap = add i64 %arg2, -1
; CHECK-NEXT:   %"i15'ipg_unwrap" = getelementptr inbounds double, double* %"arg'", i64 %i12_unwrap
; CHECK-NEXT:   %0 = load double, double* %"i15'ipg_unwrap", align 8
; CHECK-NEXT:   %1 = fadd fast double %0, %3
; CHECK-NEXT:   store double %1, double* %"i15'ipg_unwrap", align 8
; CHECK-NEXT:   br label %invertbb

; CHECK: invertbb17:                                       ; preds = %invertbb9
; CHECK-NEXT:   %i18_unwrap = sub i64 %arg2, %arg1
; CHECK-NEXT:   %i45_unwrap = ashr i64 %i18_unwrap, 1
; CHECK-NEXT:   %i46_unwrap = add i64 %i45_unwrap, %arg1
; CHECK-NEXT:   %i48_unwrap = add i64 %i46_unwrap, 1
; CHECK-NEXT:   call fastcc void @diffesum(double* %arg, double* %"arg'", i64 signext %i48_unwrap, i64 signext %arg2, double %2)
; CHECK-NEXT:   call fastcc void @diffesum.2(double* %arg, double* %"arg'", i64 signext %arg1, i64 signext %i46_unwrap, double %2)
; CHECK-NEXT:   br label %invertbb

; CHECK: invertbb9:                                        ; preds = %bb17, %bb
; CHECK-NEXT:   %2 = select fast i1 %i8, double 0.000000e+00, double %differeturn
; CHECK-NEXT:   %3 = select fast i1 %i8, double %differeturn, double 0.000000e+00
; CHECK-NEXT:   br i1 %i8, label %invertbb11, label %invertbb17
; CHECK-NEXT: }
