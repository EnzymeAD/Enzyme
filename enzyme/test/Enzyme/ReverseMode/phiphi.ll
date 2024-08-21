; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -correlated-propagation -adce -instcombine -instsimplify -early-cse -simplifycfg -correlated-propagation -adce -jump-threading -instsimplify -early-cse -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,correlated-propagation,adce,instcombine,instsimplify,early-cse,%simplifycfg,correlated-propagation,adce,jump-threading,instsimplify,early-cse,%simplifycfg)" -S | FileCheck %s

define double @iter(double* %x) {
top:
  br label %L18

L18:                                              ; preds = %L18, %top.L18_crit_edge
  %iv = phi i64 [ %iv.next, %L18 ], [ 0, %top ]
  %value_phi7 = phi double [ 0.000000e+00, %top ], [ %value_phi8, %L18 ]
  %value_phi8 = phi double [ 0.000000e+00, %top ], [ %i8, %L18 ]
  %iv.next = add nuw nsw i64 %iv, 1
  %gep = getelementptr inbounds double, double* %x, i64 %iv
  %arrayref = load double, double* %gep, align 8
  %i8 = fadd double %value_phi7, %arrayref
  %.not19 = icmp eq i64 %iv.next, 10
  br i1 %.not19, label %exit, label %L18

exit:
  ret double %i8
}

; Function Attrs: nounwind uwtable
define dso_local void @dsincos(double* noalias %x, double* noalias %xp) {
entry:
  %0 = tail call double (...) @__enzyme_autodiff(double (double*)* nonnull @iter, double* %x, double* %xp)
  ret void
}
declare double @__enzyme_autodiff(...)

; CHECK: define internal void @diffeiter(double* %x, double* %"x'", double %differeturn)
; CHECK-NEXT: top:
; CHECK-NEXT:   br label %L18

; CHECK: L18:                                              ; preds = %L18, %top
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %L18 ], [ 0, %top ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %.not19 = icmp eq i64 %iv.next2, 10
; CHECK-NEXT:   br i1 %.not19, label %invertL18, label %L18

; CHECK: inverttop:                                        ; preds = %invertL18
; CHECK-NEXT:   ret void

; CHECK: invertL18:                                        ; preds = %L18, %incinvertL18
; CHECK-NEXT:   %"value_phi8'de.0" = phi double [ %"i8'de.0", %incinvertL18 ], [ 0.000000e+00, %L18 ]
; CHECK-NEXT:   %"i8'de.0" = phi double [ %"value_phi8'de.0", %incinvertL18 ], [ %differeturn, %L18 ]
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ %3, %incinvertL18 ], [ 9, %L18 ]
; CHECK-NEXT:   %"gep'ipg_unwrap" = getelementptr inbounds double, double* %"x'", i64 %"iv1'ac.0"
; CHECK-NEXT:   %0 = load double, double* %"gep'ipg_unwrap", align 8, !alias.scope !0, !noalias !3
; CHECK-NEXT:   %1 = fadd fast double %0, %"i8'de.0"
; CHECK-NEXT:   store double %1, double* %"gep'ipg_unwrap", align 8, !alias.scope !0, !noalias !3
; CHECK-NEXT:   %2 = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   br i1 %2, label %inverttop, label %incinvertL18

; CHECK: incinvertL18:                                     ; preds = %invertL18
; CHECK-NEXT:   %3 = add nsw i64 %"iv1'ac.0", -1
; CHECK-NEXT:   br label %invertL18
; CHECK-NEXT: }

