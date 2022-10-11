; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,mem2reg,instsimplify,adce,loop-deletion,correlated-propagation,simplifycfg" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: norecurse nounwind readonly uwtable
define dso_local double @sum(double* nocapture readonly %x, i64 %n) #0 {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret double %add

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %total.07 = phi double [ 0.000000e+00, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds double, double* %x, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 8
  %add = fadd fast double %0, %total.07
  %indvars.iv.next = add nuw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv, %n
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; Function Attrs: nounwind uwtable
define dso_local void @dsum(double* %x, double* %xp, i64 %n) local_unnamed_addr #1 {
entry:
  %0 = tail call double (double (double*, i64)*, ...) @__enzyme_autodiff(double (double*, i64)* nonnull @sum, double* %x, double* %xp, i64 %n)
  ret void
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double*, i64)*, ...) #2

attributes #0 = { norecurse nounwind readonly uwtable }
attributes #1 = { nounwind uwtable }
attributes #2 = { nounwind }


; CHECK: define internal void @diffesum(double* nocapture readonly %x, double* nocapture %"x'", i64 %n, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %invertfor.body

; CHECK: invertentry:                                      ; preds = %invertfor.body
; CHECK-NEXT:   ret void

; CHECK: invertfor.body:                                   ; preds = %incinvertfor.body, %entry
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %n, %entry ], [ %4, %incinvertfor.body ]
; CHECK-NEXT:   %"arrayidx'ipg_unwrap" = getelementptr inbounds double, double* %"x'", i64 %"iv'ac.0"
; CHECK-NEXT:   %0 = load double, double* %"arrayidx'ipg_unwrap", align 8
; CHECK-NEXT:   %1 = fadd fast double %0, %differeturn
; CHECK-NEXT:   store double %1, double* %"arrayidx'ipg_unwrap", align 8
; CHECK-NEXT:   %2 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   %3 = select fast i1 %2, double 0.000000e+00, double %differeturn
; CHECK-NEXT:   br i1 %2, label %invertentry, label %incinvertfor.body

; CHECK: incinvertfor.body:                                ; preds = %invertfor.body
; CHECK-NEXT:   %4 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body
; CHECK-NEXT: }