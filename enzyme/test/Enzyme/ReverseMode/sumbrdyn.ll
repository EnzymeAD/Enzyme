; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -loop-deletion -correlated-propagation -adce -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,loop(loop-deletion),correlated-propagation,adce,%simplifycfg)" -S | FileCheck %s

declare i1 @exitcond();

; Function Attrs: norecurse nounwind readonly uwtable
define dso_local double @sum(double* nocapture readonly %x, i64 %n) #0 {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret double %add

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %extra ]
  %total.07 = phi double [ 0.000000e+00, %entry ], [ %add, %extra ]
  %arrayidx = getelementptr inbounds double, double* %x, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 8
  %add = fadd fast double %0, %total.07
  %indvars.iv.next = add nuw i64 %indvars.iv, 1
  %exitcond = call i1 @exitcond();
  br i1 %exitcond, label %for.cond.cleanup, label %extra

extra:
  br label %for.body
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
; CHECK-NEXT:   br label %for.body

; CHECK: for.body:
; CHECK-NEXT:   %[[iv:.+]] = phi i64 [ 0, %entry ], [ %[[ivnext:.+]], %for.body ]
; CHECK-NEXT:   %[[ivnext]] = add nuw nsw i64 %[[iv]], 1
; CHECK-NEXT:   %exitcond = call i1 @exitcond()
; CHECK-NEXT:   br i1 %exitcond, label %invertfor.body, label %for.body

; CHECK:  invertentry:
; CHECK-NEXT:  ret void

; CHECK: invertfor.body:
; CHECK-NEXT:   %[[antivar:.+]] = phi i64 [ %[[sub:.+]], %incinvertfor.body ], [ %[[iv]], %for.body ]
; CHECK-NEXT:   %[[arrayidxipgi:.+]] = getelementptr inbounds double, double* %"x'", i64 %[[antivar]]
; CHECK-NEXT:   %[[load:.+]] = load double, double* %[[arrayidxipgi]]
; CHECK-NEXT:   %[[add:.+]] = fadd fast double %[[load]], %differeturn
; CHECK-NEXT:   store double %[[add]], double* %[[arrayidxipgi]]
; CHECK-NEXT:   %[[cmp:.+]] = icmp eq i64 %[[antivar]], 0
; CHECK-NEXT:   br i1 %[[cmp]], label %invertentry, label %incinvertfor.body

; CHECK: incinvertfor.body:
; CHECK-NEXT:   %[[sub]] = add nsw i64 %[[antivar]], -1
; CHECK-NEXT:   br label %invertfor.body