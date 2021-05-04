; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -inline -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -instcombine -simplifycfg -S -jump-threading -instsimplify -simplifycfg -adce -loop-deletion -simplifycfg | FileCheck %s

; Function Attrs: norecurse nounwind readonly uwtable
define dso_local double @sum(double* nocapture readonly %x, i64 %n) #0 {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret double %add

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %extra ]
  %total.07 = phi double [ 0.000000e+00, %entry ], [ %add2, %extra ]
  %arrayidx = getelementptr inbounds double, double* %x, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 8
  %add = fadd fast double %0, %total.07
  %indvars.iv.next = add nuw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv, %n
  br i1 %exitcond, label %for.cond.cleanup, label %extra

extra:
  %res = uitofp i64 %indvars.iv to double
  %add2 = fmul fast double %add, %res
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


; CHECK: define dso_local void @dsum(double* %x, double* %xp, i64 %n) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %[[antiloop:.+]]

; CHECK: [[antiloop]]: 
; CHECK-NEXT:   %[[dadd:.+]] = phi double [ 1.000000e+00, %entry ], [ %[[m0dadd:.+]], %[[incantiloop:.+]] ]
; CHECK-NEXT:   %[[antivar:.+]] = phi i64 [ %n, %entry ], [ %[[sub:.+]], %[[incantiloop]] ] 
; CHECK-NEXT:   %[[arrayidxipgi:.+]] = getelementptr inbounds double, double* %xp, i64 %[[antivar]]
; CHECK-NEXT:   %[[toload:.+]] = load double, double* %[[arrayidxipgi]], align 8
; CHECK-NEXT:   %[[tostore:.+]] = fadd fast double %[[toload]], %[[dadd]]
; CHECK-NEXT:   store double %[[tostore]], double* %[[arrayidxipgi]], align 8
; CHECK-NEXT:   %[[itercmp:.+]] = icmp eq i64 %[[antivar]], 0
; CHECK-NEXT:   br i1 %[[itercmp]], label %diffesum.exit, label %[[incantiloop]]

; CHECK: [[incantiloop]]: 
; CHECK-NEXT:   %[[sub]] = add nsw i64 %[[antivar]], -1
; CHECK-NEXT:   %res_unwrap.i = uitofp i64 %[[sub]] to double
; CHECK-NEXT:   %[[m0dadd]] = fmul fast double %[[dadd]], %res_unwrap.i
; CHECK-NEXT:   br label %[[antiloop]]

; CHECK: diffesum.exit: 
; CHECK-NEXT:   ret void
; CHECK-NEXT: }


