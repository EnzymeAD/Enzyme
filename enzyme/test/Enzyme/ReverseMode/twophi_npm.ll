; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,inline,mem2reg,instsimplify,adce,loop-deletion,correlated-propagation,simplifycfg,adce" -enzyme-preopt=false -S | FileCheck %s

define void @sum(i64* %x, i64 %n) {
entry:
  %cmp = icmp eq i64 %n, 0
  br i1 %cmp, label %one, label %two

one:
  %phi1 = phi i64 [ 0, %entry ], [ %phi2, %two ]
  %cmp1 = icmp eq i64 %n, 1
  br i1 %cmp1, label %end, label %two

two:
  %phi2 = phi i64 [ 12, %entry ], [ %phi1, %one ]
  %cmp2 = icmp eq i64 %n, 2
  br i1 %cmp2, label %end, label %one

end:
  %phi3 = phi i64 [ %phi1, %one ], [ %phi2, %two ]
  store i64 %phi3, i64* %x
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local void @dsum(i64* %x, i64* %xp, i64 %n) local_unnamed_addr #1 {
entry:
  call void (void (i64*, i64)*, ...) @__enzyme_autodiff(void (i64*, i64)* nonnull @sum, metadata !"enzyme_dup", i64* %x, i64* %xp, i64 %n)
  ret void
}

; Function Attrs: nounwind
declare void @__enzyme_autodiff(void (i64*, i64)*, ...) #2

attributes #0 = { norecurse nounwind readonly uwtable }
attributes #1 = { nounwind uwtable } 
attributes #2 = { nounwind }

; CHECK: define dso_local void @dsum(i64* %x, i64* %xp, i64 %n) local_unnamed_addr #0 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %cmp.i = icmp eq i64 %n, 0
; CHECK-NEXT:   br i1 %cmp.i, label %one.i, label %two.i

; CHECK: one.i:                                            ; preds = %two.i, %entry
; CHECK-NEXT:   %phi1.i = phi i64 [ 0, %entry ], [ %phi2.i, %two.i ]
; CHECK-NEXT:   %cmp1.i = icmp eq i64 %n, 1
; CHECK-NEXT:   br i1 %cmp1.i, label %invertone.i.critedge, label %two.i

; CHECK: two.i:                                            ; preds = %one.i, %entry
; CHECK-NEXT:   %phi2.i = phi i64 [ 12, %entry ], [ %phi1.i, %one.i ]
; CHECK-NEXT:   %cmp2.i = icmp eq i64 %n, 2
; CHECK-NEXT:   br i1 %cmp2.i, label %end.i, label %one.i

; CHECK: end.i:                                            ; preds = %two.i
; CHECK-NEXT:   store i64 %phi2.i, i64* %xp, align 4
; CHECK-NEXT:   store i64 %phi2.i, i64* %x, align 4
; CHECK-NEXT:   ret void

; CHECK: invertone.i.critedge:                             ; preds = %one.i
; CHECK-NEXT:   store i64 %phi1.i, i64* %xp, align 4
; CHECK-NEXT:   store i64 %phi1.i, i64* %x, align 4
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
