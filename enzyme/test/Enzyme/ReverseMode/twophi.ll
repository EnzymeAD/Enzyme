; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,adce,%loopmssa(loop-deletion),correlated-propagation,%simplifycfg,adce)" -S | FileCheck %s

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

; CHECK: define internal void @diffesum(i64* %x, i64* %"x'", i64 %n)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %cmp = icmp eq i64 %n, 0
; CHECK-NEXT:   br i1 %cmp, label %one, label %two

; CHECK: one:
; CHECK:   %phi1 = phi i64 [ 0, %entry ], [ %phi2, %two ]
; CHECK-NEXT:   %cmp1 = icmp eq i64 %n, 1
; CHECK-NEXT:   br i1 %cmp1, label %end, label %two

; CHECK: two:
; CHECK:   %phi2 = phi i64 [ 12, %entry ], [ %phi1, %one ]
; CHECK-NEXT:   %cmp2 = icmp eq i64 %n, 2
; CHECK-NEXT:   br i1 %cmp2, label %end, label %one

; CHECK: end:
; CHECK-NEXT:   %phi3 = phi i64 [ %phi1, %one ], [ %phi2, %two ]
; CHECK-NEXT:   store i64 %phi3, i64* %"x'"
; CHECK-NEXT:   store i64 %phi3, i64* %x
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
