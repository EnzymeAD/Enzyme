; RUN: %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -enzyme-detect-readthrow=0 -passes="enzyme" -S | FileCheck %s

; The derivative of a memmove whose source and destination overlap, as of
; t(1:3) = t(2:4): the derivative of each element of the destination must be
; taken (and zeroed) before it gains that of the element copied to it.
define void @shift(ptr %t, i64 %bytes) {
entry:
  %src = getelementptr inbounds i8, ptr %t, i64 8
  tail call void @llvm.memmove.p0.p0.i64(ptr align 8 %t, ptr align 8 %src, i64 %bytes, i1 false), !tbaa !0
  ret void
}

declare void @llvm.memmove.p0.p0.i64(ptr, ptr, i64, i1)

define void @dshift(ptr %t, ptr %dt, i64 %bytes) {
entry:
  tail call void (...) @__enzyme_autodiff(ptr @shift, ptr %t, ptr %dt, i64 %bytes)
  ret void
}

declare void @__enzyme_autodiff(...)

!0 = !{!1, !1, i64 0}
!1 = !{!"double", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C++ TBAA"}

; The source is above the destination, so the elements are visited from the
; last one.
; CHECK: define internal void @diffeshift(ptr %t, ptr %"t'", i64 %bytes)
; CHECK: for.body.i:
; CHECK-NEXT:   %idx.i = phi i64
; CHECK:   %[[n1:.+]] = sub nuw i64 %[[num:.+]], 1
; CHECK-NEXT:   %[[last:.+]] = sub nuw i64 %[[n1]], %idx.i
; CHECK-NEXT:   %dst.i.i = getelementptr inbounds double, ptr %"t'", i64 %[[last]]
; CHECK-NEXT:   %dst.i.l.i = load double, ptr %dst.i.i, align 8
; CHECK-NEXT:   store double 0.000000e+00, ptr %dst.i.i, align 8
; CHECK-NEXT:   %src.i.i = getelementptr inbounds double, ptr %"src'ipg", i64 %[[last]]

; CHECK: define internal void @__enzyme_memmoveadd_doubleda8sa8(ptr {{.*}}%dst, ptr {{.*}}%src, i64 %num)
; CHECK: for.body:
; CHECK-NEXT:   %idx = phi i64 [ 0, %entry ], [ %idx.next, %for.body ]
; CHECK-NEXT:   %[[srci:.+]] = ptrtoint ptr %src to i64
; CHECK-NEXT:   %[[dsti:.+]] = ptrtoint ptr %dst to i64
; CHECK-NEXT:   %backward = icmp ugt i64 %[[srci]], %[[dsti]]
; CHECK-NEXT:   %[[m1:.+]] = sub nuw i64 %num, 1
; CHECK-NEXT:   %[[mlast:.+]] = sub nuw i64 %[[m1]], %idx
; CHECK-NEXT:   %elem = select i1 %backward, i64 %[[mlast]], i64 %idx
; CHECK-NEXT:   %dst.i = getelementptr inbounds double, ptr %dst, i64 %elem
; CHECK-NEXT:   %dst.i.l = load double, ptr %dst.i, align 8
; CHECK-NEXT:   store double 0.000000e+00, ptr %dst.i, align 8
; CHECK-NEXT:   %src.i = getelementptr inbounds double, ptr %src, i64 %elem
; CHECK-NEXT:   %src.i.l = load double, ptr %src.i, align 8
; CHECK-NEXT:   %[[add:.+]] = fadd fast double %src.i.l, %dst.i.l
; CHECK-NEXT:   store double %[[add]], ptr %src.i, align 8
; CHECK-NEXT:   %idx.next = add nuw i64 %idx, 1
