; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s; fi

; The remainder loop of a vectorized max reduction: its start value is a phi
; merging "vector loop skipped" with the vector loop's result, which cannot be
; recomputed in the reverse pass.
define void @remmax(ptr %x, i64 %n, i1 %skip, ptr %out) {
entry:
  %x0 = load double, ptr %x, align 8, !tbaa !0
  br i1 %skip, label %rem, label %vec

vec:
  %i = phi i64 [ 1, %entry ], [ %i.next, %vec ]
  %a = phi double [ %x0, %entry ], [ %a.next, %vec ]
  %p = getelementptr inbounds double, ptr %x, i64 %i
  %xi = load double, ptr %p, align 8, !tbaa !0
  %c = fcmp olt double %a, %xi
  %a.next = select i1 %c, double %xi, double %a
  %i.next = add nuw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %rem, label %vec

rem:
  %j = phi i64 [ %j.next, %rem ], [ 0, %entry ], [ 0, %vec ]
  %b = phi double [ %b.next, %rem ], [ %x0, %entry ], [ %a.next, %vec ]
  %q = getelementptr inbounds double, ptr %x, i64 %j
  %xj = load double, ptr %q, align 8, !tbaa !0
  %c2 = fcmp olt double %b, %xj
  %b.next = select i1 %c2, double %xj, double %b
  %j.next = add nuw i64 %j, 1
  %d2 = icmp eq i64 %j.next, %n
  br i1 %d2, label %exit, label %rem

exit:
  %pos = fcmp ogt double %b, 0.000000e+00
  %r = select i1 %pos, double %b.next, double 0.000000e+00
  store double %r, ptr %out, align 8, !tbaa !0
  ret void
}

define void @dremmax(ptr %x, ptr %dx, i64 %n, i1 %skip, ptr %out, ptr %dout) {
  call void (...) @__enzyme_autodiff(ptr @remmax, ptr %x, ptr %dx, i64 %n, i1 %skip, ptr %out, ptr %dout)
  ret void
}

declare void @__enzyme_autodiff(...)

!0 = !{!1, !1, i64 0}
!1 = !{!"double", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C++ TBAA"}

; The start value of the remainder loop's reduction cannot be recomputed. A
; post-cut pass used to re-add it to the recompute graph although it had
; already been chosen for caching (an assertion in computeMinCache).
; CHECK: define internal void @diffe{{.*}}remmax(
; CHECK: %b.ph = phi double [ %x0, %entry ], [ %a.next, %rem.preheader.loopexit ]
; CHECK: invertrem.preheader:
; CHECK-NEXT: %[[d:.+]] = load double, ptr %"b.ph'de"
