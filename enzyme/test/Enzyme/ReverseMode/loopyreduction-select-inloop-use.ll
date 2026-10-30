; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi

; A running max whose every intermediate value is also stored inside the loop.
; The reduction rewrite (differentiate through the index of the last selecting
; iteration) is only valid when the select's intermediate values are not used
; inside the loop, so this loop must take the ordinary select adjoint: the
; condition is cached per iteration and each iteration's adjoint goes to the
; element selected at that iteration.

define void @runmax(ptr noalias %x, ptr noalias %out, i64 %n) {
entry:
  %x0 = load double, ptr %x, align 8, !tbaa !0
  br label %loop

loop:
  %i = phi i64 [ 1, %entry ], [ %i.next, %loop ]
  %m = phi double [ %x0, %entry ], [ %m.next, %loop ]
  %p = getelementptr inbounds double, ptr %x, i64 %i
  %xi = load double, ptr %p, align 8, !tbaa !0
  %c = fcmp olt double %m, %xi
  %m.next = select i1 %c, double %xi, double %m
  %q = getelementptr inbounds double, ptr %out, i64 %i
  store double %m.next, ptr %q, align 8, !tbaa !0
  %i.next = add nuw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

define void @drunmax(ptr %x, ptr %dx, ptr %out, ptr %dout, i64 %n) {
  call void (...) @__enzyme_autodiff(ptr @runmax, ptr %x, ptr %dx, ptr %out, ptr %dout, i64 %n)
  ret void
}

declare void @__enzyme_autodiff(...)

!0 = !{!1, !1, i64 0}
!1 = !{!"double", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C++ TBAA"}

; CHECK: define internal void @differunmax(ptr noalias nocapture readonly %x, ptr nocapture %"x'", ptr noalias nocapture writeonly %out, ptr nocapture %"out'", i64 %n)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %x0 = load double, ptr %x, align 8
; CHECK-NEXT:   %[[nm2:.+]] = add i64 %n, -2
; CHECK-NEXT:   %[[cnt:.+]] = add nsw i64 %[[nm2]], 1
; CHECK-NEXT:   %[[ccache:.+]] = {{(tail )?}}call noalias nonnull ptr @malloc(i64 %[[cnt]]){{.*}}
; CHECK-NEXT:   br label %loop

; CHECK: loop:
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ]
; CHECK-NEXT:   %m = phi double [ %x0, %entry ], [ %m.next, %loop ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %p = getelementptr inbounds double, ptr %x, i64 %iv.next
; CHECK-NEXT:   %xi = load double, ptr %p, align 8
; CHECK-NEXT:   %c = fcmp olt double %m, %xi
; CHECK-NEXT:   %m.next = select i1 %c, double %xi, double %m
; CHECK-NEXT:   %q = getelementptr inbounds double, ptr %out, i64 %iv.next
; CHECK-NEXT:   store double %m.next, ptr %q, align 8
; CHECK-NEXT:   %[[cslot:.+]] = getelementptr inbounds i1, ptr %[[ccache]], i64 %iv
; CHECK-NEXT:   store i1 %c, ptr %[[cslot]], align 1
; CHECK-NEXT:   %i.next = add nuw i64 %iv.next, 1
; CHECK-NEXT:   %done = icmp eq i64 %i.next, %n
; CHECK-NEXT:   br i1 %done, label %invertloop, label %loop

; CHECK: invertentry:
; CHECK-NEXT:   %[[dx0:.+]] = load double, ptr %"x'", align 8
; CHECK-NEXT:   %[[dx0n:.+]] = fadd fast double %[[dx0]], %[[x0de:.+]]
; CHECK-NEXT:   store double %[[dx0n]], ptr %"x'", align 8
; CHECK-NEXT:   {{(tail )?}}call void @free(ptr nonnull %[[ccache]]){{.*}}
; CHECK-NEXT:   ret void

; CHECK: invertloop:
; CHECK-NEXT:   %"m.next'de.0" = phi double [ %[[mcarry:.+]], %incinvertloop ], [ 0.000000e+00, %loop ]
; CHECK-NEXT:   %"x0'de.0" = phi double [ %[[x0de]], %incinvertloop ], [ 0.000000e+00, %loop ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %[[ivdec:.+]], %incinvertloop ], [ %[[nm2]], %loop ]
; CHECK-NEXT:   %iv.next_unwrap = add nuw nsw i64 %"iv'ac.0", 1
; CHECK-NEXT:   %"q'ipg_unwrap" = getelementptr inbounds double, ptr %"out'", i64 %iv.next_unwrap
; CHECK-NEXT:   %[[dq:.+]] = load double, ptr %"q'ipg_unwrap", align 8
; CHECK-NEXT:   store double 0.000000e+00, ptr %"q'ipg_unwrap", align 8
; CHECK-NEXT:   %[[adj:.+]] = fadd fast double %"m.next'de.0", %[[dq]]
; CHECK-NEXT:   %[[rslot:.+]] = getelementptr inbounds i1, ptr %[[ccache]], i64 %"iv'ac.0"
; CHECK-NEXT:   %[[ci:.+]] = load i1, ptr %[[rslot]], align 1{{.*}}
; CHECK-NEXT:   %[[toxi:.+]] = select fast i1 %[[ci]], double %[[adj]], double 0.000000e+00
; CHECK-NEXT:   %[[tom:.+]] = select fast i1 %[[ci]], double 0.000000e+00, double %[[adj]]
; CHECK-NEXT:   %"p'ipg_unwrap" = getelementptr inbounds double, ptr %"x'", i64 %iv.next_unwrap
; CHECK-NEXT:   %[[dp:.+]] = load double, ptr %"p'ipg_unwrap", align 8
; CHECK-NEXT:   %[[dpn:.+]] = fadd fast double %[[dp]], %[[toxi]]
; CHECK-NEXT:   store double %[[dpn]], ptr %"p'ipg_unwrap", align 8
; CHECK-NEXT:   %[[first:.+]] = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   %[[mcarry]] = select fast i1 %[[first]], double 0.000000e+00, double %[[tom]]
; CHECK-NEXT:   %[[x0acc:.+]] = fadd fast double %"x0'de.0", %[[tom]]
; CHECK-NEXT:   %[[x0de]] = select fast i1 %[[first]], double %[[x0acc]], double %"x0'de.0"
; CHECK-NEXT:   br i1 %[[first]], label %invertentry, label %incinvertloop

; CHECK: incinvertloop:
; CHECK-NEXT:   %[[ivdec]] = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertloop
; CHECK-NEXT: }
