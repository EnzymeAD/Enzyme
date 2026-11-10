; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi

; When a loop-header phi's only active user is a `select` or `fdiv` backedge
; value, the single-exit loop reduction shortcut is only valid if that `select`
; or `fdiv` itself has no other active in-loop users. Otherwise per-iteration
; reverse propagation is required so intermediate iteration adjoints flow
; backward through the phi.

define double @select_inloop_sum(ptr noalias nocapture readonly %x, i64 %n) {
entry:
  %x0 = load double, ptr %x, align 8
  br label %loop

loop:
  %i = phi i64 [ 1, %entry ], [ %i.next, %loop ]
  %m = phi double [ %x0, %entry ], [ %m.next, %loop ]
  %sum = phi double [ 0.000000e+00, %entry ], [ %sum.next, %loop ]
  %p = getelementptr inbounds double, ptr %x, i64 %i
  %xi = load double, ptr %p, align 8
  %c = fcmp olt double %m, %xi
  %m.next = select i1 %c, double %xi, double %m
  %sum.next = fadd double %sum, %m.next
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret double %sum.next
}

define double @fdiv_inloop_sum(ptr noalias nocapture readonly %A, i64 %n, double %start) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %reduce = phi double [ %start, %entry ], [ %div, %loop ]
  %sum = phi double [ 0.000000e+00, %entry ], [ %sum.next, %loop ]
  %p = getelementptr inbounds double, ptr %A, i64 %i
  %ld = load double, ptr %p, align 8
  %div = fdiv double %reduce, %ld
  %sum.next = fadd double %sum, %div
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret double %sum.next
}

define double @caller(ptr %x, ptr %dx, ptr %A, ptr %dA, i64 %n, double %start) {
entry:
  call void (...) @__enzyme_autodiff(ptr @select_inloop_sum, ptr %x, ptr %dx, i64 %n)
  %ds = call double (...) @__enzyme_autodiff.1(ptr @fdiv_inloop_sum, ptr %A, ptr %dA, i64 %n, double %start)
  ret double %ds
}

declare void @__enzyme_autodiff(...)
declare double @__enzyme_autodiff.1(...)

; CHECK-LABEL: define internal void @diffeselect_inloop_sum(
; CHECK: invertloop:
; CHECK-DAG:   %[[mde:.+]] = phi double [ %[[mnext:.+]], %incinvertloop ], [ 0.000000e+00, %loop ]
; CHECK-DAG:   %[[x0de:.+]] = phi double [ %[[x0next:.+]], %incinvertloop ], [ 0.000000e+00, %loop ]
; CHECK-DAG:   %[[sde:.+]] = phi double [ %[[snext:.+]], %incinvertloop ], [ %differeturn, %loop ]
; CHECK-DAG:   %[[iv:.+]] = phi i64 [ %[[ivdec:.+]], %incinvertloop ], [ %{{.+}}, %loop ]
; CHECK:   %[[mtot:.+]] = fadd fast double %[[mde]], %[[sde]]
; CHECK:   %[[cond:.+]] = load i1, ptr %{{.+}}, align 1
; CHECK:   %[[dxi:.+]] = select{{( fast)?}} i1 %[[cond]], double %[[mtot]], double 0.000000e+00
; CHECK:   %[[dm:.+]] = select{{( fast)?}} i1 %[[cond]], double 0.000000e+00, double %[[mtot]]
; CHECK:   %[[ivp1:.+]] = add nuw nsw i64 %[[iv]], 1
; CHECK:   %[[dxp:.+]] = getelementptr inbounds {{(nuw )?}}double, ptr %"x'", i64 %[[ivp1]]
; CHECK:   %[[dxold:.+]] = load double, ptr %[[dxp]], align 8
; CHECK:   %[[dxnew:.+]] = fadd fast double %[[dxold]], %[[dxi]]
; CHECK:   store double %[[dxnew]], ptr %[[dxp]], align 8
; CHECK:   %[[first:.+]] = icmp eq i64 %[[iv]], 0
; CHECK:   %[[mnext]] = select{{( fast)?}} i1 %[[first]], double 0.000000e+00, double %[[dm]]
; CHECK:   %[[x0acc:.+]] = fadd fast double %[[x0de]], %[[dm]]
; CHECK:   %[[x0next]] = select{{( fast)?}} i1 %[[first]], double %[[x0acc]], double %[[x0de]]
; CHECK:   %[[snext]] = select{{( fast)?}} i1 %[[first]], double 0.000000e+00, double %[[sde]]
; CHECK:   br i1 %[[first]], label %invertentry, label %incinvertloop

; CHECK-LABEL: define internal { double } @diffefdiv_inloop_sum(
; CHECK: invertloop:
; CHECK-DAG:   %[[divde:.+]] = phi double [ %[[divnext:.+]], %incinvertloop ], [ 0.000000e+00, %loop ]
; CHECK-DAG:   %[[stde:.+]] = phi double [ %[[stnext:.+]], %incinvertloop ], [ 0.000000e+00, %loop ]
; CHECK-DAG:   %[[fsde:.+]] = phi double [ %[[fsnext:.+]], %incinvertloop ], [ %differeturn, %loop ]
; CHECK-DAG:   %[[fiv:.+]] = phi i64 [ %[[fivdec:.+]], %incinvertloop ], [ %{{.+}}, %loop ]
; CHECK:   %[[dtot:.+]] = fadd fast double %[[divde]], %[[fsde]]
; CHECK:   %[[ap:.+]] = getelementptr inbounds {{(nuw )?}}double, ptr %A, i64 %[[fiv]]
; CHECK:   %[[ald:.+]] = load double, ptr %[[ap]], align 8
; CHECK:   %[[dred:.+]] = fdiv fast double %[[dtot]], %[[ald]]
; CHECK:   %[[dap:.+]] = getelementptr inbounds {{(nuw )?}}double, ptr %"A'", i64 %[[fiv]]
; CHECK:   %[[daold:.+]] = load double, ptr %[[dap]], align 8
; CHECK:   %[[danew:.+]] = fadd fast double %[[daold]], %{{.+}}
; CHECK:   store double %[[danew]], ptr %[[dap]], align 8
; CHECK:   %[[ffirst:.+]] = icmp eq i64 %[[fiv]], 0
; CHECK:   %[[divnext]] = select{{( fast)?}} i1 %[[ffirst]], double 0.000000e+00, double %[[dred]]
; CHECK:   %[[stacc:.+]] = fadd fast double %[[stde]], %[[dred]]
; CHECK:   %[[stnext]] = select{{( fast)?}} i1 %[[ffirst]], double %[[stacc]], double %[[stde]]
; CHECK:   %[[fsnext]] = select{{( fast)?}} i1 %[[ffirst]], double 0.000000e+00, double %[[fsde]]
; CHECK:   br i1 %[[ffirst]], label %invertentry, label %incinvertloop
