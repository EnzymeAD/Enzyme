; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi

; A running quotient whose every intermediate value is also stored inside the
; loop. The reduction rewrite (scale the final value's adjoint by the total
; product) is only valid when the fdiv's intermediate values are not used
; inside the loop, so this loop must take the ordinary fdiv adjoint: the running
; value is cached per iteration and each iteration's adjoint is propagated with
; that iteration's quotient.

define void @runquot(ptr noalias %x, ptr noalias %out, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %r = phi double [ 1.000000e+00, %entry ], [ %r.next, %loop ]
  %p = getelementptr inbounds double, ptr %x, i64 %i
  %xi = load double, ptr %p, align 8
  %r.next = fdiv double %r, %xi
  %q = getelementptr inbounds double, ptr %out, i64 %i
  store double %r.next, ptr %q, align 8
  %i.next = add nuw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

define void @drunquot(ptr %x, ptr %dx, ptr %out, ptr %dout, i64 %n) {
  call void (...) @__enzyme_autodiff(ptr @runquot, ptr %x, ptr %dx, ptr %out, ptr %dout, i64 %n)
  ret void
}

declare void @__enzyme_autodiff(...)

; CHECK: define internal void @differunquot(ptr noalias nocapture readonly %x, ptr nocapture %"x'", ptr noalias nocapture writeonly %out, ptr nocapture %"out'", i64 %n)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[nm1:.+]] = add i64 %n, -1
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %n, 8
; CHECK-NEXT:   %[[rcache:.+]] = {{(tail )?}}call noalias nonnull ptr @malloc(i64 %mallocsize){{.*}}
; CHECK-NEXT:   br label %loop

; CHECK: loop:
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ]
; CHECK-NEXT:   %r = phi double [ 1.000000e+00, %entry ], [ %r.next, %loop ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %p = getelementptr inbounds double, ptr %x, i64 %iv
; CHECK-NEXT:   %xi = load double, ptr %p, align 8
; CHECK-NEXT:   %r.next = fdiv double %r, %xi
; CHECK-NEXT:   %q = getelementptr inbounds double, ptr %out, i64 %iv
; CHECK-NEXT:   store double %r.next, ptr %q, align 8
; CHECK-NEXT:   %[[rslot:.+]] = getelementptr inbounds double, ptr %[[rcache]], i64 %iv
; CHECK-NEXT:   store double %r, ptr %[[rslot]], align 8
; CHECK-NEXT:   %done = icmp eq i64 %iv.next, %n
; CHECK-NEXT:   br i1 %done, label %invertloop, label %loop

; CHECK: invertentry:
; CHECK-NEXT:   {{(tail )?}}call void @free(ptr nonnull %[[rcache]]){{.*}}
; CHECK-NEXT:   ret void

; CHECK: invertloop:
; CHECK-NEXT:   %"r.next'de.0" = phi double [ %[[rcarry:.+]], %incinvertloop ], [ 0.000000e+00, %loop ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %[[ivdec:.+]], %incinvertloop ], [ %[[nm1]], %loop ]
; CHECK-NEXT:   %"q'ipg_unwrap" = getelementptr inbounds double, ptr %"out'", i64 %"iv'ac.0"
; CHECK-NEXT:   %[[dq:.+]] = load double, ptr %"q'ipg_unwrap", align 8
; CHECK-NEXT:   store double 0.000000e+00, ptr %"q'ipg_unwrap", align 8
; CHECK-NEXT:   %[[adj:.+]] = fadd fast double %"r.next'de.0", %[[dq]]
; CHECK-NEXT:   %p_unwrap = getelementptr inbounds double, ptr %x, i64 %"iv'ac.0"
; CHECK-NEXT:   %xi_unwrap = load double, ptr %p_unwrap, align 8
; CHECK-NEXT:   %[[tor:.+]] = fdiv fast double %[[adj]], %xi_unwrap
; CHECK-NEXT:   %[[adjx:.+]] = fdiv fast double %[[adj]], %xi_unwrap
; CHECK-NEXT:   %[[rload:.+]] = getelementptr inbounds double, ptr %[[rcache]], i64 %"iv'ac.0"
; CHECK-NEXT:   %[[ri:.+]] = load double, ptr %[[rload]], align 8{{.*}}
; CHECK-NEXT:   %[[rnexti:.+]] = fdiv fast double %[[ri]], %xi_unwrap
; CHECK-NEXT:   %[[prod:.+]] = fmul fast double %[[adjx]], %[[rnexti]]
; CHECK-NEXT:   %[[toxi:.+]] = fneg fast double %[[prod]]
; CHECK-NEXT:   %"p'ipg_unwrap" = getelementptr inbounds double, ptr %"x'", i64 %"iv'ac.0"
; CHECK-NEXT:   %[[dp:.+]] = load double, ptr %"p'ipg_unwrap", align 8
; CHECK-NEXT:   %[[dpn:.+]] = fadd fast double %[[dp]], %[[toxi]]
; CHECK-NEXT:   store double %[[dpn]], ptr %"p'ipg_unwrap", align 8
; CHECK-NEXT:   %[[first:.+]] = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   %[[rcarry]] = select fast i1 %[[first]], double 0.000000e+00, double %[[tor]]
; CHECK-NEXT:   br i1 %[[first]], label %invertentry, label %incinvertloop

; CHECK: incinvertloop:
; CHECK-NEXT:   %[[ivdec]] = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertloop
; CHECK-NEXT: }
