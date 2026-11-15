; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi

; A sign that flips every iteration (sign = -sign), as in a cofactor expansion.
; Scalar evolution cannot express it, so the reverse pass used to cache the
; select condition once per iteration (a malloc of n bytes). It is now
; rewritten in closed form, sign_i = sign_0 * (1 - 2 * (i mod 2)), and the
; condition is recomputed from the induction variable.

define double @altsum(ptr noalias %x, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %sign = phi i32 [ 1, %entry ], [ %sign.next, %loop ]
  %acc = phi double [ 0.000000e+00, %entry ], [ %acc.next, %loop ]
  %p = getelementptr inbounds double, ptr %x, i64 %i
  %xi = load double, ptr %p, align 8
  %sq = fmul double %xi, %xi
  %pos = icmp sgt i32 %sign, 0
  %neg = fneg double %sq
  %term = select i1 %pos, double %sq, double %neg
  %acc.next = fadd double %acc, %term
  %sign.next = sub nsw i32 0, %sign
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret double %acc.next
}

define void @daltsum(ptr %x, ptr %dx, i64 %n) {
  %r = call double (...) @__enzyme_autodiff(ptr @altsum, ptr %x, ptr %dx, i64 %n)
  ret void
}

declare double @__enzyme_autodiff(...)

; CHECK: define internal void @diffealtsum(ptr noalias {{.*}}%x, ptr {{.*}}%"x'", i64 %n, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[nm1:.+]] = add i64 %n, -1
; CHECK-NEXT:   br label %loop

; CHECK: loop:
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %done = icmp eq i64 %iv.next, %n
; CHECK-NEXT:   br i1 %done, label %invertloop, label %loop

; CHECK: invertentry:
; CHECK-NEXT:   ret void

; CHECK: invertloop:
; CHECK-NEXT:   %"acc.next'de.0" = phi double [ %[[dnext:.+]], %incinvertloop ], [ %differeturn, %loop ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %[[ivdec:.+]], %incinvertloop ], [ %[[nm1]], %loop ]
; CHECK-NEXT:   %[[i32:.+]] = trunc i64 %"iv'ac.0" to i32
; CHECK-NEXT:   %[[half:.+]] = lshr i32 %[[i32]], 1
; CHECK-NEXT:   %[[half4:.+]] = shl i32 %[[half]], 2
; CHECK-NEXT:   %[[plus1:.+]] = add i32 %[[half4]], 1
; CHECK-NEXT:   %[[twoi:.+]] = shl i32 %[[i32]], 1
; CHECK-NEXT:   %[[sign:.+]] = sub i32 %[[plus1]], %[[twoi]]
; CHECK-NEXT:   %pos_unwrap = icmp sgt i32 %[[sign]], 0
; CHECK-NEXT:   %[[dsq:.+]] = select fast i1 %pos_unwrap, double %"acc.next'de.0", double 0.000000e+00
; CHECK-NEXT:   %[[dneg:.+]] = select fast i1 %pos_unwrap, double 0.000000e+00, double %"acc.next'de.0"
; CHECK-NEXT:   %[[mdneg:.+]] = fneg fast double %[[dneg]]
; CHECK-NEXT:   %[[dsqt:.+]] = fadd fast double %[[dsq]], %[[mdneg]]
; CHECK-NEXT:   %p_unwrap = getelementptr inbounds {{(nuw )?}}double, ptr %x, i64 %"iv'ac.0"
; CHECK-NEXT:   %xi_unwrap = load double, ptr %p_unwrap, align 8{{.*}}
; CHECK-NEXT:   %[[m1:.+]] = fmul fast double %[[dsqt]], %xi_unwrap
; CHECK-NEXT:   %[[m2:.+]] = fmul fast double %[[dsqt]], %xi_unwrap
; CHECK-NEXT:   %[[dxi:.+]] = fadd fast double %[[m1]], %[[m2]]
; CHECK-NEXT:   %"p'ipg_unwrap" = getelementptr inbounds {{(nuw )?}}double, ptr %"x'", i64 %"iv'ac.0"
; CHECK-NEXT:   %[[old:.+]] = load double, ptr %"p'ipg_unwrap", align 8{{.*}}
; CHECK-NEXT:   %[[new:.+]] = fadd fast double %[[old]], %[[dxi]]
; CHECK-NEXT:   store double %[[new]], ptr %"p'ipg_unwrap", align 8{{.*}}
; CHECK-NEXT:   %[[first:.+]] = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   %[[dnext]] = select fast i1 %[[first]], double 0.000000e+00, double %"acc.next'de.0"
; CHECK-NEXT:   br i1 %[[first]], label %invertentry, label %incinvertloop

; CHECK: incinvertloop:
; CHECK-NEXT:   %[[ivdec]] = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertloop
; CHECK-NEXT: }
