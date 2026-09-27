; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s

; A running index into the strictly lower triangle, as in
;
;   int idx = 0;
;   for (i = 0; i < d; i++)
;     for (j = i + 1; j < d; j++)
;       out[j] += l[idx++] * x[i];
;
; After loop rotation the value of idx at the top of iteration i reaches the
; header through a phi that merges "inner loop skipped" with the inner loop's
; exit value. Scalar evolution cannot classify that phi, so Enzyme used to
; cache it once per outer iteration. The preprocessing now derives its step
; from the inner loop's trip count and rewrites it as a closed form of the
; canonical induction variable, so nothing about it is cached.

define void @tri(i64 %d, ptr noalias %l, ptr noalias %x, ptr noalias %out) {
entry:
  %guard = icmp sgt i64 %d, 0
  br i1 %guard, label %outer, label %exit

outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %idx = phi i64 [ 0, %entry ], [ %idx.next, %outer.latch ]
  %i.next = add nuw nsw i64 %i, 1
  %xi.p = getelementptr inbounds double, ptr %x, i64 %i
  %xi = load double, ptr %xi.p, align 8
  %inner.guard = icmp slt i64 %i.next, %d
  br i1 %inner.guard, label %inner, label %outer.latch

inner:
  %j = phi i64 [ %i.next, %outer ], [ %j.next, %inner ]
  %idx.j = phi i64 [ %idx, %outer ], [ %idx.j.next, %inner ]
  %l.p = getelementptr inbounds double, ptr %l, i64 %idx.j
  %lv = load double, ptr %l.p, align 8
  %o.p = getelementptr inbounds double, ptr %out, i64 %j
  %ov = load double, ptr %o.p, align 8
  %m = fmul double %lv, %xi
  %a = fadd double %ov, %m
  store double %a, ptr %o.p, align 8
  %j.next = add nuw nsw i64 %j, 1
  %idx.j.next = add nuw nsw i64 %idx.j, 1
  %inner.cond = icmp slt i64 %j.next, %d
  br i1 %inner.cond, label %inner, label %inner.exit

inner.exit:
  br label %outer.latch

outer.latch:
  %idx.next = phi i64 [ %idx, %outer ], [ %idx.j.next, %inner.exit ]
  %outer.cond = icmp slt i64 %i.next, %d
  br i1 %outer.cond, label %outer, label %exit

exit:
  ret void
}

define void @dtri(i64 %d, ptr %l, ptr %dl, ptr %x, ptr %dx, ptr %out, ptr %dout) {
  call void (ptr, ...) @__enzyme_autodiff(ptr @tri, i64 %d, ptr %l, ptr %dl, ptr %x, ptr %dx, ptr %out, ptr %dout)
  ret void
}

declare void @__enzyme_autodiff(ptr, ...)

; The index is recomputed from the induction variable in both sweeps: no tape
; is allocated for it, and the triangular closed form i*(i-1)/2 shows up as a
; halving of the counter (the udiv by 2 is folded to a shift).
; CHECK: define internal void @diffetri(
; CHECK-NOT: @malloc
; CHECK: lshr i64 %{{.*}}, 1
; CHECK-NOT: @malloc
; CHECK: ret void
