; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s; fi

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

; The same loop with the index written in closed form by the programmer:
;
;   for (i = 0; i < d; i++) {
;     idx = i * (2 * d - i - 1) / 2;
;     for (j = i + 1; j < d; j++)
;       out[j] += l[idx++] * x[i];
;   }
;
; Here scalar evolution classifies everything, but the start of the inner
; index is a quadratic recurrence of the outer loop. When the inner index is
; rewritten in terms of the inner canonical induction variable, SCEVExpander
; expands that quadratic start literally, as a new loop-carried phi of the
; outer loop with a varying step, which is again not recomputable and would
; be cached once per outer iteration. The recurrence is now closed before the
; expansion, so no such phi appears.

define void @tri_closed(i64 %d, ptr noalias %l, ptr noalias %x, ptr noalias %out) {
entry:
  %guard = icmp sgt i64 %d, 0
  br i1 %guard, label %outer, label %exit

outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %i.next = add nuw nsw i64 %i, 1
  %d2 = shl nsw i64 %d, 1
  %t0 = sub nsw i64 %d2, %i
  %t1 = sub nsw i64 %t0, 1
  %t2 = mul nsw i64 %t1, %i
  %idx = lshr i64 %t2, 1
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
  br i1 %inner.cond, label %inner, label %outer.latch

outer.latch:
  %outer.cond = icmp slt i64 %i.next, %d
  br i1 %outer.cond, label %outer, label %exit

exit:
  ret void
}

define void @dtri_closed(i64 %d, ptr %l, ptr %dl, ptr %x, ptr %dx, ptr %out, ptr %dout) {
  call void (ptr, ...) @__enzyme_autodiff(ptr @tri_closed, i64 %d, ptr %l, ptr %dl, ptr %x, ptr %dx, ptr %out, ptr %dout)
  ret void
}

; A cubic start: the running output index of row i begins at the tetrahedral
; number i*(i+1)*(i+2)/6, so the inner index phi's start is a degree-three
; recurrence of the outer loop. LLVM's own closed form would need an i65
; canonical induction variable; ours stays in i64.

define void @tetra(i64 %d, ptr noalias %l, ptr noalias %x, ptr noalias %out) {
entry:
  %guard = icmp sgt i64 %d, 0
  br i1 %guard, label %outer, label %exit

outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %i.next = add nuw nsw i64 %i, 1
  %i.next2 = add nuw nsw i64 %i, 2
  %p = mul nuw nsw i64 %i, %i.next
  %p2 = mul nuw nsw i64 %p, %i.next2
  %idx = udiv i64 %p2, 6
  %xi.p = getelementptr inbounds double, ptr %x, i64 %i
  %xi = load double, ptr %xi.p, align 8
  br label %inner

inner:
  %j = phi i64 [ 0, %outer ], [ %j.next, %inner ]
  %o.i = phi i64 [ %idx, %outer ], [ %o.i.next, %inner ]
  %l.p = getelementptr inbounds double, ptr %l, i64 %j
  %lv = load double, ptr %l.p, align 8
  %o.p = getelementptr inbounds double, ptr %out, i64 %o.i
  %ov = load double, ptr %o.p, align 8
  %m = fmul double %lv, %xi
  %a = fadd double %ov, %m
  store double %a, ptr %o.p, align 8
  %j.next = add nuw nsw i64 %j, 1
  %o.i.next = add nuw nsw i64 %o.i, 1
  %inner.cond = icmp ult i64 %j.next, %i.next
  br i1 %inner.cond, label %inner, label %outer.latch

outer.latch:
  %outer.cond = icmp slt i64 %i.next, %d
  br i1 %outer.cond, label %outer, label %exit

exit:
  ret void
}

define void @dtetra(i64 %d, ptr %l, ptr %dl, ptr %x, ptr %dx, ptr %out, ptr %dout) {
  call void (ptr, ...) @__enzyme_autodiff(ptr @tetra, i64 %d, ptr %l, ptr %dl, ptr %x, ptr %dx, ptr %out, ptr %dout)
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

; CHECK: define internal void @diffetri_closed(
; CHECK-NOT: @malloc
; CHECK: ret void

; CHECK: define internal void @diffetetra(
; CHECK-NOT: @malloc
; CHECK-NOT: i65
; CHECK: ret void
