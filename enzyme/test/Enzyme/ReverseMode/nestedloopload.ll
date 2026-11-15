; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

; x[i] and x[j] in `for i, for j` over the same range: their addresses are add
; recurrences with the same start, step and trip count, but over the outer and
; the inner loop. The reverse pass must not reuse the cache of one load for the
; other, as the values differ at every (i, j) with i != j.

define internal double @g(double %a, double %b) noinline {
entry:
  %m = fmul double %a, %b
  %m2 = fmul double %m, %b
  ret double %m2
}

define double @f(ptr %x, i64 %n) {
entry:
  br label %outer

outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %acc.o = phi double [ 0.0, %entry ], [ %acc.i.lcssa, %outer.latch ]
  %pi = getelementptr inbounds double, ptr %x, i64 %i
  br label %inner

inner:
  %j = phi i64 [ 0, %outer ], [ %j.next, %inner ]
  %acc.i = phi double [ %acc.o, %outer ], [ %acc.n, %inner ]
  %a = load double, ptr %pi, align 8
  %pj = getelementptr inbounds double, ptr %x, i64 %j
  %b = load double, ptr %pj, align 8
  %m2 = call double @g(double %a, double %b)
  %acc.n = fadd double %acc.i, %m2
  %j.next = add nuw nsw i64 %j, 1
  %j.done = icmp eq i64 %j.next, %n
  br i1 %j.done, label %outer.latch, label %inner

outer.latch:
  %acc.i.lcssa = phi double [ %acc.n, %inner ]
  %i.next = add nuw nsw i64 %i, 1
  %i.done = icmp eq i64 %i.next, %n
  br i1 %i.done, label %exit, label %outer

exit:
  %res = phi double [ %acc.i.lcssa, %outer.latch ]
  store double 0.000000e+00, ptr %x, align 8
  ret double %res
}

declare void @__enzyme_autodiff(...)

define void @dtop(ptr %x, ptr %dx, i64 %n) {
entry:
  call void (...) @__enzyme_autodiff(ptr @f, ptr %x, ptr %dx, i64 %n)
  ret void
}

; CHECK: define internal void @diffef(
; CHECK:        %a_cache = alloca ptr
; CHECK:        %b_cache = alloca ptr
; CHECK:      invertinner:
; CHECK:        load ptr, ptr %a_cache
; CHECK:        %[[av:.+]] = load double, ptr
; CHECK:        load ptr, ptr %b_cache
; CHECK:        %[[bv:.+]] = load double, ptr
; CHECK:        call { double, double } @diffeg(double %[[av]], double %[[bv]], double
