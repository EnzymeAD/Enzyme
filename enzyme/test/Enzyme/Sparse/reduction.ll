; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-auto-sparsity=1 -S | FileCheck %s; fi

; A reduction that stores its partial sum in every iteration, as the derivative
; of `y[0] += 3 * x[i]` does. The inner loop is sparsified to its iteration
; %i = %c; the outlined body runs a single iteration, so the partial sum starts
; from its initial value.

declare ptr @__enzyme_todense(...)

declare void @accumulate(i64, i64, double, ptr) #0

define internal double @seed_load(i64 %off, i64 %coff) #1 {
  %c = icmp eq i64 %off, %coff
  %r = uitofp i1 %c to double
  ret double %r
}

define internal void @nostore(double %v, i64 %off, i64 %coff) #1 {
  ret void
}

define internal double @zero_load(i64 %off, i64 %col, ptr %acc) #1 {
  ret double 0.000000e+00
}

define internal void @acc_store(double %v, i64 %off, i64 %col, ptr %acc) #1 {
entry:
  %nz = fcmp une double %v, 0.000000e+00
  br i1 %nz, label %do, label %end

do:
  call void @accumulate(i64 %off, i64 %col, double %v, ptr %acc)
  br label %end

end:
  ret void
}

define void @jac(i64 %n, ptr %acc) {
entry:
  br label %outer

outer:
  %c = phi i64 [ 0, %entry ], [ %c.next, %outer.latch ]
  %coff = mul nuw nsw i64 %c, 8
  %dx = call ptr (...) @__enzyme_todense(ptr @seed_load, ptr @nostore, i64 %coff)
  %dy = call ptr (...) @__enzyme_todense(ptr @zero_load, ptr @acc_store, i64 %c, ptr %acc)
  br label %inner

inner:
  %i = phi i64 [ 0, %outer ], [ %i.next, %inner ]
  %sum = phi double [ 0.000000e+00, %outer ], [ %sum.next, %inner ]
  %i.next = add nuw nsw i64 %i, 1
  %pa = getelementptr inbounds double, ptr %dx, i64 %i
  %a = load double, ptr %pa, align 8
  %t = fmul fast double %a, 3.000000e+00
  %sum.next = fadd fast double %sum, %t
  store double %sum.next, ptr %dy, align 8
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %outer.latch, label %inner

outer.latch:
  %c.next = add nuw nsw i64 %c, 1
  %cdone = icmp eq i64 %c.next, %n
  br i1 %cdone, label %exit, label %outer

exit:
  ret void
}

attributes #0 = { noinline "enzyme_sparse_accumulate" }
attributes #1 = { alwaysinline }

; CHECK: define void @jac(i64 %n, ptr {{.*}}%acc)
; CHECK:   call void @jac.inner(i64 %{{.+}}, i64 %c,
; CHECK-NOT: call void @jac.inner
; CHECK: define internal void @jac.inner(
; CHECK:   %sum = phi double [ 0.000000e+00, %newFuncRoot ]
