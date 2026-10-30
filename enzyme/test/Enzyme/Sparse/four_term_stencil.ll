; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-auto-sparsity=1 -S | FileCheck %s; fi

; Columns of the Jacobian of y[i+2] = -x[i] - 3 x[i+2] + x[i+4] - 3 x[i+5].
; Column %c of the seed is non-zero at element %c, so the inner loop only has
; to run for %i in {%c, %c - 2, %c - 4, %c - 5}. Each of these must be the
; exact quotient of the byte offsets, also when it is negative.

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
  %i.next = add nuw nsw i64 %i, 1
  %i2 = add nuw nsw i64 %i, 2
  %i4 = add nuw nsw i64 %i, 4
  %i5 = add nuw nsw i64 %i, 5
  %p0 = getelementptr inbounds double, ptr %dx, i64 %i
  %a0 = load double, ptr %p0, align 8
  %p2 = getelementptr inbounds double, ptr %dx, i64 %i2
  %a2 = load double, ptr %p2, align 8
  %p4 = getelementptr inbounds double, ptr %dx, i64 %i4
  %a4 = load double, ptr %p4, align 8
  %p5 = getelementptr inbounds double, ptr %dx, i64 %i5
  %a5 = load double, ptr %p5, align 8
  %t2 = fmul fast double %a2, -3.000000e+00
  %t5 = fmul fast double %a5, -3.000000e+00
  %s0 = fsub fast double %t2, %a0
  %s1 = fadd fast double %s0, %a4
  %s2 = fadd fast double %s1, %t5
  %q = getelementptr inbounds double, ptr %dy, i64 %i2
  store double %s2, ptr %q, align 8
  %done = icmp eq i64 %i5, %n
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
; CHECK: outer:
; CHECK-NEXT:   %c = phi i64 [ 0, %entry ], [ %c.next, %outer.latch ]
; CHECK-DAG:    %[[CM4:.+]] = add i64 %c, -4
; CHECK-DAG:    %[[CM2:.+]] = add i64 %c, -2
; CHECK-DAG:    %[[CM5:.+]] = add i64 %c, -5
; CHECK-DAG:   call void @jac.inner(i64 %{{[^,]+}}, i64 %c, {{.*}}i64 %c,
; CHECK-DAG:   call void @jac.inner(i64 %{{[^,]+}}, i64 %[[CM4]], {{.*}}i64 %c,
; CHECK-DAG:   call void @jac.inner(i64 %{{[^,]+}}, i64 %[[CM2]], {{.*}}i64 %c,
; CHECK-DAG:   call void @jac.inner(i64 %{{[^,]+}}, i64 %[[CM5]], {{.*}}i64 %c,
; CHECK-NOT: call void @jac.inner
; CHECK: define internal void @jac.inner(
