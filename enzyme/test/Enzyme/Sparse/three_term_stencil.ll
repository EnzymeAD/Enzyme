; RUN: if [ %llvmver -ge 16 ]; then (%opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-auto-sparsity=1 -S 2>&1 || true) | FileCheck %s; fi

; Columns of the Jacobian of y[i+1] = -x[i] - 3 * x[i+2] + 2 * x[i+4] for
; i = 0, ..., n - 5. Column %c of the seed is non-zero at element %c, so the
; inner loop has to run for %i = %c, %i = %c - 2 and %i = %c - 4.
;
; The loader divides the byte offset with an `ashr exact`, so the comparisons
; for the shifted terms are of a `sext i61` that cannot be solved. They sit
; under the `not` of the `!(all terms zero)` check in the store. Defaulting
; them like float comparisons kept only %i = %c and dropped the other two
; columns. The over-approximation keeps all indices, so the loop stays dense.

declare ptr @__enzyme_todense(...)

declare void @accumulate(i64, i64, double, ptr) #0

define internal double @seed_load(i64 %off, i64 %col) #1 {
  %idx = ashr exact i64 %off, 3
  %c = icmp eq i64 %idx, %col
  %r = uitofp i1 %c to double
  ret double %r
}

define internal void @nostore(double %v, i64 %off, i64 %col) #1 {
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
  %end = add nsw i64 %n, -5
  br label %outer

outer:
  %c = phi i64 [ 0, %entry ], [ %c.next, %outer.latch ]
  %dx = call ptr (...) @__enzyme_todense(ptr @seed_load, ptr @nostore, i64 %c)
  %dy = call ptr (...) @__enzyme_todense(ptr @zero_load, ptr @acc_store, i64 %c, ptr %acc)
  br label %inner

inner:
  %i = phi i64 [ 0, %outer ], [ %i.next, %inner ]
  %i.next = add nuw nsw i64 %i, 1
  %ip2 = add nuw nsw i64 %i, 2
  %ip4 = add nuw nsw i64 %i, 4
  %pa = getelementptr inbounds double, ptr %dx, i64 %i
  %a = load double, ptr %pa, align 8
  %pb = getelementptr inbounds double, ptr %dx, i64 %ip2
  %b = load double, ptr %pb, align 8
  %pc = getelementptr inbounds double, ptr %dx, i64 %ip4
  %cc = load double, ptr %pc, align 8
  %b3 = fmul fast double %b, 3.000000e+00
  %c2 = fmul fast double %cc, 2.000000e+00
  %s1 = fadd fast double %a, %b3
  %s2 = fsub fast double %c2, %s1
  %q = getelementptr inbounds double, ptr %dy, i64 %i.next
  store double %s2, ptr %q, align 8
  %done = icmp eq i64 %i, %end
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

; CHECK: No sparsification: not sparse solvable(nosoltn): solutions:All
; CHECK-NOT: call void @jac.inner
