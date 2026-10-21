; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-auto-sparsity=1 -S | FileCheck %s; fi

; Hessian-like product of two seeds, y[i] = xa[i+2] * xb[i+j], where the index
; j is a 32-bit value widened to 64 bits (zext (trunc %j)). Seed xa is non-zero
; at element %ca and seed xb at element %cb, so the product is non-zero only
; for %i = %ca - 2, and only if %ca - 2 == %cb - j. The second condition does
; not depend on %i; it is only recognized as such by looking through the casts.
; The inner loop then runs once per column, guarded by that condition.

declare ptr @__enzyme_todense(...)

declare void @accumulate(i64, i64, double, ptr) #0

define internal double @seed_load(i64 %off, i64 %col) #1 {
  %idx = lshr exact i64 %off, 3
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

define void @hess(i64 %n, i64 %j, i64 %ca, i64 %cb, ptr %acc) {
entry:
  %j32 = trunc i64 %j to i32
  %jidx = zext i32 %j32 to i64
  br label %outer

outer:
  %c = phi i64 [ 0, %entry ], [ %c.next, %outer.latch ]
  %dxa = call ptr (...) @__enzyme_todense(ptr @seed_load, ptr @nostore, i64 %ca)
  %dxb = call ptr (...) @__enzyme_todense(ptr @seed_load, ptr @nostore, i64 %cb)
  %dy = call ptr (...) @__enzyme_todense(ptr @zero_load, ptr @acc_store, i64 %c, ptr %acc)
  br label %inner

inner:
  %i = phi i64 [ 0, %outer ], [ %i.next, %inner ]
  %i.next = add nuw nsw i64 %i, 1
  %i2 = add nuw nsw i64 %i, 2
  %ij = add nuw nsw i64 %i, %jidx
  %pa = getelementptr inbounds double, ptr %dxa, i64 %i2
  %a = load double, ptr %pa, align 8
  %pb = getelementptr inbounds double, ptr %dxb, i64 %ij
  %b = load double, ptr %pb, align 8
  %d = fmul fast double %a, %b
  %q = getelementptr inbounds double, ptr %dy, i64 %i
  store double %d, ptr %q, align 8
  %done = icmp eq i64 %i2, %n
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

; CHECK: define void @hess(i64 %n, i64 %j, i64 %ca, i64 %cb, ptr {{.*}}%acc)
; CHECK: entry:
; CHECK-DAG:   %[[J:.+]] = and i64 %j, 4294967295
; CHECK-DAG:   %[[CAM2:.+]] = add i64 %ca, -2
; CHECK-DAG:   %[[DIFF:.+]] = sub i64 %{{.+}}, %cb
; CHECK: outer:
; CHECK:   %[[COND:.+]] = icmp eq i64 %[[DIFF]], 0
; CHECK:   br i1 %{{.+}}, label %tostore, label %poststore
; CHECK: tostore:
; CHECK-NEXT:   call void @hess.inner(i64 %{{[^,]+}}, i64 %[[CAM2]], {{.*}}ptr %acc, i64 %n)
; CHECK-NOT: call void @hess.inner
; CHECK: define internal void @hess.inner(
