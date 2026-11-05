; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-auto-sparsity=1 -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,default<O2>" -enzyme-auto-sparsity=1 -S | %lli - ; fi

declare ptr @__enzyme_todense(...)

define void @accumulate(i64 %row_bytes, i64 %col, double %val, ptr %acc) #0 {
entry:
  %count = load i64, ptr %acc, align 8
  %next = add i64 %count, 1
  store i64 %next, ptr %acc, align 8
  %sum_ptr = getelementptr inbounds double, ptr %acc, i64 1
  %old_sum = load double, ptr %sum_ptr, align 8
  %new_sum = fadd double %old_sum, %val
  store double %new_sum, ptr %sum_ptr, align 8
  ret void
}

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

define internal void @acc_store_oeq(double %v, i64 %off, i64 %col, ptr %acc) #1 {
entry:
  %z = fcmp oeq double %v, 0.000000e+00
  br i1 %z, label %end, label %do

do:
  call void @accumulate(i64 %off, i64 %col, double %v, ptr %acc)
  br label %end

end:
  ret void
}

define internal void @acc_store_une(double %v, i64 %off, i64 %col, ptr %acc) #1 {
entry:
  %nz = fcmp une double %v, 0.000000e+00
  br i1 %nz, label %do, label %end

do:
  call void @accumulate(i64 %off, i64 %col, double %v, ptr %acc)
  br label %end

end:
  ret void
}

define void @jac_three(i64 %n, ptr %acc) {
entry:
  br label %outer

outer:
  %c = phi i64 [ 0, %entry ], [ %c.next, %outer.latch ]
  %dx = call ptr (...) @__enzyme_todense(ptr @seed_load, ptr @nostore, i64 %c)
  %dy = call ptr (...) @__enzyme_todense(ptr @zero_load, ptr @acc_store_oeq, i64 %c, ptr %acc)
  %v0 = load double, ptr %dx, align 8
  %p1 = getelementptr inbounds double, ptr %dx, i64 1
  %v1 = load double, ptr %p1, align 8
  %s01 = fadd fast double %v0, %v1
  %p2 = getelementptr inbounds double, ptr %dx, i64 2
  %v2 = load double, ptr %p2, align 8
  %sum = fadd fast double %s01, %v2
  store double %sum, ptr %dy, align 8
  br label %outer.latch

outer.latch:
  %c.next = add nuw nsw i64 %c, 1
  %cdone = icmp eq i64 %c.next, %n
  br i1 %cdone, label %exit, label %outer

exit:
  ret void
}

define void @stencil_three(i64 %n, ptr %acc) {
entry:
  br label %outer

outer:
  %c = phi i64 [ 0, %entry ], [ %c.next, %outer.latch ]
  %dx = call ptr (...) @__enzyme_todense(ptr @seed_load, ptr @nostore, i64 %c)
  %dy = call ptr (...) @__enzyme_todense(ptr @zero_load, ptr @acc_store_une, i64 %c, ptr %acc)
  br label %inner

inner:
  %i = phi i64 [ 0, %outer ], [ %i.next, %inner ]
  %i.next = add nuw nsw i64 %i, 1
  %i2 = add nuw nsw i64 %i, 2
  %p0 = getelementptr inbounds double, ptr %dx, i64 %i
  %v0 = load double, ptr %p0, align 8
  %p1 = getelementptr inbounds double, ptr %dx, i64 %i.next
  %v1 = load double, ptr %p1, align 8
  %s01 = fadd fast double %v0, %v1
  %p2 = getelementptr inbounds double, ptr %dx, i64 %i2
  %v2 = load double, ptr %p2, align 8
  %sum = fadd fast double %s01, %v2
  %q = getelementptr inbounds double, ptr %dy, i64 %i.next
  store double %sum, ptr %q, align 8
  %done = icmp eq i64 %i2, %n
  br i1 %done, label %outer.latch, label %inner

outer.latch:
  %c.next = add nuw nsw i64 %c, 1
  %cdone = icmp eq i64 %c.next, %n
  br i1 %cdone, label %exit, label %outer

exit:
  ret void
}

define i32 @main() {
entry:
  %acc = alloca [2 x i64], align 8
  store i64 0, ptr %acc, align 8
  %sum_ptr = getelementptr inbounds double, ptr %acc, i64 1
  store double 0.000000e+00, ptr %sum_ptr, align 8
  call void @jac_three(i64 5, ptr %acc)
  %count1 = load i64, ptr %acc, align 8
  %ok1 = icmp eq i64 %count1, 3
  br i1 %ok1, label %check_sum1, label %fail

check_sum1:
  %sum1 = load double, ptr %sum_ptr, align 8
  %ok_sum1 = fcmp oeq double %sum1, 3.000000e+00
  br i1 %ok_sum1, label %test2, label %fail

test2:
  store i64 0, ptr %acc, align 8
  store double 0.000000e+00, ptr %sum_ptr, align 8
  call void @stencil_three(i64 4, ptr %acc)
  %count2 = load i64, ptr %acc, align 8
  %ok2 = icmp eq i64 %count2, 8
  br i1 %ok2, label %check_sum2, label %fail

check_sum2:
  %sum2 = load double, ptr %sum_ptr, align 8
  %ok_sum2 = fcmp oeq double %sum2, 8.000000e+00
  br i1 %ok_sum2, label %pass, label %fail

pass:
  ret i32 0

fail:
  ret i32 1
}

attributes #0 = { noinline "enzyme_sparse_accumulate" }
attributes #1 = { alwaysinline }

; CHECK-LABEL: define void @jac_three(
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[NM1:.+]] = add i64 %n, -1
; CHECK:   call void @jac_three.outer(i64 %[[NM1]], i64 1, i64 0, ptr %acc, i64 %n)
; CHECK:   call void @jac_three.outer(i64 %[[NM1]], i64 0, i64 0, ptr %acc, i64 %n)
; CHECK:   call void @jac_three.outer(i64 %[[NM1]], i64 2, i64 0, ptr %acc, i64 %n)
; CHECK-NOT: call void @jac_three.outer

; CHECK-LABEL: define void @stencil_three(
; CHECK: outer:
; CHECK-NEXT:   %c = phi i64 [ 0, %entry ], [ %c.next, %outer.latch ]
; CHECK-NEXT:   %[[CM1:.+]] = add i64 %c, -1
; CHECK-NEXT:   %[[CM2:.+]] = add i64 %c, -2
; CHECK:   call void @stencil_three.inner(i64 %{{.+}}, i64 %c, i64 %c, i64 0, ptr %acc, i64 %n)
; CHECK:   call void @stencil_three.inner(i64 %{{.+}}, i64 %[[CM1]], i64 %c, i64 0, ptr %acc, i64 %n)
; CHECK:   call void @stencil_three.inner(i64 %{{.+}}, i64 %[[CM2]], i64 %c, i64 0, ptr %acc, i64 %n)
; CHECK-NOT: call void @stencil_three.inner
