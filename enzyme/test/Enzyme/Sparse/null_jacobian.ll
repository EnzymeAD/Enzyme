; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-postopt=1 -enzyme-auto-sparsity=1 -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-postopt=1 -enzyme-auto-sparsity=1 -S | %lli - ; fi

declare ptr @__enzyme_todense(...)
declare void @__enzyme_fwddiff(...)

define void @accumulate(i64 %row, i64 %col, double %val, ptr %count) #0 {
entry:
  %old = load i64, ptr %count, align 8
  %inc = add i64 %old, 1
  store i64 %inc, ptr %count, align 8
  ret void
}

define internal double @seed_load(i64 %off, i64 %col) #1 {
entry:
  %idx = lshr exact i64 %off, 3
  %cmp = icmp eq i64 %idx, %col
  %res = uitofp i1 %cmp to double
  ret double %res
}

define internal void @nostore(double %val, i64 %off, i64 %col) #1 {
entry:
  ret void
}

define internal double @zero_load(i64 %off, i64 %col, ptr %count) #1 {
entry:
  ret double 0.000000e+00
}

define internal void @acc_store(double %val, i64 %off, i64 %col, ptr %count) #1 {
entry:
  %nz = fcmp une double %val, 0.000000e+00
  br i1 %nz, label %do, label %end

do:
  call void @accumulate(i64 %off, i64 %col, double %val, ptr %count)
  br label %end

end:
  ret void
}

define internal void @residual_independent(i64 %n, ptr %x, ptr %y, ptr %f) #1 {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %px = getelementptr inbounds double, ptr %x, i64 %i
  %vx = load double, ptr %px, align 8
  %pf = getelementptr inbounds double, ptr %f, i64 %i
  store double %vx, ptr %pf, align 8
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

define internal void @residual_cancelled(i64 %n, ptr %x, ptr %y, ptr %f) #1 {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %py = getelementptr inbounds double, ptr %y, i64 %i
  %vy = load double, ptr %py, align 8
  %zero = fsub fast double %vy, %vy
  %pf = getelementptr inbounds double, ptr %f, i64 %i
  store double %zero, ptr %pf, align 8
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; CHECK-LABEL: define void @jac_independent(
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret void
define void @jac_independent(i64 %n, ptr %x, ptr %y, ptr %count) {
entry:
  br label %outer

outer:
  %c = phi i64 [ 0, %entry ], [ %c.next, %outer.latch ]
  %dy = call ptr (...) @__enzyme_todense(ptr @seed_load, ptr @nostore, i64 %c)
  %df = call ptr (...) @__enzyme_todense(ptr @zero_load, ptr @acc_store, i64 %c, ptr %count)
  call void (...) @__enzyme_fwddiff(ptr @residual_independent, metadata !"enzyme_const", i64 %n, metadata !"enzyme_const", ptr %x, metadata !"enzyme_dup", ptr %y, ptr %dy, metadata !"enzyme_dupnoneed", ptr null, ptr %df)
  br label %outer.latch

outer.latch:
  %c.next = add nuw nsw i64 %c, 1
  %cdone = icmp eq i64 %c.next, %n
  br i1 %cdone, label %exit, label %outer

exit:
  ret void
}

; CHECK-LABEL: define void @jac_cancelled(
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret void
define void @jac_cancelled(i64 %n, ptr %x, ptr %y, ptr %count) {
entry:
  br label %outer

outer:
  %c = phi i64 [ 0, %entry ], [ %c.next, %outer.latch ]
  %dy = call ptr (...) @__enzyme_todense(ptr @seed_load, ptr @nostore, i64 %c)
  %df = call ptr (...) @__enzyme_todense(ptr @zero_load, ptr @acc_store, i64 %c, ptr %count)
  call void (...) @__enzyme_fwddiff(ptr @residual_cancelled, metadata !"enzyme_const", i64 %n, metadata !"enzyme_const", ptr %x, metadata !"enzyme_dup", ptr %y, ptr %dy, metadata !"enzyme_dupnoneed", ptr null, ptr %df)
  br label %outer.latch

outer.latch:
  %c.next = add nuw nsw i64 %c, 1
  %cdone = icmp eq i64 %c.next, %n
  br i1 %cdone, label %exit, label %outer

exit:
  ret void
}

define i32 @main() {
entry:
  %x = alloca [4 x double], align 8
  %y = alloca [4 x double], align 8
  %count = alloca i64, align 8
  store i64 0, ptr %count, align 8
  call void @jac_independent(i64 4, ptr %x, ptr %y, ptr %count)
  call void @jac_cancelled(i64 4, ptr %x, ptr %y, ptr %count)
  %final = load i64, ptr %count, align 8
  %res = trunc i64 %final to i32
  ret i32 %res
}

attributes #0 = { noinline "enzyme_sparse_accumulate" }
attributes #1 = { alwaysinline }
