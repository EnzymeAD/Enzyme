; RUN: if [ %llvmver -ge 16 ]; then (%opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-auto-sparsity=1 -S 2>&1 || true) | FileCheck %s; fi

; The inner loop also exits through the error path of a bounds check. Its body
; cannot be outlined and called per solution, so it is not sparsified.

declare ptr @__enzyme_todense(...)

declare void @accumulate(i64, i64, double, ptr) #0

declare void @throw_bounds() noreturn

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
  %i = phi i64 [ 0, %outer ], [ %i.next, %body ]
  %i.next = add nuw nsw i64 %i, 1
  %i2 = add nuw nsw i64 %i, 2
  %inb = icmp ult i64 %i2, %n
  br i1 %inb, label %body, label %oob

body:
  %pa = getelementptr inbounds double, ptr %dx, i64 %i2
  %a = load double, ptr %pa, align 8
  %pb = getelementptr inbounds double, ptr %dx, i64 %i.next
  %b = load double, ptr %pb, align 8
  %b2 = fmul fast double %b, 2.000000e+00
  %d = fsub fast double %a, %b2
  %q = getelementptr inbounds double, ptr %dy, i64 %i.next
  store double %d, ptr %q, align 8
  %done = icmp eq i64 %i2, %n
  br i1 %done, label %outer.latch, label %inner

oob:
  call void @throw_bounds()
  unreachable

outer.latch:
  %c.next = add nuw nsw i64 %c, 1
  %cdone = icmp eq i64 %c.next, %n
  br i1 %cdone, label %exit, label %outer

exit:
  ret void
}

attributes #0 = { noinline "enzyme_sparse_accumulate" }
attributes #1 = { alwaysinline }

; CHECK: Loop has more than one exit block (e.g. the error path of a bounds check)
