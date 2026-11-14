; RUN: if [ %llvmver -ge 16 ]; then split-file %s %t && \
; RUN:   not %opt < %t/unknown_btc.ll %OPnewLoadEnzyme -passes="enzyme" -enzyme-auto-sparsity=1 -disable-output 2>&1 | FileCheck %s --check-prefix=UNKNOWN && \
; RUN:   not %opt < %t/inv_only.ll %OPnewLoadEnzyme -passes="enzyme" -enzyme-auto-sparsity=1 -disable-output 2>&1 | FileCheck %s --check-prefix=NOSOL && \
; RUN:   not %opt < %t/negated_iv.ll %OPnewLoadEnzyme -passes="enzyme" -enzyme-auto-sparsity=1 -disable-output 2>&1 | FileCheck %s --check-prefix=NOSOL && \
; RUN:   not %opt < %t/wrapping_mul.ll %OPnewLoadEnzyme -passes="enzyme" -enzyme-auto-sparsity=1 -disable-output 2>&1 | FileCheck %s --check-prefix=NOSOL && \
; RUN:   not %opt < %t/wrapping_trunc.ll %OPnewLoadEnzyme -passes="enzyme" -enzyme-auto-sparsity=1 -disable-output 2>&1 | FileCheck %s --check-prefix=NOSOL; fi

; UNKNOWN: No sparsification: unknown loop trip count
; NOSOL: No sparsification: not sparse solvable(nosoltn)

;--- unknown_btc.ll
declare ptr @__enzyme_todense(...)
declare void @accumulate(i64, double) #0
declare i1 @should_stop(i64)

define internal void @nostore(double %v, i64 %off) #1 { ret void }
define internal double @zero_load(i64 %off) #1 { ret double 0.000000e+00 }
define internal void @acc_store(double %v, i64 %off) #1 {
entry:
  %nz = fcmp une double %v, 0.000000e+00
  br i1 %nz, label %do, label %end
do:
  call void @accumulate(i64 %off, double %v)
  br label %end
end:
  ret void
}
define internal double @load_eq2(i64 %off) #1 {
  %idx = lshr exact i64 %off, 3
  %c = icmp eq i64 %idx, 2
  %r = uitofp i1 %c to double
  ret double %r
}
define void @test_unknown_btc() {
entry:
  %dx = call ptr (...) @__enzyme_todense(ptr @load_eq2, ptr @nostore)
  %dy = call ptr (...) @__enzyme_todense(ptr @zero_load, ptr @acc_store)
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %inext, %loop ]
  %pa = getelementptr inbounds double, ptr %dx, i64 %i
  %v = load double, ptr %pa, align 8
  %q = getelementptr inbounds double, ptr %dy, i64 %i
  store double %v, ptr %q, align 8
  %inext = add nuw nsw i64 %i, 1
  %done = call i1 @should_stop(i64 %inext)
  br i1 %done, label %exit, label %loop
exit:
  ret void
}
attributes #0 = { noinline "enzyme_sparse_accumulate" }
attributes #1 = { alwaysinline }

;--- inv_only.ll
declare ptr @__enzyme_todense(...)
declare void @accumulate(i64, double) #0

define internal void @nostore(double %v, i64 %off, i64 %a) #1 { ret void }
define internal double @zero_load(i64 %off) #1 { ret double 0.000000e+00 }
define internal void @acc_store(double %v, i64 %off) #1 {
entry:
  %nz = fcmp une double %v, 0.000000e+00
  br i1 %nz, label %do, label %end
do:
  call void @accumulate(i64 %off, double %v)
  br label %end
end:
  ret void
}
define internal double @load_inv(i64 %off, i64 %flag) #1 {
  %c = icmp eq i64 %flag, 0
  %r = uitofp i1 %c to double
  ret double %r
}
define void @test_inv_only(i64 %flag) {
entry:
  %dx = call ptr (...) @__enzyme_todense(ptr @load_inv, ptr @nostore, i64 %flag)
  %dy = call ptr (...) @__enzyme_todense(ptr @zero_load, ptr @acc_store)
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %inext, %loop ]
  %pa = getelementptr inbounds double, ptr %dx, i64 %i
  %v = load double, ptr %pa, align 8
  %q = getelementptr inbounds double, ptr %dy, i64 %i
  store double %v, ptr %q, align 8
  %inext = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %inext, 10
  br i1 %done, label %exit, label %loop
exit:
  ret void
}
attributes #0 = { noinline "enzyme_sparse_accumulate" }
attributes #1 = { alwaysinline }

;--- negated_iv.ll
declare ptr @__enzyme_todense(...)
declare void @accumulate(i64, double) #0

define internal void @nostore(double %v, i64 %off) #1 { ret void }
define internal double @zero_load(i64 %off) #1 { ret double 0.000000e+00 }
define internal void @acc_store(double %v, i64 %off) #1 {
entry:
  %nz = fcmp une double %v, 0.000000e+00
  br i1 %nz, label %do, label %end
do:
  call void @accumulate(i64 %off, double %v)
  br label %end
end:
  ret void
}
define internal double @load_ne2(i64 %off) #1 {
  %idx = lshr exact i64 %off, 3
  %c = icmp ne i64 %idx, 2
  %r = uitofp i1 %c to double
  ret double %r
}
define void @test_negated_iv() {
entry:
  %dx = call ptr (...) @__enzyme_todense(ptr @load_ne2, ptr @nostore)
  %dy = call ptr (...) @__enzyme_todense(ptr @zero_load, ptr @acc_store)
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %inext, %loop ]
  %pa = getelementptr inbounds double, ptr %dx, i64 %i
  %v = load double, ptr %pa, align 8
  %q = getelementptr inbounds double, ptr %dy, i64 %i
  store double %v, ptr %q, align 8
  %inext = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %inext, 10
  br i1 %done, label %exit, label %loop
exit:
  ret void
}
attributes #0 = { noinline "enzyme_sparse_accumulate" }
attributes #1 = { alwaysinline }

;--- wrapping_mul.ll
declare ptr @__enzyme_todense(...)
declare void @accumulate(i64, double) #0

define internal void @nostore(double %v, i64 %off) #1 { ret void }
define internal double @zero_load(i64 %off) #1 { ret double 0.000000e+00 }
define internal void @acc_store(double %v, i64 %off) #1 {
entry:
  %nz = fcmp une double %v, 0.000000e+00
  br i1 %nz, label %do, label %end
do:
  call void @accumulate(i64 %off, double %v)
  br label %end
end:
  ret void
}
define internal double @load_3i(i64 %off) #1 {
  %idx = lshr exact i64 %off, 3
  %m = mul i64 %idx, 3
  %c = icmp eq i64 %m, 2
  %r = uitofp i1 %c to double
  ret double %r
}
define void @test_wrapping_mul() {
entry:
  %dx = call ptr (...) @__enzyme_todense(ptr @load_3i, ptr @nostore)
  %dy = call ptr (...) @__enzyme_todense(ptr @zero_load, ptr @acc_store)
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %inext, %loop ]
  %pa = getelementptr inbounds double, ptr %dx, i64 %i
  %v = load double, ptr %pa, align 8
  %q = getelementptr inbounds double, ptr %dy, i64 %i
  store double %v, ptr %q, align 8
  %inext = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %inext, 10
  br i1 %done, label %exit, label %loop
exit:
  ret void
}
attributes #0 = { noinline "enzyme_sparse_accumulate" }
attributes #1 = { alwaysinline }

;--- wrapping_trunc.ll
declare ptr @__enzyme_todense(...)
declare void @accumulate(i64, double) #0

define internal void @nostore(double %v, i64 %off) #1 { ret void }
define internal double @zero_load(i64 %off) #1 { ret double 0.000000e+00 }
define internal void @acc_store(double %v, i64 %off) #1 {
entry:
  %nz = fcmp une double %v, 0.000000e+00
  br i1 %nz, label %do, label %end
do:
  call void @accumulate(i64 %off, double %v)
  br label %end
end:
  ret void
}
define internal double @load_trunc_i8(i64 %off) #1 {
  %idx = lshr exact i64 %off, 3
  %t = trunc i64 %idx to i8
  %c = icmp eq i8 %t, 5
  %r = uitofp i1 %c to double
  ret double %r
}
define void @test_wrapping_trunc_iv() {
entry:
  %dx = call ptr (...) @__enzyme_todense(ptr @load_trunc_i8, ptr @nostore)
  %dy = call ptr (...) @__enzyme_todense(ptr @zero_load, ptr @acc_store)
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %inext, %loop ]
  %pa = getelementptr inbounds double, ptr %dx, i64 %i
  %v = load double, ptr %pa, align 8
  %q = getelementptr inbounds double, ptr %dy, i64 %i
  store double %v, ptr %q, align 8
  %inext = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %inext, 300
  br i1 %done, label %exit, label %loop
exit:
  ret void
}
attributes #0 = { noinline "enzyme_sparse_accumulate" }
attributes #1 = { alwaysinline }
