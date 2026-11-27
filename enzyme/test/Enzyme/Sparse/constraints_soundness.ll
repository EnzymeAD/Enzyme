; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-auto-sparsity=1 -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-auto-sparsity=1 -S | %lli -; fi

; Regression tests for soundness of sparse constraint solving, bounds checking,
; and multi-width enzyme.sparse.inbounds lowering:
;   1. test_sign_union: (%a == 0 && %i == %a + 2) || (%i == %c) must not flip
;      +2 to -2 in Constraints::andB.
;   2. test_affine_udiv: 2*%i == %target and 3*%i + 2 == %col (nuw) must emit
;      exact remainder checks rather than accepting inexact UDiv quotients.
;   3. test_cast_invariance: cannotDependOnLoopIV must look through SCEVCastExpr
;      (zext/trunc) so (a - b != 0 && 2*(a - b) == 0 && i == 1) is not dropped.
;   4. test_i8_tripcount: unsigned icmp ule bounds check + unsigned range check
;      in make_compare must handle i8 loops with trip count >= 128.
;   5. test_trunc_flag: loop-invariant i64 %flag == 0 inside an i32 loop must
;      expand at i64 (not truncate to i32).
;   6. test_zext_iv: zext i32 %i to i64 == %target64 must check upper bits of
;      %target64 instead of truncating it to i32.
;   7. test_outer_addrec: non-nuw outer-loop AddRec (2*%c == %target) combined
;      with inner %i == 3 must remain valid as an unscoped Compare.

@g_cnt = internal global i64 0, align 8
@g_sum = internal global i64 0, align 8

declare ptr @__enzyme_todense(...)

define void @accumulate(i64 %off, double %v) #0 {
  %c = load i64, ptr @g_cnt, align 8
  %c1 = add i64 %c, 1
  store i64 %c1, ptr @g_cnt, align 8
  %s = load i64, ptr @g_sum, align 8
  %s1 = add i64 %s, %off
  store i64 %s1, ptr @g_sum, align 8
  ret void
}

define internal void @nostore1(double %v, i64 %off, i64 %a) #1 {
  ret void
}

define internal void @nostore2(double %v, i64 %off, i64 %a, i64 %b) #1 {
  ret void
}

define internal void @nostore_i8(double %v, i64 %off, i8 %a) #1 {
  ret void
}

define internal double @zero_load(i64 %off) #1 {
  ret double 0.000000e+00
}

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

; 1) (%a == 0 && %i == %a + 2) || (%i == %c)
define internal double @load_sign_union(i64 %off, i64 %a, i64 %c) #1 {
  %idx = lshr exact i64 %off, 3
  %a0 = icmp eq i64 %a, 0
  %ap2 = add i64 %a, 2
  %eq1 = icmp eq i64 %idx, %ap2
  %b1 = and i1 %a0, %eq1
  %eq2 = icmp eq i64 %idx, %c
  %u = or i1 %b1, %eq2
  %r = uitofp i1 %u to double
  ret double %r
}

define void @test_sign_union(i64 %n, i64 %a, i64 %c) {
entry:
  %dx = call ptr (...) @__enzyme_todense(ptr @load_sign_union, ptr @nostore2, i64 %a, i64 %c)
  %dy = call ptr (...) @__enzyme_todense(ptr @zero_load, ptr @acc_store)
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inext, %loop ]
  %pa = getelementptr inbounds double, ptr %dx, i64 %i
  %v = load double, ptr %pa, align 8
  %q = getelementptr inbounds double, ptr %dy, i64 %i
  store double %v, ptr %q, align 8
  %inext = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %inext, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; 2a) 2 * %i == %target (nuw)
define internal double @load_mul2(i64 %off, i64 %target) #1 {
  %idx = lshr exact i64 %off, 3
  %m = mul nuw i64 %idx, 2
  %c = icmp eq i64 %m, %target
  %r = uitofp i1 %c to double
  ret double %r
}

define void @test_mul2(i64 %n, i64 %target) {
entry:
  %dx = call ptr (...) @__enzyme_todense(ptr @load_mul2, ptr @nostore1, i64 %target)
  %dy = call ptr (...) @__enzyme_todense(ptr @zero_load, ptr @acc_store)
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inext, %loop ]
  %pa = getelementptr inbounds double, ptr %dx, i64 %i
  %v = load double, ptr %pa, align 8
  %q = getelementptr inbounds double, ptr %dy, i64 %i
  store double %v, ptr %q, align 8
  %inext = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %inext, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; 2b) 3 * %i + 2 == %col (nuw)
define internal double @load_3d(i64 %off, i64 %col) #1 {
  %idx = lshr exact i64 %off, 3
  %i3 = mul nuw i64 %idx, 3
  %i3p2 = add nuw i64 %i3, 2
  %c = icmp eq i64 %i3p2, %col
  %r = uitofp i1 %c to double
  ret double %r
}

define void @test_3d(i64 %col) {
entry:
  %dx = call ptr (...) @__enzyme_todense(ptr @load_3d, ptr @nostore1, i64 %col)
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

; 3) (%a - %b != 0) && (2 * (%a - %b) == 0) && (%i == 1)
define internal double @load_cast_inv(i64 %off, i64 %a, i64 %b) #1 {
  %idx = lshr exact i64 %off, 3
  %d = sub i64 %a, %b
  %d2 = mul i64 %d, 2
  %c1 = icmp ne i64 %d, 0
  %c2 = icmp eq i64 %d2, 0
  %c3 = icmp eq i64 %idx, 1
  %a1 = and i1 %c1, %c2
  %a2 = and i1 %a1, %c3
  %r = uitofp i1 %a2 to double
  ret double %r
}

define void @test_cast_invariance(i64 %a, i64 %b) {
entry:
  %dx = call ptr (...) @__enzyme_todense(ptr @load_cast_inv, ptr @nostore2, i64 %a, i64 %b)
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

; 4) i8 loop with trip count 200 (>= 128)
define internal double @load_i8(i64 %off, i8 %target) #1 {
  %idx64 = lshr exact i64 %off, 3
  %idx = trunc i64 %idx64 to i8
  %c = icmp eq i8 %idx, %target
  %r = uitofp i1 %c to double
  ret double %r
}

define void @test_i8_tripcount(i8 %target) {
entry:
  %dx = call ptr (...) @__enzyme_todense(ptr @load_i8, ptr @nostore_i8, i8 %target)
  %dy = call ptr (...) @__enzyme_todense(ptr @zero_load, ptr @acc_store)
  br label %loop

loop:
  %i = phi i8 [ 0, %entry ], [ %inext, %loop ]
  %iz = zext i8 %i to i64
  %pa = getelementptr inbounds double, ptr %dx, i64 %iz
  %v = load double, ptr %pa, align 8
  %q = getelementptr inbounds double, ptr %dy, i64 %iz
  store double %v, ptr %q, align 8
  %inext = add i8 %i, 1
  %done = icmp eq i8 %inext, 200
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; 5) i64 %flag == 0 && i32 %i == 2
define internal double @load_trunc_flag(i64 %off, i64 %flag) #1 {
  %idx = lshr exact i64 %off, 3
  %idx32 = trunc i64 %idx to i32
  %c1 = icmp eq i64 %flag, 0
  %c2 = icmp eq i32 %idx32, 2
  %c = and i1 %c1, %c2
  %r = uitofp i1 %c to double
  ret double %r
}

define void @test_trunc_flag(i64 %flag) {
entry:
  %dx = call ptr (...) @__enzyme_todense(ptr @load_trunc_flag, ptr @nostore1, i64 %flag)
  %dy = call ptr (...) @__enzyme_todense(ptr @zero_load, ptr @acc_store)
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %inext, %loop ]
  %iz = zext i32 %i to i64
  %pa = getelementptr inbounds double, ptr %dx, i64 %iz
  %v = load double, ptr %pa, align 8
  %q = getelementptr inbounds double, ptr %dy, i64 %iz
  store double %v, ptr %q, align 8
  %inext = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %inext, 10
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; 6) zext i32 %i to i64 == %target64
define internal double @load_eq64(i64 %off, i64 %target64) #1 {
  %idx = lshr exact i64 %off, 3
  %c = icmp eq i64 %idx, %target64
  %r = uitofp i1 %c to double
  ret double %r
}

define void @test_zext_iv(i64 %target64) {
entry:
  %dx = call ptr (...) @__enzyme_todense(ptr @load_eq64, ptr @nostore1, i64 %target64)
  %dy = call ptr (...) @__enzyme_todense(ptr @zero_load, ptr @acc_store)
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %inext, %loop ]
  %iz = zext i32 %i to i64
  %pa = getelementptr inbounds double, ptr %dx, i64 %iz
  %v = load double, ptr %pa, align 8
  %q = getelementptr inbounds double, ptr %dy, i64 %iz
  store double %v, ptr %q, align 8
  %inext = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %inext, 10
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; 7) Outer-loop non-nuw 2 * %c == %target && inner %i == 3
define internal double @load_outer(i64 %off, i64 %c, i64 %target) #1 {
  %idx = lshr exact i64 %off, 3
  %m = mul i64 %c, 2
  %c1 = icmp eq i64 %m, %target
  %c2 = icmp eq i64 %idx, 3
  %both = and i1 %c1, %c2
  %r = uitofp i1 %both to double
  ret double %r
}

define void @test_outer_addrec(i64 %target) {
entry:
  br label %outer

outer:
  %c = phi i64 [ 0, %entry ], [ %cnext, %outer.latch ]
  %dx = call ptr (...) @__enzyme_todense(ptr @load_outer, ptr @nostore2, i64 %c, i64 %target)
  %dy = call ptr (...) @__enzyme_todense(ptr @zero_load, ptr @acc_store)
  br label %inner

inner:
  %i = phi i64 [ 0, %outer ], [ %inext, %inner ]
  %pa = getelementptr inbounds double, ptr %dx, i64 %i
  %v = load double, ptr %pa, align 8
  %q = getelementptr inbounds double, ptr %dy, i64 %i
  store double %v, ptr %q, align 8
  %inext = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %inext, 10
  br i1 %done, label %outer.latch, label %inner

outer.latch:
  %cnext = add nuw nsw i64 %c, 1
  %cdone = icmp eq i64 %cnext, 10
  br i1 %cdone, label %exit, label %outer

exit:
  ret void
}

define internal i32 @check_reset(i64 %exp_cnt, i64 %exp_sum, i32 %err) {
  %c = load i64, ptr @g_cnt, align 8
  %s = load i64, ptr @g_sum, align 8
  store i64 0, ptr @g_cnt, align 8
  store i64 0, ptr @g_sum, align 8
  %ok1 = icmp eq i64 %c, %exp_cnt
  %ok2 = icmp eq i64 %s, %exp_sum
  %ok = and i1 %ok1, %ok2
  %ret = select i1 %ok, i32 0, i32 %err
  ret i32 %ret
}

define i32 @main() {
  ; 1) test_sign_union(10, 0, 0): hits i=0 (off=0) and i=2 (off=16) -> cnt=2, sum=16
  call void @test_sign_union(i64 10, i64 0, i64 0)
  %r1 = call i32 @check_reset(i64 2, i64 16, i32 1)
  %e1 = icmp ne i32 %r1, 0
  br i1 %e1, label %fail1, label %t2
fail1:
  ret i32 %r1

t2:
  ; 2a) test_mul2: target=3 -> 0 hits; target=4 -> i=2 (off=16) -> cnt=1, sum=16
  call void @test_mul2(i64 10, i64 3)
  call void @test_mul2(i64 10, i64 4)
  %r2 = call i32 @check_reset(i64 1, i64 16, i32 2)
  %e2 = icmp ne i32 %r2, 0
  br i1 %e2, label %fail2, label %t3
fail2:
  ret i32 %r2

t3:
  ; 2b) test_3d: col=0..5 -> col=2 (i=0, off=0) and col=5 (i=1, off=8) -> cnt=2, sum=8
  call void @test_3d(i64 0)
  call void @test_3d(i64 1)
  call void @test_3d(i64 2)
  call void @test_3d(i64 3)
  call void @test_3d(i64 4)
  call void @test_3d(i64 5)
  %r3 = call i32 @check_reset(i64 2, i64 8, i32 3)
  %e3 = icmp ne i32 %r3, 0
  br i1 %e3, label %fail3, label %t4
fail3:
  ret i32 %r3

t4:
  ; 3) test_cast_invariance(5, 3): 2*(5-3)=4 != 0 -> cnt=0, sum=0
  call void @test_cast_invariance(i64 5, i64 3)
  %r4 = call i32 @check_reset(i64 0, i64 0, i32 4)
  %e4 = icmp ne i32 %r4, 0
  br i1 %e4, label %fail4, label %t5
fail4:
  ret i32 %r4

t5:
  ; 4) test_i8_tripcount: 10 (off=80) and 150 (off=1200) -> cnt=2, sum=1280
  call void @test_i8_tripcount(i8 10)
  call void @test_i8_tripcount(i8 150)
  call void @test_i8_tripcount(i8 210)
  %r5 = call i32 @check_reset(i64 2, i64 1280, i32 5)
  %e5 = icmp ne i32 %r5, 0
  br i1 %e5, label %fail5, label %t6
fail5:
  ret i32 %r5

t6:
  ; 5) test_trunc_flag: 4294967296 -> 0 hits; 0 -> i=2 (off=16) -> cnt=1, sum=16
  call void @test_trunc_flag(i64 4294967296)
  call void @test_trunc_flag(i64 0)
  %r6 = call i32 @check_reset(i64 1, i64 16, i32 6)
  %e6 = icmp ne i32 %r6, 0
  br i1 %e6, label %fail6, label %t7
fail6:
  ret i32 %r6

t7:
  ; 6) test_zext_iv: 4294967298 -> 0 hits; 2 -> i=2 (off=16) -> cnt=1, sum=16
  call void @test_zext_iv(i64 4294967298)
  call void @test_zext_iv(i64 2)
  %r7 = call i32 @check_reset(i64 1, i64 16, i32 7)
  %e7 = icmp ne i32 %r7, 0
  br i1 %e7, label %fail7, label %t8
fail7:
  ret i32 %r7

t8:
  ; 7) test_outer_addrec: target=3 -> 0 hits; target=6 -> c=3, i=3 (off=24) -> cnt=1, sum=24
  call void @test_outer_addrec(i64 3)
  call void @test_outer_addrec(i64 6)
  %r8 = call i32 @check_reset(i64 1, i64 24, i32 8)
  ret i32 %r8
}

attributes #0 = { noinline "enzyme_sparse_accumulate" }
attributes #1 = { alwaysinline }

; CHECK: define void @test_sign_union(i64 %n, i64 %a, i64 %c)
; CHECK:   call void @test_sign_union.loop(
; CHECK:   call void @test_sign_union.loop(
; CHECK: define void @test_mul2(i64 %n, i64 %target)
; CHECK:   %[[Q:.+]] = lshr i64 %target, 1
; CHECK:   %[[M:.+]] = shl nuw i64 %[[Q]], 1
; CHECK:   %[[R:.+]] = sub i64 %[[M]], %target
; CHECK:   icmp eq i64 %[[R]], 0
; CHECK:   call void @test_mul2.loop(
; CHECK: define void @test_3d(i64 %col)
; CHECK:   udiv i64 {{.+}}, 3
; CHECK:   call void @test_3d.loop(
; CHECK: define void @test_cast_invariance(i64 %a, i64 %b)
; CHECK:   call void @test_cast_invariance.loop(
; CHECK: define void @test_i8_tripcount(i8 %target)
; CHECK:   call void @test_i8_tripcount.loop(i8 %target,
; CHECK: define void @test_trunc_flag(i64 %flag)
; CHECK-NOT: trunc i64 %flag to i32
; CHECK:   icmp eq i64 %flag, 0
; CHECK:   call void @test_trunc_flag.loop(i32 2,
; CHECK: define void @test_zext_iv(i64 %target64)
; CHECK:   %[[TR:.+]] = trunc i64 %target64 to i32
; CHECK:   %[[ZX:.+]] = zext i32 %[[TR]] to i64
; CHECK:   %[[DF:.+]] = sub i64 %[[ZX]], %target64
; CHECK:   icmp eq i64 %[[DF]], 0
; CHECK:   call void @test_zext_iv.loop(i32 %[[TR]],
; CHECK: define void @test_outer_addrec(i64 %target)
; CHECK:   call void @test_outer_addrec.inner(
; CHECK: define internal void @test_i8_tripcount.loop(
; CHECK:   icmp ule i8 %loop.idx, -57
; CHECK-NOT: call void @enzyme.sparse.inbounds
