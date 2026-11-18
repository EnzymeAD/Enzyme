; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -enzyme-auto-sparsity=1 -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -enzyme-auto-sparsity=1 -S | %lli - | FileCheck %s --check-prefix=EVAL; fi

@.fmt = private unnamed_addr constant [72 x i8] c"iadd=%lld imul=%d cmp=%d inv=%d vec=%d,%d msub=%d fcmp=%d,%d,%d mcc=%d\0A\00", align 1

declare ptr @__enzyme_todense(...)
declare i32 @printf(ptr, ...)

define void @acc(i64 %off, double %v) #0 {
entry:
  ret void
}
define void @sink_i64(ptr %out, i64 %v) noinline {
entry:
  store i64 %v, ptr %out, align 8
  ret void
}
define void @sink_i8(ptr %out, i8 %v) noinline {
entry:
  store i8 %v, ptr %out, align 1
  ret void
}
define void @sink_i1(ptr %out, i1 %v) noinline {
entry:
  %z = zext i1 %v to i8
  store i8 %z, ptr %out, align 1
  ret void
}
define void @sink_dbl(ptr %out, double %v) noinline {
entry:
  store double %v, ptr %out, align 8
  ret void
}
define void @sink_v2i32(ptr %out, <2 x i32> %v) noinline {
entry:
  store <2 x i32> %v, ptr %out, align 8
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
define internal double @zero_load(i64 %off) #1 {
  ret double 0.000000e+00
}
define internal void @acc_store(double %v, i64 %off) #1 {
entry:
  %nz = fcmp une double %v, 0.000000e+00
  br i1 %nz, label %do, label %end
do:
  call void @acc(i64 %off, double %v)
  br label %end
end:
  ret void
}

; 1. IAddDivConst: (x + 8) >> 3 must shift x as well, not become x + 1.
define void @test_iadddivconst(i64 %n, i64 %x, ptr %out) {
entry:
  br label %outer
outer:
  %c = phi i64 [ 0, %entry ], [ %cnext, %outer.latch ]
  %dx = call ptr (...) @__enzyme_todense(ptr @seed_load, ptr @nostore, i64 %c)
  %dy = call ptr (...) @__enzyme_todense(ptr @zero_load, ptr @acc_store)
  br label %inner
inner:
  %i = phi i64 [ 0, %outer ], [ %inext, %inner ]
  %a = add nuw i64 %x, 8
  %d = lshr exact i64 %a, 3
  call void @sink_i64(ptr %out, i64 %d)
  %p = getelementptr inbounds double, ptr %dx, i64 %i
  %v = load double, ptr %p, align 8
  %q = getelementptr inbounds double, ptr %dy, i64 %i
  store double %v, ptr %q, align 8
  %inext = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %inext, %n
  br i1 %done, label %outer.latch, label %inner
outer.latch:
  %cnext = add nuw nsw i64 %c, 1
  %cdone = icmp eq i64 %cnext, %n
  br i1 %cdone, label %exit, label %outer
exit:
  ret void
}

; 2. IMulDivConst: lshr exact (mul %x, 12), 2 must not cancel without nuw.
define void @test_imuldivconst(i64 %n, i8 %x, ptr %out) {
entry:
  br label %outer
outer:
  %c = phi i64 [ 0, %entry ], [ %cnext, %outer.latch ]
  %dx = call ptr (...) @__enzyme_todense(ptr @seed_load, ptr @nostore, i64 %c)
  %dy = call ptr (...) @__enzyme_todense(ptr @zero_load, ptr @acc_store)
  br label %inner
inner:
  %i = phi i64 [ 0, %outer ], [ %inext, %inner ]
  %m = mul i8 %x, 12
  %d = lshr exact i8 %m, 2
  call void @sink_i8(ptr %out, i8 %d)
  %p = getelementptr inbounds double, ptr %dx, i64 %i
  %v = load double, ptr %p, align 8
  %q = getelementptr inbounds double, ptr %dy, i64 %i
  store double %v, ptr %q, align 8
  %inext = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %inext, %n
  br i1 %done, label %outer.latch, label %inner
outer.latch:
  %cnext = add nuw nsw i64 %c, 1
  %cdone = icmp eq i64 %cnext, %n
  br i1 %cdone, label %exit, label %outer
exit:
  ret void
}

; 3. CmpAddAdd: (x + x) == 0 must not fold to x == 0 without nuw/nsw (x = -128).
define void @test_cmpaddadd(i64 %n, i1 %cond, i8 %x, ptr %scratch, ptr %out) {
entry:
  br label %outer
outer:
  %c = phi i64 [ 0, %entry ], [ %cnext, %outer.latch ]
  %dx = call ptr (...) @__enzyme_todense(ptr @seed_load, ptr @nostore, i64 %c)
  %dy = call ptr (...) @__enzyme_todense(ptr @zero_load, ptr @acc_store)
  br label %inner
inner:
  %i = phi i64 [ 0, %outer ], [ %inext, %inner ]
  %s1 = select i1 %cond, i8 %x, i8 0
  %s2 = select i1 %cond, i8 %x, i8 1
  %s3 = select i1 %cond, i8 0, i8 2
  call void @sink_i8(ptr %scratch, i8 %s1)
  call void @sink_i8(ptr %scratch, i8 %s2)
  call void @sink_i8(ptr %scratch, i8 %s3)
  %sum = add i8 %s1, %s2
  call void @sink_i8(ptr %scratch, i8 %sum)
  %cmp = icmp eq i8 %sum, %s3
  call void @sink_i1(ptr %out, i1 %cmp)
  %p = getelementptr inbounds double, ptr %dx, i64 %i
  %v = load double, ptr %p, align 8
  %q = getelementptr inbounds double, ptr %dy, i64 %i
  store double %v, ptr %q, align 8
  %inext = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %inext, %n
  br i1 %done, label %outer.latch, label %inner
outer.latch:
  %cnext = add nuw nsw i64 %c, 1
  %cdone = icmp eq i64 %cnext, %n
  br i1 %cdone, label %exit, label %outer
exit:
  ret void
}

; 4. InvDistributive: drop unsound nsw/nuw, keep nuw only when all 3 have nuw,
;    and handle integer vectors without emitting fmul.
define void @test_invdist(i64 %n, i8 %a, i8 %b, i8 %k, <2 x i32> %va, <2 x i32> %vb, <2 x i32> %vk, ptr %scratch, ptr %out_i8, ptr %out_v, ptr %scratch_v) {
entry:
  br label %outer
outer:
  %c = phi i64 [ 0, %entry ], [ %cnext, %outer.latch ]
  %dx = call ptr (...) @__enzyme_todense(ptr @seed_load, ptr @nostore, i64 %c)
  %dy = call ptr (...) @__enzyme_todense(ptr @zero_load, ptr @acc_store)
  br label %inner
inner:
  %i = phi i64 [ 0, %outer ], [ %inext, %inner ]
  ; nsw must NOT be placed on factored mul (a=64, b=64, k=-1)
  %m1 = mul nsw i8 %a, %k
  %m2 = mul nsw i8 %b, %k
  call void @sink_i8(ptr %scratch, i8 %m1)
  call void @sink_i8(ptr %scratch, i8 %m2)
  %s = add nsw i8 %m1, %m2
  call void @sink_i8(ptr %out_i8, i8 %s)
  ; nuw must NOT be placed when mul2/add lack nuw
  %u1 = mul nuw i8 %a, %k
  %u2 = mul i8 %b, %k
  call void @sink_i8(ptr %scratch, i8 %u1)
  call void @sink_i8(ptr %scratch, i8 %u2)
  %us = add i8 %u1, %u2
  call void @sink_i8(ptr %scratch, i8 %us)
  ; nuw IS preserved when mul1, mul2, and add all have nuw
  %p1 = mul nuw i8 %a, %k
  %p2 = mul nuw i8 %b, %k
  call void @sink_i8(ptr %scratch, i8 %p1)
  call void @sink_i8(ptr %scratch, i8 %p2)
  %ps = add nuw i8 %p1, %p2
  call void @sink_i8(ptr %scratch, i8 %ps)
  ; integer vector must emit mul, not fmul
  %vm1 = mul <2 x i32> %va, %vk
  %vm2 = mul <2 x i32> %vb, %vk
  call void @sink_v2i32(ptr %scratch_v, <2 x i32> %vm1)
  call void @sink_v2i32(ptr %scratch_v, <2 x i32> %vm2)
  %vs = add <2 x i32> %vm1, %vm2
  call void @sink_v2i32(ptr %out_v, <2 x i32> %vs)
  %p = getelementptr inbounds double, ptr %dx, i64 %i
  %v = load double, ptr %p, align 8
  %q = getelementptr inbounds double, ptr %dy, i64 %i
  store double %v, ptr %q, align 8
  %inext = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %inext, %n
  br i1 %done, label %outer.latch, label %inner
outer.latch:
  %cnext = add nuw nsw i64 %c, 1
  %cdone = icmp eq i64 %cnext, %n
  br i1 %cdone, label %exit, label %outer
exit:
  ret void
}

; 5 & 6. MulSubNegConst & SubMulConstant: drop unsound nuw/nsw and avoid INT_MIN loop.
define void @test_mulsub_submul(i64 %n, i8 %x, i8 %y, ptr %scratch, ptr %out) {
entry:
  br label %outer
outer:
  %c = phi i64 [ 0, %entry ], [ %cnext, %outer.latch ]
  %dx = call ptr (...) @__enzyme_todense(ptr @seed_load, ptr @nostore, i64 %c)
  %dy = call ptr (...) @__enzyme_todense(ptr @zero_load, ptr @acc_store)
  br label %inner
inner:
  %i = phi i64 [ 0, %outer ], [ %inext, %inner ]
  %sub = sub nuw i8 %x, %y
  call void @sink_i8(ptr %scratch, i8 %sub)
  %m = mul i8 %sub, -3
  call void @sink_i8(ptr %out, i8 %m)
  ; INT_MIN constant must not infinitely loop
  %mmin = mul i8 %sub, -128
  call void @sink_i8(ptr %scratch, i8 %mmin)
  ; SubMulConstant must not emit mul nuw nsw
  %mc = mul nuw nsw i8 %x, 3
  call void @sink_i8(ptr %scratch, i8 %mc)
  %neg = sub i8 0, %mc
  call void @sink_i8(ptr %scratch, i8 %neg)
  %p = getelementptr inbounds double, ptr %dx, i64 %i
  %v = load double, ptr %p, align 8
  %q = getelementptr inbounds double, ptr %dy, i64 %i
  store double %v, ptr %q, align 8
  %inext = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %inext, %n
  br i1 %done, label %outer.latch, label %inner
outer.latch:
  %cnext = add nuw nsw i64 %c, 1
  %cdone = icmp eq i64 %cnext, %n
  br i1 %cdone, label %exit, label %outer
exit:
  ret void
}

; 7. FCmpSelectConst: preserve predicate direction when constant is LHS.
define void @test_fcmpselectconst(i64 %n, i1 %cond, i1 %cond2, double %b, ptr %scratch_d, ptr %out) {
entry:
  br label %outer
outer:
  %c = phi i64 [ 0, %entry ], [ %cnext, %outer.latch ]
  %dx = call ptr (...) @__enzyme_todense(ptr @seed_load, ptr @nostore, i64 %c)
  %dy = call ptr (...) @__enzyme_todense(ptr @zero_load, ptr @acc_store)
  br label %inner
inner:
  %i = phi i64 [ 0, %outer ], [ %inext, %inner ]
  %sel = select i1 %cond, double 2.000000e+00, double %b
  call void @sink_dbl(ptr %scratch_d, double %sel)
  %s1 = select i1 %cond2, double 1.000000e+00, double 3.000000e+00
  %s2 = select i1 %cond2, double %sel, double 5.000000e+00
  call void @sink_dbl(ptr %scratch_d, double %s1)
  call void @sink_dbl(ptr %scratch_d, double %s2)
  %cmp = fcmp olt double %s1, %s2
  call void @sink_i1(ptr %out, i1 %cmp)
  %p = getelementptr inbounds double, ptr %dx, i64 %i
  %v = load double, ptr %p, align 8
  %q = getelementptr inbounds double, ptr %dy, i64 %i
  store double %v, ptr %q, align 8
  %inext = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %inext, %n
  br i1 %done, label %outer.latch, label %inner
outer.latch:
  %cnext = add nuw nsw i64 %c, 1
  %cdone = icmp eq i64 %cnext, %n
  br i1 %cdone, label %exit, label %outer
exit:
  ret void
}

; 8. MulConstConst: integer mul (mul x, 3), 5 -> mul x, 15.
define void @test_mulconstconst(i64 %n, i8 %x, ptr %scratch_8, ptr %out_8) {
entry:
  br label %outer
outer:
  %c = phi i64 [ 0, %entry ], [ %cnext, %outer.latch ]
  %dx = call ptr (...) @__enzyme_todense(ptr @seed_load, ptr @nostore, i64 %c)
  %dy = call ptr (...) @__enzyme_todense(ptr @zero_load, ptr @acc_store)
  br label %inner
inner:
  %i = phi i64 [ 0, %outer ], [ %inext, %inner ]
  %im1 = mul i8 %x, 3
  call void @sink_i8(ptr %scratch_8, i8 %im1)
  %im2 = mul i8 %im1, 5
  call void @sink_i8(ptr %out_8, i8 %im2)
  %p = getelementptr inbounds double, ptr %dx, i64 %i
  %v = load double, ptr %p, align 8
  %q = getelementptr inbounds double, ptr %dy, i64 %i
  store double %v, ptr %q, align 8
  %inext = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %inext, %n
  br i1 %done, label %outer.latch, label %inner
outer.latch:
  %cnext = add nuw nsw i64 %c, 1
  %cdone = icmp eq i64 %cnext, %n
  br i1 %cdone, label %exit, label %outer
exit:
  ret void
}

define i32 @main() {
entry:
  %out64 = alloca i64, align 8
  %out8 = alloca i8, align 1
  %scr8 = alloca i8, align 1
  %out_cmp = alloca i8, align 1
  %out_inv = alloca i8, align 1
  %out_v = alloca <2 x i32>, align 8
  %scr_v = alloca <2 x i32>, align 8
  %out_msub = alloca i8, align 1
  %scr_d = alloca double, align 8
  %out_f1 = alloca i8, align 1
  %out_f2 = alloca i8, align 1
  %out_f3 = alloca i8, align 1
  %out_mcc8 = alloca i8, align 1

  ; 1. (16 + 8) >> 3 = 3 (buggy code produced 16 + 1 = 17)
  call void @test_iadddivconst(i64 1, i64 16, ptr %out64)
  %r_iadd = load i64, ptr %out64, align 8

  ; 2. ((64 * 12) mod 256) >> 2 = 0 (buggy code produced 64 * 3 = -64)
  call void @test_imuldivconst(i64 1, i8 64, ptr %out8)
  %r_imul8 = load i8, ptr %out8, align 1
  %r_imul = sext i8 %r_imul8 to i32

  ; 3. (-128 + -128) == 0 is 1 (buggy code checked -128 == 0 -> 0)
  call void @test_cmpaddadd(i64 1, i1 true, i8 -128, ptr %scr8, ptr %out_cmp)
  %r_cmp8 = load i8, ptr %out_cmp, align 1
  %r_cmp = zext i8 %r_cmp8 to i32

  ; 4. InvDistributive: 64*(-1) + 64*(-1) = -128; vector (1+3)*5=20, (2+4)*7=42
  call void @test_invdist(i64 1, i8 64, i8 64, i8 -1, <2 x i32> <i32 1, i32 2>, <2 x i32> <i32 3, i32 4>, <2 x i32> <i32 5, i32 7>, ptr %scr8, ptr %out_inv, ptr %out_v, ptr %scr_v)
  %r_inv8 = load i8, ptr %out_inv, align 1
  %r_inv = sext i8 %r_inv8 to i32
  %rv = load <2 x i32>, ptr %out_v, align 8
  %rv0 = extractelement <2 x i32> %rv, i32 0
  %rv1 = extractelement <2 x i32> %rv, i32 1

  ; 5 & 6. (5 - 2) * (-3) = -9
  call void @test_mulsub_submul(i64 1, i8 5, i8 2, ptr %scr8, ptr %out_msub)
  %r_msub8 = load i8, ptr %out_msub, align 1
  %r_msub = sext i8 %r_msub8 to i32

  ; 7. cond2=true: (cond=true -> 1.0<2.0 = 1), (cond=false,b=0.5 -> 1.0<0.5 = 0),
  ;                (cond=false,b=1.5 -> 1.0<1.5 = 1)
  call void @test_fcmpselectconst(i64 1, i1 true, i1 true, double 0.500000e+00, ptr %scr_d, ptr %out_f1)
  call void @test_fcmpselectconst(i64 1, i1 false, i1 true, double 0.500000e+00, ptr %scr_d, ptr %out_f2)
  call void @test_fcmpselectconst(i64 1, i1 false, i1 true, double 1.500000e+00, ptr %scr_d, ptr %out_f3)
  %rf1_8 = load i8, ptr %out_f1, align 1
  %rf2_8 = load i8, ptr %out_f2, align 1
  %rf3_8 = load i8, ptr %out_f3, align 1
  %rf1 = zext i8 %rf1_8 to i32
  %rf2 = zext i8 %rf2_8 to i32
  %rf3 = zext i8 %rf3_8 to i32

  ; 8. MulConstConst: 4 * 3 * 5 = 60
  call void @test_mulconstconst(i64 1, i8 4, ptr %scr8, ptr %out_mcc8)
  %r_mcc8 = load i8, ptr %out_mcc8, align 1
  %r_mcc = sext i8 %r_mcc8 to i32

  call i32 (ptr, ...) @printf(ptr @.fmt, i64 %r_iadd, i32 %r_imul, i32 %r_cmp, i32 %r_inv, i32 %rv0, i32 %rv1, i32 %r_msub, i32 %rf1, i32 %rf2, i32 %rf3, i32 %r_mcc)
  ret i32 0
}

attributes #0 = { noinline "enzyme_sparse_accumulate" }
attributes #1 = { alwaysinline }

; EVAL: iadd=3 imul=0 cmp=1 inv=-128 vec=20,42 msub=-9 fcmp=1,0,1 mcc=60

; CHECK-LABEL: define internal void @test_iadddivconst.inner(
; CHECK: %[[SH:.+]] = lshr exact i64 %x, 3
; CHECK-NEXT: %madd.d = add nuw i64 %[[SH]], 1

; CHECK-LABEL: define internal void @test_imuldivconst.inner(
; CHECK: %m = mul i8 %x, 12
; CHECK-NEXT: %d = lshr exact i8 %m, 2

; CHECK-LABEL: define internal void @test_cmpaddadd.inner(
; CHECK: %tval.sum = add i8 %x, %x
; CHECK: %tval.cmp = icmp eq i8 %tval.sum, 0

; CHECK-LABEL: define internal void @test_invdist.inner(
; CHECK: %[[S1:.+]] = add i8 %a, %b
; CHECK-NEXT: %[[M1:.+]] = mul i8 %[[S1]], %k
; CHECK-NEXT: call void @sink_i8(ptr %out_i8, i8 %[[M1]])
; CHECK: %[[S2:.+]] = add i8 %a, %b
; CHECK-NEXT: %[[M2:.+]] = mul i8 %[[S2]], %k
; CHECK: %[[S3:.+]] = add i8 %a, %b
; CHECK-NEXT: %[[M3:.+]] = mul nuw i8 %[[S3]], %k
; CHECK: %[[VS:.+]] = add <2 x i32> %va, %vb
; CHECK-NEXT: %[[VM:.+]] = mul <2 x i32> %[[VS]], %vk

; CHECK-LABEL: define internal void @test_mulsub_submul.inner(
; CHECK: %[[SUB:.+]] = sub i8 %y, %x
; CHECK-NEXT: %[[MUL3:.+]] = mul i8 %[[SUB]], 3
; CHECK: %neg.mc = mul i8 %x, -3

; CHECK-LABEL: define internal void @test_fcmpselectconst.inner(
; CHECK: %[[GT:.+]] = fcmp ogt double %b, 1.000000e+00
; CHECK: or i1 %cond, %[[GT]]

; CHECK-LABEL: define internal void @test_mulconstconst.inner(
; CHECK: mul i8 %x, 15
