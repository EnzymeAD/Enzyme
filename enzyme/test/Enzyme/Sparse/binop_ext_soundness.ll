; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -enzyme-auto-sparsity=1 -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -enzyme-auto-sparsity=1 -S | %lli - | FileCheck %s --check-prefix=EVAL; fi

; Regression tests for BinopExtToExtBinop in fixSparse_inner:
; 1. Vector uitofp/sitofp must not crash cast<IntegerType>.
; 2. uitofp i64 + 1.0 must not convert to sitofp i64 (wrong sign for u >= 2^63).
; 3. uitofp i32 * uitofp i32 must not overflow signed i64 (mul nsw poison).
; 4. sitofp i32 -> float must not fold across 24-bit significand rounding.
; 5. fmul (sitofp i8 -1), (uitofp i1 false) must preserve -0.0 without nsz,
;    while still folding when nsz/fast is present.
; 6. Exact in-range ops (uitofp i16 * uitofp i16 -> double, uitofp i64 with
;    !range [0, 100) + 1.0 -> double) must still fold to integer arithmetic.

@.fmt = private unnamed_addr constant [73 x i8] c"vec=%g,%g u64=%g mul32=%g fsub=%g nsz0=%d nsz1=%d ok_mul=%g ok_range=%g\0A\00", align 1

declare ptr @__enzyme_todense(...)
declare i32 @printf(ptr, ...)

define void @acc(i64 %off, double %v) #0 {
entry:
  ret void
}
define void @sink_d(ptr %out, double %v) noinline {
entry:
  store double %v, ptr %out, align 8
  ret void
}
define void @sink_f(ptr %out, float %v) noinline {
entry:
  store float %v, ptr %out, align 4
  ret void
}
define void @sink_v2d(ptr %out, <2 x double> %v) noinline {
entry:
  store <2 x double> %v, ptr %out, align 16
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

define void @test_vec(i64 %n, <2 x i32> %a, <2 x i32> %b, ptr %out) {
entry:
  br label %outer
outer:
  %c = phi i64 [ 0, %entry ], [ %cnext, %outer.latch ]
  %dx = call ptr (...) @__enzyme_todense(ptr @seed_load, ptr @nostore, i64 %c)
  %dy = call ptr (...) @__enzyme_todense(ptr @zero_load, ptr @acc_store)
  br label %inner
inner:
  %i = phi i64 [ 0, %outer ], [ %inext, %inner ]
  %fa = uitofp <2 x i32> %a to <2 x double>
  %fb = uitofp <2 x i32> %b to <2 x double>
  %s = fadd <2 x double> %fa, %fb
  call void @sink_v2d(ptr %out, <2 x double> %s)
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

define void @test_scalar(
    i64 %n, i64 %u, i32 %a, i32 %b, i32 %fa_i, i32 %fb_i,
    i8 %neg1, i1 %z, i16 %s1, i16 %s2, ptr %rp,
    ptr %out_u64, ptr %out_mul32, ptr %out_fsub,
    ptr %out_m0, ptr %out_m1, ptr %out_okmul, ptr %out_okrange) {
entry:
  br label %outer
outer:
  %c = phi i64 [ 0, %entry ], [ %cnext, %outer.latch ]
  %dx = call ptr (...) @__enzyme_todense(ptr @seed_load, ptr @nostore, i64 %c)
  %dy = call ptr (...) @__enzyme_todense(ptr @zero_load, ptr @acc_store)
  br label %inner
inner:
  %i = phi i64 [ 0, %outer ], [ %inext, %inner ]
  ; Must NOT fold: uitofp i64 may exceed INT64_MAX / 2^53.
  %uf = uitofp i64 %u to double
  %uadd = fadd double %uf, 1.000000e+00
  call void @sink_d(ptr %out_u64, double %uadd)

  ; Must NOT fold: (2^32 - 1)^2 overflows signed i64.
  %da = uitofp i32 %a to double
  %db = uitofp i32 %b to double
  %dmul = fmul double %da, %db
  call void @sink_d(ptr %out_mul32, double %dmul)

  ; Must NOT fold: i32 exceeds float's 24-bit exact integer range.
  %ffa = sitofp i32 %fa_i to float
  %ffb = sitofp i32 %fb_i to float
  %fsub = fsub float %ffa, %ffb
  call void @sink_f(ptr %out_fsub, float %fsub)

  ; Without nsz: (-1.0) * (+0.0) must preserve -0.0.
  %dn1 = sitofp i8 %neg1 to double
  %dz = uitofp i1 %z to double
  %m_strict = fmul double %dn1, %dz
  call void @sink_d(ptr %out_m0, double %m_strict)

  ; With nsz: (-1.0) * (+0.0) may fold to sitofp (mul nsw).
  %m_nsz = fmul nsz double %dn1, %dz
  call void @sink_d(ptr %out_m1, double %m_nsz)

  ; Exact in-range positive case: uint16 * uint16 <= 2^32 < 2^53.
  %ds1 = uitofp i16 %s1 to double
  %ds2 = uitofp i16 %s2 to double
  %okmul = fmul double %ds1, %ds2
  call void @sink_d(ptr %out_okmul, double %okmul)

  ; Exact in-range positive case: uitofp i64 with !range [0, 100) + 1.0.
  %rv = load i64, ptr %rp, align 8, !range !0
  %dr = uitofp i64 %rv to double
  %okr = fadd double %dr, 1.000000e+00
  call void @sink_d(ptr %out_okrange, double %okr)

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
  %ov = alloca <2 x double>, align 16
  %o_u64 = alloca double, align 8
  %o_mul32 = alloca double, align 8
  %o_fsub = alloca float, align 4
  %o_m0 = alloca double, align 8
  %o_m1 = alloca double, align 8
  %o_okmul = alloca double, align 8
  %o_okrange = alloca double, align 8
  %rp = alloca i64, align 8
  store i64 41, ptr %rp, align 8

  call void @test_vec(i64 1, <2 x i32> <i32 2, i32 3>, <2 x i32> <i32 4, i32 5>, ptr %ov)
  %vv = load <2 x double>, ptr %ov, align 16
  %v0 = extractelement <2 x double> %vv, i32 0
  %v1 = extractelement <2 x double> %vv, i32 1

  call void @test_scalar(
      i64 1, i64 -9223372036854775808, i32 -1, i32 -1, i32 16777217, i32 16777216,
      i8 -1, i1 false, i16 1000, i16 2000, ptr %rp,
      ptr %o_u64, ptr %o_mul32, ptr %o_fsub,
      ptr %o_m0, ptr %o_m1, ptr %o_okmul, ptr %o_okrange)

  %r_u64 = load double, ptr %o_u64, align 8
  %r_mul32 = load double, ptr %o_mul32, align 8
  %r_fsub_f = load float, ptr %o_fsub, align 4
  %r_fsub = fpext float %r_fsub_f to double

  %r_m0 = load double, ptr %o_m0, align 8
  %r_m0_b = bitcast double %r_m0 to i64
  %r_m0_s64 = lshr i64 %r_m0_b, 63
  %r_m0_s = trunc i64 %r_m0_s64 to i32

  %r_m1 = load double, ptr %o_m1, align 8
  %r_m1_b = bitcast double %r_m1 to i64
  %r_m1_s64 = lshr i64 %r_m1_b, 63
  %r_m1_s = trunc i64 %r_m1_s64 to i32

  %r_okmul = load double, ptr %o_okmul, align 8
  %r_okrange = load double, ptr %o_okrange, align 8

  call i32 (ptr, ...) @printf(ptr @.fmt, double %v0, double %v1, double %r_u64, double %r_mul32, double %r_fsub, i32 %r_m0_s, i32 %r_m1_s, double %r_okmul, double %r_okrange)
  ret i32 0
}

!0 = !{i64 0, i64 100}

attributes #0 = { noinline "enzyme_sparse_accumulate" }
attributes #1 = { alwaysinline }

; EVAL: vec=6,8 u64=9.22337e+18 mul32=1.84467e+19 fsub=0 nsz0=1 nsz1=0 ok_mul=2e+06 ok_range=42

; CHECK-LABEL: define internal void @test_vec.inner(
; CHECK: %fa = uitofp <2 x i32> %a to <2 x double>
; CHECK: %fb = uitofp <2 x i32> %b to <2 x double>
; CHECK: %s = fadd <2 x double> %fa, %fb

; CHECK-LABEL: define internal void @test_scalar.inner(
; CHECK: %uf = uitofp i64 %u to double
; CHECK-NEXT: %uadd = fadd double %uf, 1.000000e+00
; CHECK: %da = uitofp i32 %a to double
; CHECK-NEXT: %db = uitofp i32 %b to double
; CHECK-NEXT: %dmul = fmul double %da, %db
; CHECK: %ffa = sitofp i32 %fa_i to float
; CHECK-NEXT: %ffb = sitofp i32 %fb_i to float
; CHECK-NEXT: %fsub = fsub float %ffa, %ffb
; CHECK: %m_strict = fmul double %dn1, %dz
; CHECK: %[[MNSZ_I:.+]] = mul nsw i64
; CHECK-NEXT: %[[MNSZ_F:.+]] = sitofp i64 %[[MNSZ_I]] to double
; CHECK-NEXT: call void @sink_d(ptr %out_m1, double %[[MNSZ_F]])
; CHECK: %[[OKMUL_I:.+]] = mul nsw i64
; CHECK-NEXT: %[[OKMUL_F:.+]] = sitofp i64 %[[OKMUL_I]] to double
; CHECK-NEXT: call void @sink_d(ptr %out_okmul, double %[[OKMUL_F]])
; CHECK: %[[OKR_I:.+]] = add nsw i64 %rv, 1
; CHECK-NEXT: %[[OKR_F:.+]] = sitofp i64 %[[OKR_I]] to double
; CHECK-NEXT: call void @sink_d(ptr %out_okrange, double %[[OKR_F]])
