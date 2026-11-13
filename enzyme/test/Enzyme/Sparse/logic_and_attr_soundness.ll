; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -enzyme-auto-sparsity=1 -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -enzyme-auto-sparsity=1 -S | %lli | FileCheck %s --check-prefix=EXEC; fi

declare ptr @__enzyme_todense(...)
declare i32 @printf(ptr, ...)

@.fmt_f = private unnamed_addr constant [12 x i8] c"f:%s=%.17g\0A\00"
@.fmt_i = private unnamed_addr constant [10 x i8] c"i:%s=%ld\0A\00"
@.s_u1 = private unnamed_addr constant [3 x i8] c"u1\00"
@.s_u0 = private unnamed_addr constant [3 x i8] c"u0\00"
@.s_m1 = private unnamed_addr constant [3 x i8] c"m1\00"
@.s_m0 = private unnamed_addr constant [3 x i8] c"m0\00"
@.s_imul = private unnamed_addr constant [5 x i8] c"imul\00"
@.s_and32 = private unnamed_addr constant [6 x i8] c"and32\00"
@.s_or32 = private unnamed_addr constant [5 x i8] c"or32\00"
@.s_rsum1 = private unnamed_addr constant [6 x i8] c"rsum1\00"
@.s_rsum0 = private unnamed_addr constant [6 x i8] c"rsum0\00"
@.s_noinl = private unnamed_addr constant [6 x i8] c"noinl\00"

define void @acc(i64 %off, double %v) #0 {
  ret void
}

define void @sink_f(ptr %name, double %x) noinline {
  %r = call i32 (ptr, ...) @printf(ptr @.fmt_f, ptr %name, double %x)
  ret void
}

define void @sink_i(ptr %name, i64 %x) noinline {
  %r = call i32 (ptr, ...) @printf(ptr @.fmt_i, ptr %name, i64 %x)
  ret void
}

define void @keep_f(double %x) noinline {
  ret void
}

define void @keep_i1(i1 %x) noinline {
  ret void
}

define void @keep_i8(i8 %x) noinline {
  ret void
}

define void @keep_v2i1(<2 x i1> %x) noinline {
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
  ret double 0.0
}

define internal void @acc_store(double %v, i64 %off) #1 {
entry:
  %nz = fcmp une double %v, 0.0
  br i1 %nz, label %do, label %end
do:
  call void @acc(i64 %off, double %v)
  br label %end
end:
  ret void
}

; 1a. SumSelect must NOT merge opposite-polarity conditions:
;     uitofp(cond) + select(cond, 0.0, y) != select(!cond, y + 1.0, 0.0).
define void @test_sumselect_opposite(i64 %n, i1 %cond, double %y, ptr %tag) {
entry:
  br label %outer
outer:
  %c = phi i64 [ 0, %entry ], [ %cnext, %outer.latch ]
  %dx = call ptr (...) @__enzyme_todense(ptr @seed_load, ptr @nostore, i64 %c)
  %dy = call ptr (...) @__enzyme_todense(ptr @zero_load, ptr @acc_store)
  br label %inner
inner:
  %i = phi i64 [ 0, %outer ], [ %inext, %inner ]
  %s1 = uitofp i1 %cond to double
  %s2 = select i1 %cond, double 0.0, double %y
  call void @keep_f(double %s1)
  call void @keep_f(double %s2)
  %sum = fadd fast double %s1, %s2
  call void @sink_f(ptr %tag, double %sum)
  %pa = getelementptr inbounds double, ptr %dx, i64 %i
  %v = load double, ptr %pa, align 8
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

; 1b. SumSelect SHOULD merge matching-polarity negated conditions:
;     uitofp(cond) + select(!cond, 0.0, y) == select(cond, y + 1.0, 0.0).
define void @test_sumselect_matching(i64 %n, i1 %cond, double %y, ptr %tag) {
entry:
  br label %outer
outer:
  %c = phi i64 [ 0, %entry ], [ %cnext, %outer.latch ]
  %dx = call ptr (...) @__enzyme_todense(ptr @seed_load, ptr @nostore, i64 %c)
  %dy = call ptr (...) @__enzyme_todense(ptr @zero_load, ptr @acc_store)
  br label %inner
inner:
  %i = phi i64 [ 0, %outer ], [ %inext, %inner ]
  %ncond = xor i1 %cond, true
  call void @keep_i1(i1 %ncond)
  %s1 = uitofp i1 %cond to double
  %s2 = select i1 %ncond, double 0.0, double %y
  call void @keep_f(double %s1)
  call void @keep_f(double %s2)
  %sum = fadd fast double %s1, %s2
  call void @sink_f(ptr %tag, double %sum)
  %pa = getelementptr inbounds double, ptr %dx, i64 %i
  %v = load double, ptr %pa, align 8
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

; 2. CmpIMulSplit must NOT split wrapping integer mul: (16 * 16) == 0 in i8.
define void @test_cmpimul_wrap(i64 %n, i8 %a, i8 %b) {
entry:
  br label %outer
outer:
  %c = phi i64 [ 0, %entry ], [ %cnext, %outer.latch ]
  %dx = call ptr (...) @__enzyme_todense(ptr @seed_load, ptr @nostore, i64 %c)
  %dy = call ptr (...) @__enzyme_todense(ptr @zero_load, ptr @acc_store)
  br label %inner
inner:
  %i = phi i64 [ 0, %outer ], [ %inext, %inner ]
  %m = mul i8 %a, %b
  %eq = icmp eq i8 %m, 0
  %z = zext i1 %eq to i64
  call void @sink_i(ptr @.s_imul, i64 %z)
  %pa = getelementptr inbounds double, ptr %dx, i64 %i
  %v = load double, ptr %pa, align 8
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

; 3. AndReplace/OrReplace on i32 and SelectReplace on <2 x i1> must not create
;    ill-typed IR with i1 constants.
define void @test_non_i1_logic(i64 %n, i32 %x, i32 %y, <2 x i1> %vc, <2 x i1> %vx, <2 x i1> %vy) {
entry:
  br label %outer
outer:
  %c = phi i64 [ 0, %entry ], [ %cnext, %outer.latch ]
  %dx = call ptr (...) @__enzyme_todense(ptr @seed_load, ptr @nostore, i64 %c)
  %dy = call ptr (...) @__enzyme_todense(ptr @zero_load, ptr @acc_store)
  br label %inner
inner:
  %i = phi i64 [ 0, %outer ], [ %inext, %inner ]
  %a1 = add i32 %x, 1
  %and32 = and i32 %a1, %x
  %zand = zext i32 %and32 to i64
  call void @sink_i(ptr @.s_and32, i64 %zand)
  %a2 = add i32 %x, %y
  %or32 = or i32 %a2, %x
  %zor = zext i32 %or32 to i64
  call void @sink_i(ptr @.s_or32, i64 %zor)
  %vxor = xor <2 x i1> %vc, %vx
  call void @keep_v2i1(<2 x i1> %vxor)
  %vsel = select <2 x i1> %vc, <2 x i1> %vxor, <2 x i1> %vy
  call void @keep_v2i1(<2 x i1> %vsel)
  %pa = getelementptr inbounds double, ptr %dx, i64 %i
  %v = load double, ptr %pa, align 8
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

; 4. replace() must return the rewritten sum/product call inside SelectReplace.
define void @test_replace_sum(i64 %n, i1 %cond, double %x, double %y, ptr %tag) {
entry:
  br label %outer
outer:
  %c = phi i64 [ 0, %entry ], [ %cnext, %outer.latch ]
  %dx = call ptr (...) @__enzyme_todense(ptr @seed_load, ptr @nostore, i64 %c)
  %dy = call ptr (...) @__enzyme_todense(ptr @zero_load, ptr @acc_store)
  br label %inner
inner:
  %i = phi i64 [ 0, %outer ], [ %inext, %inner ]
  %u = uitofp i1 %cond to double
  %a = fadd fast double %x, %u
  call void @keep_f(double %a)
  %s = select i1 %cond, double %a, double %y
  call void @sink_f(ptr %tag, double %s)
  %pa = getelementptr inbounds double, ptr %dx, i64 %i
  %v = load double, ptr %pa, align 8
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

; 5. AndReplace must drop nuw/nsw in replace() so substituting cond=true does
;    not introduce poison when cond is false at runtime.
define void @test_and_nuw_poison(i64 %n, i1 %cond, i8 %y) {
entry:
  br label %outer
outer:
  %c = phi i64 [ 0, %entry ], [ %cnext, %outer.latch ]
  %dx = call ptr (...) @__enzyme_todense(ptr @seed_load, ptr @nostore, i64 %c)
  %dy = call ptr (...) @__enzyme_todense(ptr @zero_load, ptr @acc_store)
  br label %inner
inner:
  %i = phi i64 [ 0, %outer ], [ %inext, %inner ]
  %z = zext i1 %cond to i8
  %a = add nuw i8 %z, %y
  call void @keep_i8(i8 %a)
  %c2 = icmp eq i8 %a, 0
  call void @keep_i1(i1 %c2)
  %r = and i1 %cond, %c2
  call void @keep_i1(i1 %r)
  %pa = getelementptr inbounds double, ptr %dx, i64 %i
  %v = load double, ptr %pa, align 8
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

; 6. LowerSparsification must strip noinline/optnone before adding alwaysinline.
define void @test_noinline(i64 %n) noinline {
entry:
  br label %outer
outer:
  %c = phi i64 [ 0, %entry ], [ %cnext, %outer.latch ]
  %dx = call ptr (...) @__enzyme_todense(ptr @seed_load, ptr @nostore, i64 %c)
  %dy = call ptr (...) @__enzyme_todense(ptr @zero_load, ptr @acc_store)
  br label %inner
inner:
  %i = phi i64 [ 0, %outer ], [ %inext, %inner ]
  call void @sink_i(ptr @.s_noinl, i64 %i)
  %pa = getelementptr inbounds double, ptr %dx, i64 %i
  %v = load double, ptr %pa, align 8
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
  call void @test_sumselect_opposite(i64 1, i1 true, double 20.0, ptr @.s_u1)
  call void @test_sumselect_opposite(i64 1, i1 false, double 20.0, ptr @.s_u0)
  call void @test_sumselect_matching(i64 1, i1 true, double 20.0, ptr @.s_m1)
  call void @test_sumselect_matching(i64 1, i1 false, double 20.0, ptr @.s_m0)
  call void @test_cmpimul_wrap(i64 1, i8 16, i8 16)
  call void @test_non_i1_logic(i64 1, i32 6, i32 1, <2 x i1> <i1 true, i1 false>, <2 x i1> <i1 true, i1 true>, <2 x i1> <i1 false, i1 true>)
  call void @test_replace_sum(i64 1, i1 true, double 5.0, double 9.0, ptr @.s_rsum1)
  call void @test_replace_sum(i64 1, i1 false, double 5.0, double 9.0, ptr @.s_rsum0)
  call void @test_and_nuw_poison(i64 1, i1 false, i8 255)
  call void @test_noinline(i64 1)
  ret i32 0
}

attributes #0 = { noinline "enzyme_sparse_accumulate" }
attributes #1 = { alwaysinline }

; CHECK-LABEL: define internal void @test_sumselect_opposite.inner
; CHECK-NOT: %sumpnot =
; CHECK: fadd fast double

; CHECK-LABEL: define internal void @test_sumselect_matching.inner
; CHECK: fadd fast double {{.*}}1.000000e+00
; CHECK: select i1 %cond, double {{.*}}, double 0.000000e+00

; CHECK-LABEL: define internal void @test_cmpimul_wrap.inner
; CHECK: %m = mul i8 %a, %b
; CHECK: %eq = icmp eq i8 %m, 0

; CHECK-LABEL: define internal void @test_replace_sum.inner
; CHECK: [[SUM:%[0-9a-zA-Z_.]+]] = fadd fast double 1.000000e+00, %x
; CHECK: %postsel.s = select i1 %cond, double [[SUM]], double %y

; CHECK-LABEL: define internal void @test_and_nuw_poison.inner
; CHECK: %sel.a = add i8 1, %y
; CHECK-NOT: add nuw i8 1, %y
; CHECK: %postand.r = and i1 %sel.c2, %cond

; CHECK-LABEL: define internal void @test_noinline.inner

; EXEC: f:u1=1
; EXEC-NEXT: f:u0=20
; EXEC-NEXT: f:m1=21
; EXEC-NEXT: f:m0=0
; EXEC-NEXT: i:imul=1
; EXEC-NEXT: i:and32=6
; EXEC-NEXT: i:or32=7
; EXEC-NEXT: f:rsum1=6
; EXEC-NEXT: f:rsum0=9
; EXEC-NEXT: i:noinl=0
