; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -enzyme-auto-sparsity=1 -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -enzyme-auto-sparsity=1 -S | %lli - | FileCheck %s --check-prefix=EVAL; fi

declare ptr @__enzyme_todense(...)
declare i32 @printf(ptr, ...)

@.fmt_f = private unnamed_addr constant [12 x i8] c"f:%s=%.17g\0A\00"
@.fmt_i = private unnamed_addr constant [10 x i8] c"i:%s=%ld\0A\00"
@.s_sitofp = private unnamed_addr constant [7 x i8] c"sitofp\00"
@.s_sitofp_nz = private unnamed_addr constant [10 x i8] c"sitofp_nz\00"
@.s_fneg_pz = private unnamed_addr constant [8 x i8] c"fneg_pz\00"
@.s_fneg_nz = private unnamed_addr constant [8 x i8] c"fneg_nz\00"
@.s_fneg_u2 = private unnamed_addr constant [8 x i8] c"fneg_u2\00"
@.s_neg0 = private unnamed_addr constant [5 x i8] c"neg0\00"
@.s_neg1 = private unnamed_addr constant [5 x i8] c"neg1\00"
@.s_neg2 = private unnamed_addr constant [5 x i8] c"neg2\00"
@.s_zext5 = private unnamed_addr constant [6 x i8] c"zext5\00"
@.s_zext1 = private unnamed_addr constant [6 x i8] c"zext1\00"
@.s_uitofp_nz = private unnamed_addr constant [10 x i8] c"uitofp_nz\00"
@.s_uitofp_1 = private unnamed_addr constant [9 x i8] c"uitofp_1\00"
@.s_fmul = private unnamed_addr constant [5 x i8] c"fmul\00"
@.s_udiv = private unnamed_addr constant [5 x i8] c"udiv\00"
@.s_dag = private unnamed_addr constant [4 x i8] c"dag\00"
@.s_wrap = private unnamed_addr constant [5 x i8] c"wrap\00"
@.s_wrap_icmp = private unnamed_addr constant [10 x i8] c"wrap_icmp\00"
@.s_start1 = private unnamed_addr constant [7 x i8] c"start1\00"
@.s_self = private unnamed_addr constant [5 x i8] c"self\00"
@.s_self_f = private unnamed_addr constant [7 x i8] c"self_f\00"

@counter1 = internal global i64 0
@counter2 = internal global i64 0

define void @acc(i64 %off, double %v) #0 {
entry:
  ret void
}

define void @barrier() noinline {
entry:
  call void asm sideeffect "", ""()
  ret void
}

define void @sink_f(ptr %name, double %x) noinline {
entry:
  %r = call i32 (ptr, ...) @printf(ptr @.fmt_f, ptr %name, double %x)
  ret void
}

define void @sink_i(ptr %name, i64 %x) noinline {
entry:
  %r = call i32 (ptr, ...) @printf(ptr @.fmt_i, ptr %name, i64 %x)
  ret void
}

define void @sink_v2d(<2 x double> %x) noinline {
entry:
  ret void
}

define void @sink_v2i64(<2 x i64> %x) noinline {
entry:
  ret void
}

define void @sink_i1(i1 %x) noinline {
entry:
  ret void
}

define void @sink_d_quiet(double %x) noinline {
entry:
  call void asm sideeffect "", "x"(double %x)
  ret void
}

define i1 @tick(ptr %ctr, i64 %limit, i8 %idx) noinline {
entry:
  %v = load i64, ptr %ctr, align 8
  %n = add i64 %v, 1
  store i64 %n, ptr %ctr, align 8
  %done = icmp eq i64 %n, %limit
  ret i1 %done
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

; 1. PhiSIToFP: +2^63 must not flip to -2^63, -0.0 must preserve sign bit,
;    and vector <2 x double> must not crash cast<IntegerType>.
; CHECK-LABEL: define void @test_sitofp(
define void @test_sitofp(i64 %n, i1 %cond, i64 %x, <2 x i64> %vx) {
entry:
  br label %outer
outer:
  %c = phi i64 [ 0, %entry ], [ %cnext, %outer.latch ]
  %dx = call ptr (...) @__enzyme_todense(ptr @seed_load, ptr @nostore, i64 %c)
  %dy = call ptr (...) @__enzyme_todense(ptr @zero_load, ptr @acc_store)
  br i1 %cond, label %b1, label %b2
b1:
  call void @barrier()
  %vsf1 = sitofp <2 x i64> %vx to <2 x double>
  call void @sink_v2d(<2 x double> %vsf1)
  br label %merge
b2:
  call void @barrier()
  %sf = sitofp i64 %x to double
  %vsf2 = sitofp <2 x i64> %vx to <2 x double>
  br label %merge
merge:
  %p1 = phi double [ 9.223372036854775808e+18, %b1 ], [ %sf, %b2 ]
  %p2 = phi double [ -0.000000e+00, %b1 ], [ %sf, %b2 ]
  %pv = phi <2 x double> [ %vsf1, %b1 ], [ %vsf2, %b2 ]
  call void @sink_f(ptr @.s_sitofp, double %p1)
  %p2bits = bitcast double %p2 to i64
  call void @sink_i(ptr @.s_sitofp_nz, i64 %p2bits)
  call void @sink_v2d(<2 x double> %pv)
  br label %inner
inner:
  %i = phi i64 [ 0, %merge ], [ %inext, %inner ]
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

; 2. PhiFNeg: +0.0/-0.0 sign preservation, multi-user update, no self-loop hang.
; CHECK-LABEL: define void @test_phifneg(
define void @test_phifneg(i64 %n, i1 %cond, double %x, i64 %m) {
entry:
  br label %outer
outer:
  %c = phi i64 [ 0, %entry ], [ %cnext, %outer.latch ]
  %dx = call ptr (...) @__enzyme_todense(ptr @seed_load, ptr @nostore, i64 %c)
  %dy = call ptr (...) @__enzyme_todense(ptr @zero_load, ptr @acc_store)
  br i1 %cond, label %b1, label %b2
b1:
  call void @barrier()
  br label %merge
b2:
  call void @barrier()
  %nx = fneg double %x
  call void @sink_d_quiet(double %nx)
  br label %merge
merge:
  %p1 = phi double [ 0.000000e+00, %b1 ], [ %nx, %b2 ]
  %p2 = phi double [ -0.000000e+00, %b1 ], [ %nx, %b2 ]
  %b1bits = bitcast double %p1 to i64
  %b2bits = bitcast double %p2 to i64
  call void @sink_i(ptr @.s_fneg_pz, i64 %b1bits)
  call void @sink_i(ptr @.s_fneg_nz, i64 %b2bits)
  %b1bits2 = bitcast double %p1 to i64
  call void @sink_i(ptr @.s_fneg_u2, i64 %b1bits2)
  br label %sloop
sloop:
  %k = phi i64 [ 0, %merge ], [ %knext, %sloop ]
  %sp = phi double [ 1.000000e+00, %merge ], [ %sneg, %sloop ]
  %sneg = fneg double %sp
  %knext = add nuw nsw i64 %k, 1
  %kdone = icmp eq i64 %knext, %m
  br i1 %kdone, label %sexit, label %sloop
sexit:
  call void @sink_f(ptr @.s_self_f, double %sp)
  br label %inner
inner:
  %i = phi i64 [ 0, %sexit ], [ %inext, %inner ]
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

; 3. PHINeg: multi-user update (no use-iterator invalidation) and self-loop.
; CHECK-LABEL: define void @test_phineg(
define void @test_phineg(i64 %n, i1 %cond, i64 %b, i64 %m) {
entry:
  br label %outer
outer:
  %c = phi i64 [ 0, %entry ], [ %cnext, %outer.latch ]
  %dx = call ptr (...) @__enzyme_todense(ptr @seed_load, ptr @nostore, i64 %c)
  %dy = call ptr (...) @__enzyme_todense(ptr @zero_load, ptr @acc_store)
  br i1 %cond, label %b1, label %b2
b1:
  call void @barrier()
  br label %merge
b2:
  call void @barrier()
  %nb = sub i64 0, %b
  call void @sink_i(ptr @.s_neg0, i64 %nb)
  br label %merge
merge:
  %p = phi i64 [ 5, %b1 ], [ %nb, %b2 ]
  call void @sink_i(ptr @.s_neg1, i64 %p)
  call void @sink_i(ptr @.s_neg2, i64 %p)
  br label %sloop
sloop:
  %k = phi i64 [ 0, %merge ], [ %knext, %sloop ]
  %sp = phi i64 [ 1, %merge ], [ %sneg, %sloop ]
  %sneg = sub i64 0, %sp
  %knext = add nuw nsw i64 %k, 1
  %kdone = icmp eq i64 %knext, %m
  br i1 %kdone, label %sexit, label %sloop
sexit:
  call void @sink_i(ptr @.s_self, i64 %sp)
  br label %inner
inner:
  %i = phi i64 [ 0, %sexit ], [ %inext, %inner ]
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

; 4. PHIBinop: ZExt constants (5, 1) and vector <2 x i1>, UIToFP (-0.0, 1.0),
;    and FMul (-0.0).
; CHECK-LABEL: define void @test_phibinop(
define void @test_phibinop(i64 %n, i1 %cond, i1 %flag, double %a, double %b, <2 x i1> %vx, <2 x i1> %vy) {
entry:
  br label %outer
outer:
  %c = phi i64 [ 0, %entry ], [ %cnext, %outer.latch ]
  %dx = call ptr (...) @__enzyme_todense(ptr @seed_load, ptr @nostore, i64 %c)
  %dy = call ptr (...) @__enzyme_todense(ptr @zero_load, ptr @acc_store)
  br i1 %cond, label %b1, label %b2
b1:
  call void @barrier()
  %vz1 = zext <2 x i1> %vx to <2 x i64>
  call void @sink_v2i64(<2 x i64> %vz1)
  br label %merge
b2:
  call void @barrier()
  %z = zext i1 %flag to i64
  %u = uitofp i1 %flag to double
  %m = fmul double %a, %b
  %vz2 = zext <2 x i1> %vy to <2 x i64>
  br label %merge
merge:
  %pz5 = phi i64 [ 5, %b1 ], [ %z, %b2 ]
  %pz1 = phi i64 [ 1, %b1 ], [ %z, %b2 ]
  %pvz = phi <2 x i64> [ %vz1, %b1 ], [ %vz2, %b2 ]
  %punz = phi double [ -0.000000e+00, %b1 ], [ %u, %b2 ]
  %pu1 = phi double [ 1.000000e+00, %b1 ], [ %u, %b2 ]
  %pfm = phi double [ -0.000000e+00, %b1 ], [ %m, %b2 ]
  call void @sink_i(ptr @.s_zext5, i64 %pz5)
  call void @sink_i(ptr @.s_zext1, i64 %pz1)
  call void @sink_v2i64(<2 x i64> %pvz)
  %unzb = bitcast double %punz to i64
  call void @sink_i(ptr @.s_uitofp_nz, i64 %unzb)
  %u1i = fptosi double %pu1 to i64
  call void @sink_i(ptr @.s_uitofp_1, i64 %u1i)
  %fmb = bitcast double %pfm to i64
  call void @sink_i(ptr @.s_fmul, i64 %fmb)
  br label %inner
inner:
  %i = phi i64 [ 0, %merge ], [ %inext, %inner ]
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

; 5. TPhiSel: must not speculate udiv (no SIGFPE), and diamond DAG must be
;    hoisted in valid topological order.
; CHECK-LABEL: define void @test_tphisel(
define void @test_tphisel(i64 %n, i64 %a, i64 %b, i1 %cond, i64 %x) {
entry:
  br label %outer
outer:
  %c = phi i64 [ 0, %entry ], [ %cnext, %outer.latch ]
  %dx = call ptr (...) @__enzyme_todense(ptr @seed_load, ptr @nostore, i64 %c)
  %dy = call ptr (...) @__enzyme_todense(ptr @zero_load, ptr @acc_store)
  %iszero = icmp eq i64 %b, 0
  br i1 %iszero, label %merge1, label %dodiv
dodiv:
  %qval = udiv i64 %a, %b
  br label %merge1
merge1:
  %p1 = phi i64 [ 0, %outer ], [ %qval, %dodiv ]
  call void @sink_i(ptr @.s_udiv, i64 %p1)
  br i1 %cond, label %merge2, label %else2
else2:
  %e1 = add i64 %x, 1
  %e2 = mul i64 %e1, 3
  %e3 = xor i64 %e1, %e2
  call void @barrier()
  br label %merge2
merge2:
  %p2 = phi i64 [ 0, %merge1 ], [ %e3, %else2 ]
  call void @sink_i(ptr @.s_dag, i64 %p2)
  br label %inner
inner:
  %i = phi i64 [ 0, %merge2 ], [ %inext, %inner ]
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

; 6. InductVarSCEV (non-zero start + 2-latch loop) and PhiLoop0Sel/PHIBinop
;    wrapping i8 canonical IV must not crash or miscompile on iter 256.
; CHECK-LABEL: define void @test_loops(
define void @test_loops(i64 %n, i64 %m, i64 %limit, i1 %flag, i64 %a, i64 %b) {
entry:
  br label %outer
outer:
  %c = phi i64 [ 0, %entry ], [ %cnext, %outer.latch ]
  %dx = call ptr (...) @__enzyme_todense(ptr @seed_load, ptr @nostore, i64 %c)
  %dy = call ptr (...) @__enzyme_todense(ptr @zero_load, ptr @acc_store)
  br label %start1
start1:
  %k = phi i64 [ 1, %outer ], [ %knext, %start1 ]
  %knext = add nuw nsw i64 %k, 1
  %kdone = icmp eq i64 %knext, %m
  br i1 %kdone, label %start1.exit, label %start1
start1.exit:
  call void @sink_i(ptr @.s_start1, i64 %k)
  br label %wrap1
wrap1:
  %idx1 = phi i8 [ 0, %start1.exit ], [ %idx1.next, %wrap1 ]
  %val1 = phi i64 [ 999, %start1.exit ], [ 7, %wrap1 ]
  %idx1.next = add i8 %idx1, 1
  %w1done = call i1 @tick(ptr @counter1, i64 %limit, i8 %idx1)
  br i1 %w1done, label %wrap1.exit, label %wrap1
wrap1.exit:
  call void @sink_i(ptr @.s_wrap, i64 %val1)
  br label %wrap2
wrap2:
  %idx2 = phi i8 [ 0, %wrap1.exit ], [ %idx2.next, %wrap2 ]
  %val2 = phi i1 [ false, %wrap1.exit ], [ %cmp2, %wrap2 ]
  %cmp2 = icmp eq i64 %a, %b
  %idx2.next = add i8 %idx2, 1
  %w2done = call i1 @tick(ptr @counter2, i64 %limit, i8 %idx2)
  br i1 %w2done, label %wrap2.exit, label %wrap2
wrap2.exit:
  %val2z = zext i1 %val2 to i64
  call void @sink_i(ptr @.s_wrap_icmp, i64 %val2z)
  br label %mloop
mloop:
  %midx = phi i64 [ %midx.next, %latch1 ], [ %midx.next, %latch2 ], [ 0, %wrap2.exit ]
  %mp = phi i1 [ %mcmp1, %latch1 ], [ %mcmp2, %latch2 ], [ false, %wrap2.exit ]
  call void @sink_i1(i1 %mp)
  %midx.next = add nuw nsw i64 %midx, 1
  br i1 %flag, label %latch1, label %latch2
latch1:
  call void @barrier()
  %mcmp1 = icmp eq i64 %midx, %a
  %md1 = icmp eq i64 %midx.next, %m
  br i1 %md1, label %mexit, label %mloop
latch2:
  call void @barrier()
  %mcmp2 = icmp eq i64 %midx, %b
  %md2 = icmp eq i64 %midx.next, %m
  br i1 %md2, label %mexit, label %mloop
mexit:
  br label %inner
inner:
  %i = phi i64 [ 0, %mexit ], [ %inext, %inner ]
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
entry:
  call void @test_sitofp(i64 1, i1 true, i64 7, <2 x i64> <i64 1, i64 2>)
  call void @test_phifneg(i64 1, i1 true, double 3.500000e+00, i64 4)
  call void @test_phineg(i64 1, i1 true, i64 99, i64 4)
  call void @test_phibinop(i64 1, i1 true, i1 false, double 2.000000e+00, double 3.000000e+00, <2 x i1> <i1 true, i1 false>, <2 x i1> <i1 false, i1 true>)
  call void @test_tphisel(i64 1, i64 10, i64 0, i1 false, i64 5)
  call void @test_loops(i64 1, i64 5, i64 257, i1 true, i64 7, i64 7)
  ret i32 0
}

attributes #0 = { noinline "enzyme_sparse_accumulate" }
attributes #1 = { alwaysinline }

; EVAL: f:sitofp=9.2233720368547758e+18
; EVAL-NEXT: i:sitofp_nz=-9223372036854775808
; EVAL-NEXT: i:fneg_pz=0
; EVAL-NEXT: i:fneg_nz=-9223372036854775808
; EVAL-NEXT: i:fneg_u2=0
; EVAL-NEXT: f:self_f=-1
; EVAL-NEXT: i:neg1=5
; EVAL-NEXT: i:neg2=5
; EVAL-NEXT: i:self=-1
; EVAL-NEXT: i:zext5=5
; EVAL-NEXT: i:zext1=1
; EVAL-NEXT: i:uitofp_nz=-9223372036854775808
; EVAL-NEXT: i:uitofp_1=1
; EVAL-NEXT: i:fmul=-9223372036854775808
; EVAL-NEXT: i:udiv=0
; EVAL-NEXT: i:dag=20
; EVAL-NEXT: i:start1=4
; EVAL-NEXT: i:wrap=7
; EVAL-NEXT: i:wrap_icmp=1
