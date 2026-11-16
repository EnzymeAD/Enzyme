; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,default<O1>,verify" -enzyme-preopt=false -S | %lli; fi

declare [2 x double] @__enzyme_batch.v2f64(...)
declare [4 x double] @__enzyme_batch.v4f64(...)
declare [2 x ptr] @__enzyme_batch.v2ptr(...)
declare [2 x i32] @__enzyme_batch.v2i32(...)
declare double @__enzyme_batch.f64(...)
declare void @__enzyme_batch.void(...)

@G_SCALAR = constant double 10.0, align 8
@G_ARR = constant [2 x double] [double 10.0, double 20.0], align 8
@LOG_BUF = global [4 x double] zeroinitializer, align 8
@LOG_IDX = global i32 0, align 4

; 1. Alloca-derived GEP must stay vectorized so lanes do not alias lane 0's stack slot.
define double @use_stack_array(double %x) {
entry:
  %arr = alloca [2 x double], align 8
  %p1 = getelementptr inbounds [2 x double], ptr %arr, i64 0, i64 1
  store double %x, ptr %p1, align 8
  %v = load double, ptr %p1, align 8
  ret double %v
}

define [2 x double] @test_alloca_gep(double %x0, double %x1) {
entry:
  %r = call [2 x double] (...) @__enzyme_batch.v2f64(ptr @use_stack_array, metadata !"enzyme_width", i64 2, metadata !"enzyme_vector", double %x0, double %x1)
  ret [2 x double] %r
}

; 2. Alloca-derived select and PHI must stay vectorized.
define double @use_stack_select(double %x, i1 %c) {
entry:
  %a = alloca double, align 8
  %b = alloca double, align 8
  store double %x, ptr %a, align 8
  %x1 = fadd double %x, 1.0
  store double %x1, ptr %b, align 8
  %p = select i1 %c, ptr %a, ptr %b
  %v = load double, ptr %p, align 8
  ret double %v
}

define [2 x double] @test_alloca_select(double %x0, double %x1, i1 %c) {
entry:
  %r = call [2 x double] (...) @__enzyme_batch.v2f64(ptr @use_stack_select, metadata !"enzyme_width", i64 2, metadata !"enzyme_vector", double %x0, double %x1, metadata !"enzyme_scalar", i1 %c)
  ret [2 x double] %r
}

define double @use_stack_phi(double %x, i1 %c) {
entry:
  %a = alloca double, align 8
  %b = alloca double, align 8
  store double %x, ptr %a, align 8
  %x2 = fmul double %x, 2.0
  store double %x2, ptr %b, align 8
  br i1 %c, label %l1, label %l2
l1:
  br label %join
l2:
  br label %join
join:
  %p = phi ptr [ %a, %l1 ], [ %b, %l2 ]
  %v = load double, ptr %p, align 8
  ret double %v
}

define [2 x double] @test_alloca_phi(double %x0, double %x1, i1 %c) {
entry:
  %r = call [2 x double] (...) @__enzyme_batch.v2f64(ptr @use_stack_phi, metadata !"enzyme_width", i64 2, metadata !"enzyme_vector", double %x0, double %x1, metadata !"enzyme_scalar", i1 %c)
  ret [2 x double] %r
}

; 3. Void instructions (calls/stores) must execute in ascending lane order 0, 1, 2, 3.
define void @record_val(double %v) noinline {
entry:
  %idx = load i32, ptr @LOG_IDX, align 4
  %idx64 = sext i32 %idx to i64
  %slot = getelementptr inbounds [4 x double], ptr @LOG_BUF, i64 0, i64 %idx64
  store double %v, ptr %slot, align 8
  %next = add i32 %idx, 1
  store i32 %next, ptr @LOG_IDX, align 4
  ret void
}

define void @log_val(ptr %fn, double %x) {
entry:
  call void %fn(double %x)
  ret void
}

define void @test_void_order(double %a, double %b, double %c, double %d) {
entry:
  call void (...) @__enzyme_batch.void(ptr @log_val, metadata !"enzyme_width", i64 4, metadata !"enzyme_scalar", ptr @record_val, metadata !"enzyme_vector", double %a, double %b, double %c, double %d)
  ret void
}

; 4. Direct global loads, constant aggregates, and ConstantExpr GEPs must not crash.
define double @read_global_and_const_agg(double %x) {
entry:
  %g0 = load double, ptr @G_SCALAR, align 8
  %g1 = load double, ptr getelementptr inbounds ([2 x double], ptr @G_ARR, i64 0, i64 1), align 8
  %s = insertvalue { double, double } { double 1.0, double 2.0 }, double %x, 0
  %a = extractvalue { double, double } %s, 0
  %b = extractvalue { double, double } %s, 1
  %sum0 = fadd double %a, %b
  %sum1 = fadd double %sum0, %g0
  %sum2 = fadd double %sum1, %g1
  ret double %sum2
}

define [2 x double] @test_globals_and_consts(double %x0, double %x1) {
entry:
  %r = call [2 x double] (...) @__enzyme_batch.v2f64(ptr @read_global_and_const_agg, metadata !"enzyme_width", i64 2, metadata !"enzyme_vector", double %x0, double %x1)
  ret [2 x double] %r
}

; 5. Indirect calls and defined calls with function-pointer or global arguments.
define double @inc(double %x) {
entry:
  %r = fadd double %x, 1.0
  ret double %r
}

define double @apply_and_add_global(ptr %fn, ptr %gp, double %x) {
entry:
  %v = call double %fn(double %x)
  %g = load double, ptr %gp, align 8
  %r = fadd double %v, %g
  ret double %r
}

define double @outer_apply(double %x) {
entry:
  %r = call double @apply_and_add_global(ptr @inc, ptr @G_SCALAR, double %x)
  ret double %r
}

define [2 x double] @test_indirect_and_func_global_arg(double %x0, double %x1) {
entry:
  %r = call [2 x double] (...) @__enzyme_batch.v2f64(ptr @outer_apply, metadata !"enzyme_width", i64 2, metadata !"enzyme_vector", double %x0, double %x1)
  ret [2 x double] %r
}

; 6. width == 1 with enzyme_vector must not crash.
define double @square1(double %x) {
entry:
  %r = fmul double %x, %x
  ret double %r
}

define double @test_width_1(double %x0) {
entry:
  %r = call double (...) @__enzyme_batch.f64(ptr @square1, metadata !"enzyme_width", i64 1, metadata !"enzyme_vector", double %x0)
  ret double %r
}

; 7. Direct and mutual recursion must not stack-overflow.
define double @pow_rec(double %x, i32 %n) {
entry:
  %c = icmp sle i32 %n, 0
  br i1 %c, label %base, label %rec
base:
  ret double 1.0
rec:
  %n1 = sub i32 %n, 1
  %p = call double @pow_rec(double %x, i32 %n1)
  %r = fmul double %x, %p
  ret double %r
}

define [2 x double] @test_direct_rec(double %x0, double %x1, i32 %n) {
entry:
  %r = call [2 x double] (...) @__enzyme_batch.v2f64(ptr @pow_rec, metadata !"enzyme_width", i64 2, metadata !"enzyme_vector", double %x0, double %x1, metadata !"enzyme_scalar", i32 %n)
  ret [2 x double] %r
}

define double @even_rec(double %x, i32 %n) {
entry:
  %c = icmp sle i32 %n, 0
  br i1 %c, label %base, label %rec
base:
  ret double %x
rec:
  %n1 = sub i32 %n, 1
  %r = call double @odd_rec(double %x, i32 %n1)
  ret double %r
}

define double @odd_rec(double %x, i32 %n) {
entry:
  %c = icmp sle i32 %n, 0
  br i1 %c, label %base, label %rec
base:
  %r0 = fadd double %x, 5.0
  ret double %r0
rec:
  %n1 = sub i32 %n, 1
  %r = call double @even_rec(double %x, i32 %n1)
  ret double %r
}

define [2 x double] @test_mutual_rec(double %x0, double %x1, i32 %n) {
entry:
  %r = call [2 x double] (...) @__enzyme_batch.v2f64(ptr @even_rec, metadata !"enzyme_width", i64 2, metadata !"enzyme_vector", double %x0, double %x1, metadata !"enzyme_scalar", i32 %n)
  ret [2 x double] %r
}

; 8. fastcc calling convention must be preserved at both top-level and nested call sites.
define internal fastcc double @fast_helper(double %x) noinline {
entry:
  %r = fadd double %x, 3.0
  ret double %r
}

define internal fastcc double @fast_outer(double %x) noinline {
entry:
  %r = call fastcc double @fast_helper(double %x)
  ret double %r
}

define [2 x double] @test_fastcc(double %x0, double %x1) {
entry:
  %r = call [2 x double] (...) @__enzyme_batch.v2f64(ptr @fast_outer, metadata !"enzyme_width", i64 2, metadata !"enzyme_vector", double %x0, double %x1)
  ret [2 x double] %r
}

; 9. Incompatible return and parameter attributes (nonnull, signext, returned) must be cleared.
define nonnull ptr @ret_nonnull(ptr %p) {
entry:
  ret ptr %p
}

define [2 x ptr] @test_ret_nonnull(ptr %p0, ptr %p1) {
entry:
  %r = call [2 x ptr] (...) @__enzyme_batch.v2ptr(ptr @ret_nonnull, metadata !"enzyme_width", i64 2, metadata !"enzyme_vector", ptr %p0, ptr %p1)
  ret [2 x ptr] %r
}

define signext i32 @ret_signext(ptr nonnull readonly align 4 dereferenceable(4) %p, i32 signext %k) {
entry:
  %v = load i32, ptr %p, align 4
  %r = add i32 %v, %k
  ret i32 %r
}

define [2 x i32] @test_ret_signext(ptr %p0, ptr %p1, i32 %k0, i32 %k1) {
entry:
  %r = call [2 x i32] (...) @__enzyme_batch.v2i32(ptr @ret_signext, metadata !"enzyme_width", i64 2, metadata !"enzyme_vector", ptr %p0, ptr %p1, metadata !"enzyme_vector", i32 %k0, i32 %k1)
  ret [2 x i32] %r
}

define double @ret_returned(double returned %a, double %x) {
entry:
  ret double %a
}

define [2 x double] @test_returned_attr(double %a, double %x0, double %x1) {
entry:
  %r = call [2 x double] (...) @__enzyme_batch.v2f64(ptr @ret_returned, metadata !"enzyme_width", i64 2, metadata !"enzyme_scalar", double %a, metadata !"enzyme_vector", double %x0, double %x1)
  ret [2 x double] %r
}

; 10. Vararg defined callee falls back to lane-by-lane calls instead of passing [2 x T] as vararg.
define double @va_callee(double %x, ...) {
entry:
  ret double %x
}

define double @outer_va(double %x, double %y) {
entry:
  %r = call double (double, ...) @va_callee(double %x, double %y)
  ret double %r
}

define [2 x double] @test_vararg_callee(double %x0, double %x1, double %y0, double %y1) {
entry:
  %r = call [2 x double] (...) @__enzyme_batch.v2f64(ptr @outer_va, metadata !"enzyme_width", i64 2, metadata !"enzyme_vector", double %x0, double %x1, metadata !"enzyme_vector", double %y0, double %y1)
  ret [2 x double] %r
}

; 11. Zero-arg function with enzyme_width and trailing enzyme_width metadata.
define double @const_zero_arg() {
entry:
  ret double 7.0
}

define [2 x double] @test_zero_arg_width() {
entry:
  %r = call [2 x double] (...) @__enzyme_batch.v2f64(ptr @const_zero_arg, metadata !"enzyme_width", i64 2)
  ret [2 x double] %r
}

define [2 x double] @test_trailing_width(double %x0, double %x1) {
entry:
  %r = call [2 x double] (...) @__enzyme_batch.v2f64(ptr @square1, metadata !"enzyme_vector", double %x0, double %x1, metadata !"enzyme_width", i64 2)
  ret [2 x double] %r
}

; End-to-end runtime verification for %lli.
define i32 @main() {
entry:
  ; Check 1: alloca + GEP
  %r1 = call [2 x double] @test_alloca_gep(double 10.0, double 20.0)
  %r1_0 = extractvalue [2 x double] %r1, 0
  %r1_1 = extractvalue [2 x double] %r1, 1
  %ok1_0 = fcmp oeq double %r1_0, 10.0
  %ok1_1 = fcmp oeq double %r1_1, 20.0
  %ok1 = and i1 %ok1_0, %ok1_1
  br i1 %ok1, label %c2, label %fail

c2:
  ; Check 2: alloca + select
  %r2 = call [2 x double] @test_alloca_select(double 10.0, double 20.0, i1 false)
  %r2_0 = extractvalue [2 x double] %r2, 0
  %r2_1 = extractvalue [2 x double] %r2, 1
  %ok2_0 = fcmp oeq double %r2_0, 11.0
  %ok2_1 = fcmp oeq double %r2_1, 21.0
  %ok2 = and i1 %ok2_0, %ok2_1
  br i1 %ok2, label %c3, label %fail

c3:
  ; Check 3: alloca + PHI
  %r3 = call [2 x double] @test_alloca_phi(double 3.0, double 5.0, i1 false)
  %r3_0 = extractvalue [2 x double] %r3, 0
  %r3_1 = extractvalue [2 x double] %r3, 1
  %ok3_0 = fcmp oeq double %r3_0, 6.0
  %ok3_1 = fcmp oeq double %r3_1, 10.0
  %ok3 = and i1 %ok3_0, %ok3_1
  br i1 %ok3, label %c4, label %fail

c4:
  ; Check 4: void call lane order 0, 1, 2, 3
  call void @test_void_order(double 1.0, double 2.0, double 3.0, double 4.0)
  %l0 = load double, ptr @LOG_BUF, align 8
  %l1 = load double, ptr getelementptr inbounds ([4 x double], ptr @LOG_BUF, i64 0, i64 1), align 8
  %l2 = load double, ptr getelementptr inbounds ([4 x double], ptr @LOG_BUF, i64 0, i64 2), align 8
  %l3 = load double, ptr getelementptr inbounds ([4 x double], ptr @LOG_BUF, i64 0, i64 3), align 8
  %ok4_0 = fcmp oeq double %l0, 1.0
  %ok4_1 = fcmp oeq double %l1, 2.0
  %ok4_2 = fcmp oeq double %l2, 3.0
  %ok4_3 = fcmp oeq double %l3, 4.0
  %ok4_a = and i1 %ok4_0, %ok4_1
  %ok4_b = and i1 %ok4_2, %ok4_3
  %ok4 = and i1 %ok4_a, %ok4_b
  br i1 %ok4, label %c5, label %fail

c5:
  ; Check 5: direct global, ConstantExpr GEP, ConstantStruct
  %r5 = call [2 x double] @test_globals_and_consts(double 5.0, double 8.0)
  %r5_0 = extractvalue [2 x double] %r5, 0
  %r5_1 = extractvalue [2 x double] %r5, 1
  %ok5_0 = fcmp oeq double %r5_0, 37.0
  %ok5_1 = fcmp oeq double %r5_1, 40.0
  %ok5 = and i1 %ok5_0, %ok5_1
  br i1 %ok5, label %c6, label %fail

c6:
  ; Check 6: indirect call + func/global args
  %r6 = call [2 x double] @test_indirect_and_func_global_arg(double 4.0, double 9.0)
  %r6_0 = extractvalue [2 x double] %r6, 0
  %r6_1 = extractvalue [2 x double] %r6, 1
  %ok6_0 = fcmp oeq double %r6_0, 15.0
  %ok6_1 = fcmp oeq double %r6_1, 20.0
  %ok6 = and i1 %ok6_0, %ok6_1
  br i1 %ok6, label %c7, label %fail

c7:
  ; Check 7: width == 1
  %r7 = call double @test_width_1(double 3.0)
  %ok7 = fcmp oeq double %r7, 9.0
  br i1 %ok7, label %c8, label %fail

c8:
  ; Check 8: direct recursion (2^3=8, 3^3=27) & mutual recursion (n=3 -> odd_rec -> x+5)
  %r8 = call [2 x double] @test_direct_rec(double 2.0, double 3.0, i32 3)
  %r8_0 = extractvalue [2 x double] %r8, 0
  %r8_1 = extractvalue [2 x double] %r8, 1
  %ok8_0 = fcmp oeq double %r8_0, 8.0
  %ok8_1 = fcmp oeq double %r8_1, 27.0
  %ok8 = and i1 %ok8_0, %ok8_1
  br i1 %ok8, label %c9, label %fail

c9:
  %r9 = call [2 x double] @test_mutual_rec(double 2.0, double 10.0, i32 3)
  %r9_0 = extractvalue [2 x double] %r9, 0
  %r9_1 = extractvalue [2 x double] %r9, 1
  %ok9_0 = fcmp oeq double %r9_0, 7.0
  %ok9_1 = fcmp oeq double %r9_1, 15.0
  %ok9 = and i1 %ok9_0, %ok9_1
  br i1 %ok9, label %c10, label %fail

c10:
  ; Check 10: fastcc + zero-arg width + trailing width
  %r10 = call [2 x double] @test_fastcc(double 4.0, double 6.0)
  %r10_0 = extractvalue [2 x double] %r10, 0
  %r10_1 = extractvalue [2 x double] %r10, 1
  %ok10_0 = fcmp oeq double %r10_0, 7.0
  %ok10_1 = fcmp oeq double %r10_1, 9.0
  %ok10 = and i1 %ok10_0, %ok10_1
  br i1 %ok10, label %c11, label %fail

c11:
  %r11 = call [2 x double] @test_zero_arg_width()
  %r11_0 = extractvalue [2 x double] %r11, 0
  %r11_1 = extractvalue [2 x double] %r11, 1
  %ok11_0 = fcmp oeq double %r11_0, 7.0
  %ok11_1 = fcmp oeq double %r11_1, 7.0
  %ok11 = and i1 %ok11_0, %ok11_1
  br i1 %ok11, label %c12, label %fail

c12:
  %r12 = call [2 x double] @test_trailing_width(double 4.0, double 5.0)
  %r12_0 = extractvalue [2 x double] %r12, 0
  %r12_1 = extractvalue [2 x double] %r12, 1
  %ok12_0 = fcmp oeq double %r12_0, 16.0
  %ok12_1 = fcmp oeq double %r12_1, 25.0
  %ok12 = and i1 %ok12_0, %ok12_1
  br i1 %ok12, label %pass, label %fail

pass:
  ret i32 0

fail:
  ret i32 1
}

; CHECK-LABEL: define [2 x double] @test_fastcc(double %x0, double %x1)
; CHECK: call fastcc [2 x double] @batch_fast_outer(

; CHECK-LABEL: define internal [2 x double] @batch_use_stack_array([2 x double] %x)
; CHECK:   %arr0 = alloca [2 x double], align 8
; CHECK-NEXT:   %arr1 = alloca [2 x double], align 8
; CHECK-NEXT:   %p10 = getelementptr inbounds [2 x double], ptr %arr0, i64 0, i64 1
; CHECK-NEXT:   %p11 = getelementptr inbounds [2 x double], ptr %arr1, i64 0, i64 1
; CHECK-NEXT:   store double %unwrap.x0, ptr %p10, align 8
; CHECK-NEXT:   store double %unwrap.x1, ptr %p11, align 8
; CHECK-NEXT:   %v0 = load double, ptr %p10, align 8
; CHECK-NEXT:   %v1 = load double, ptr %p11, align 8

; CHECK-LABEL: define internal [2 x double] @batch_use_stack_select([2 x double] %x, i1 %c)
; CHECK:   %p0 = select i1 %c, ptr %a0, ptr %b0
; CHECK-NEXT:   %p1 = select i1 %c, ptr %a1, ptr %b1
; CHECK-NEXT:   %v0 = load double, ptr %p0, align 8
; CHECK-NEXT:   %v1 = load double, ptr %p1, align 8

; CHECK-LABEL: define internal [2 x double] @batch_use_stack_phi([2 x double] %x, i1 %c)
; CHECK: join:
; CHECK-NEXT:   %p0 = phi ptr [ %a0, %l1 ], [ %b0, %l2 ]
; CHECK-NEXT:   %p = phi ptr [ %a1, %l1 ], [ %b1, %l2 ]
; CHECK-NEXT:   %v0 = load double, ptr %p0, align 8
; CHECK-NEXT:   %v1 = load double, ptr %p, align 8

; CHECK-LABEL: define internal void @batch_log_val(ptr %fn, [4 x double] %x)
; CHECK:   call void %fn(double %unwrap.x0)
; CHECK-NEXT:   call void %fn(double %unwrap.x1)
; CHECK-NEXT:   call void %fn(double %unwrap.x2)
; CHECK-NEXT:   call void %fn(double %unwrap.x3)

; CHECK-LABEL: define internal double @batch_square1(double %x)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %r = fmul double %x, %x
; CHECK-NEXT:   ret double %r

; CHECK-LABEL: define internal [2 x double] @batch_pow_rec([2 x double] %x, i32 %n)
; CHECK:   %p = call [2 x double] @batch_pow_rec([2 x double]

; CHECK-LABEL: define internal fastcc [2 x double] @batch_fast_outer([2 x double] %x)
; CHECK:   %r = call fastcc [2 x double] @batch_fast_helper([2 x double]

; CHECK-LABEL: define internal [2 x double] @batch_outer_va([2 x double] %x, [2 x double] %y)
; CHECK:   %r0 = call double (double, ...) @va_callee(double %unwrap.x0, double %unwrap.y0)
; CHECK-NEXT:   %r1 = call double (double, ...) @va_callee(double %unwrap.x1, double %unwrap.y1)
