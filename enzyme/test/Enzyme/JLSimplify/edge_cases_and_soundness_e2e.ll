; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="jl-inst-simplify,verify" -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="jl-inst-simplify,verify" | %lli; fi

declare noalias ptr @malloc(i64)
declare noalias ptr @calloc(i64, i64)
declare void @free(ptr)
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)

@G = global [2 x i64] [i64 1, i64 2], align 8

; Helper with enzyme_allocation attribute for E2E runtime tests (returns a
; freshly allocated box initialized with a pointer to @G).
define noalias nonnull ptr @make_box() #0 {
  %p = call noalias ptr @malloc(i64 8)
  store ptr @G, ptr %p, align 8
  ret ptr %p
}

define ptr @read_ptr_readonly(ptr %p) readonly {
  %v = load ptr, ptr %p, align 8
  ret ptr %v
}

; 1. Ordered pointer comparisons between distinct allocas must not fold.
define i1 @test_ordered_ptr_cmp() {
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %lt = icmp ult ptr %a, %b
  %gt = icmp ugt ptr %a, %b
  %le = icmp ule ptr %a, %b
  %ge = icmp uge ptr %a, %b
  %slt = icmp slt ptr %a, %b
  %sgt = icmp sgt ptr %a, %b
  %sle = icmp sle ptr %a, %b
  %sge = icmp sge ptr %a, %b
  %x1 = xor i1 %lt, %gt
  %x2 = xor i1 %le, %ge
  %x3 = xor i1 %slt, %sgt
  %x4 = xor i1 %sle, %sge
  %r1 = and i1 %x1, %x2
  %r2 = and i1 %x3, %x4
  %res = and i1 %r1, %r2
  ret i1 %res
}
; CHECK-LABEL: define i1 @test_ordered_ptr_cmp()
; CHECK: %lt = icmp ult ptr %a, %b
; CHECK: %gt = icmp ugt ptr %a, %b
; CHECK: %le = icmp ule ptr %a, %b
; CHECK: %ge = icmp uge ptr %a, %b
; CHECK: %x1 = xor i1 %lt, %gt
; CHECK: %x2 = xor i1 %le, %ge
; CHECK: ret i1 %res

; 2. Integer comparisons across different casts must not fold.
define i1 @test_int_zext_sext(i32 %x) {
  %z = zext i32 %x to i64
  %s = sext i32 %x to i64
  %eq = icmp eq i64 %z, %s
  ret i1 %eq
}
; CHECK-LABEL: define i1 @test_int_zext_sext(i32 %x)
; CHECK: %eq = icmp eq i64 %z, %s
; CHECK-NEXT: ret i1 %eq

define i1 @test_int_bitcast_fptosi(double %x) {
  %b = bitcast double %x to i64
  %f = fptosi double %x to i64
  %eq = icmp eq i64 %b, %f
  ret i1 %eq
}
; CHECK-LABEL: define i1 @test_int_bitcast_fptosi(double %x)
; CHECK: %eq = icmp eq i64 %b, %f
; CHECK-NEXT: ret i1 %eq

; 3. Non-zero ConstantExpr GEP must not fold equal to base global.
define i1 @test_const_gep_neq() {
  %eq = icmp eq ptr @G, getelementptr inbounds ([2 x i64], ptr @G, i64 0, i64 1)
  ret i1 %eq
}
; CHECK-LABEL: define i1 @test_const_gep_neq()
; CHECK: %eq = icmp eq ptr @G, getelementptr inbounds ([2 x i64], ptr @G, i64 0, i64 1)
; CHECK-NEXT: ret i1 %eq

; 4. Lossy pointer cast roundtrip must not fold equal to original pointer.
define i1 @test_lossy_ptr_roundtrip(ptr %p) {
  %i = ptrtoint ptr %p to i64
  %t = trunc i64 %i to i32
  %z = zext i32 %t to i64
  %q = inttoptr i64 %z to ptr
  %eq = icmp eq ptr %p, %q
  ret i1 %eq
}
; CHECK-LABEL: define i1 @test_lossy_ptr_roundtrip(ptr %p)
; CHECK: %eq = icmp eq ptr %p, %q
; CHECK-NEXT: ret i1 %eq

; 5. Untouched noalias pointer arguments may be identical pointers.
define i1 @test_noalias_args(ptr noalias %a, ptr noalias %b) {
  %eq = icmp eq ptr %a, %b
  ret i1 %eq
}
; CHECK-LABEL: define i1 @test_noalias_args(ptr noalias %a, ptr noalias %b)
; CHECK: %eq = icmp eq ptr %a, %b
; CHECK-NEXT: ret i1 %eq

define i1 @test_noalias_one_arg(ptr noalias %a, ptr %b) {
  %eq = icmp eq ptr %a, %b
  ret i1 %eq
}
; CHECK-LABEL: define i1 @test_noalias_one_arg(ptr noalias %a, ptr %b)
; CHECK: %eq = icmp eq ptr %a, %b
; CHECK-NEXT: ret i1 %eq

; 6. Capture of a fresh pointer BEFORE the box allocation must be detected.
define i1 @test_capture_before_alloc(ptr %gslot) {
  %e = alloca i8, align 1
  store ptr %e, ptr %gslot, align 8
  %box = call noalias nonnull ptr @malloc(i64 8)
  %copied = load ptr, ptr %gslot, align 8
  store ptr %copied, ptr %box, align 8
  %v = load ptr, ptr %box, align 8
  %eq = icmp eq ptr %v, %e
  call void @free(ptr %box)
  ret i1 %eq
}
; CHECK-LABEL: define i1 @test_capture_before_alloc(ptr %gslot)
; CHECK: %eq = icmp eq ptr %v, %e
; CHECK: ret i1 %eq

; 7. Loop-carried capture after the load in the same basic block.
define i1 @test_single_bb_loop_capture(ptr %gslot) {
entry:
  %e = alloca i8, align 1
  store ptr null, ptr %gslot, align 8
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %inc, %loop ]
  %v = load ptr, ptr %gslot, align 8
  %eq = icmp eq ptr %v, %e
  store ptr %e, ptr %gslot, align 8
  %inc = add i32 %i, 1
  %cont = icmp slt i32 %inc, 2
  br i1 %cont, label %loop, label %exit

exit:
  ret i1 %eq
}
; CHECK-LABEL: define i1 @test_single_bb_loop_capture(ptr %gslot)
; CHECK: %eq = icmp eq ptr %v, %e
; CHECK: exit:
; CHECK-NEXT: ret i1 %eq

; 8. Loop-carried capture in a 2-BB loop latch.
define i1 @test_two_bb_loop_capture(ptr %gslot) {
entry:
  %e = alloca i8, align 1
  store ptr null, ptr %gslot, align 8
  br label %header

header:
  %i = phi i32 [ 0, %entry ], [ %inc, %latch ]
  %v = load ptr, ptr %gslot, align 8
  %eq = icmp eq ptr %v, %e
  %done = icmp eq i32 %i, 1
  br i1 %done, label %exit, label %latch

latch:
  store ptr %e, ptr %gslot, align 8
  %inc = add i32 %i, 1
  br label %header

exit:
  ret i1 %eq
}
; CHECK-LABEL: define i1 @test_two_bb_loop_capture(ptr %gslot)
; CHECK: %eq = icmp eq ptr %v, %e
; CHECK: exit:
; CHECK-NEXT: ret i1 %eq

; 9. Loop-carried capture of a load from an unwritten allocation when end
; dominates the load inside the loop.
define i1 @test_loop_alloc_load_capture(ptr %gslot) {
entry:
  %box = call noalias nonnull ptr @make_box()
  store ptr null, ptr %gslot, align 8
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %inc, %loop ]
  %end = load ptr, ptr %gslot, align 8
  %ld = load ptr, ptr %box, align 8
  store ptr %ld, ptr %gslot, align 8
  %eq = icmp eq ptr %ld, %end
  %inc = add i32 %i, 1
  %cont = icmp slt i32 %inc, 2
  br i1 %cont, label %loop, label %exit

exit:
  call void @free(ptr %box)
  ret i1 %eq
}
; CHECK-LABEL: define i1 @test_loop_alloc_load_capture(ptr %gslot)
; CHECK: %eq = icmp eq ptr %ld, %end
; CHECK: exit:
; CHECK: ret i1 %eq

; 10. calloc zero-initializes memory to null, so loads compare equal to null.
define i1 @test_calloc_load_null() {
  %c = call noalias ptr @calloc(i64 1, i64 8)
  %p = load ptr, ptr %c, align 8
  %eq = icmp eq ptr %p, null
  call void @free(ptr %c)
  ret i1 %eq
}
; CHECK-LABEL: define i1 @test_calloc_load_null()
; CHECK: %eq = icmp eq ptr %p, null
; CHECK: ret i1 %eq

define i1 @test_calloc_two_loads() {
  %c1 = call noalias ptr @calloc(i64 1, i64 8)
  %c2 = call noalias ptr @calloc(i64 1, i64 8)
  %p1 = load ptr, ptr %c1, align 8
  %p2 = load ptr, ptr %c2, align 8
  %eq = icmp eq ptr %p1, %p2
  call void @free(ptr %c1)
  call void @free(ptr %c2)
  ret i1 %eq
}
; CHECK-LABEL: define i1 @test_calloc_two_loads()
; CHECK: %eq = icmp eq ptr %p1, %p2
; CHECK: ret i1 %eq

; 11. Escaped / copied / selected / readonly-called allocations must not fold.
define i1 @test_alloc_escaped_via_slot() {
  %a = call noalias nonnull ptr @make_box()
  %p1 = load ptr, ptr %a, align 8
  %slot = alloca ptr, align 8
  store ptr %a, ptr %slot, align 8
  %a2 = load ptr, ptr %slot, align 8
  %p2 = load ptr, ptr %a2, align 8
  %eq = icmp eq ptr %p1, %p2
  call void @free(ptr %a)
  ret i1 %eq
}
; CHECK-LABEL: define i1 @test_alloc_escaped_via_slot()
; CHECK: %eq = icmp eq ptr %p1, %p2
; CHECK: ret i1 %eq

define i1 @test_alloc_copied_via_memcpy() {
  %a = call noalias nonnull ptr @make_box()
  %p1 = load ptr, ptr %a, align 8
  %other = alloca ptr, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr %other, ptr %a, i64 8, i1 false)
  %p2 = load ptr, ptr %other, align 8
  %eq = icmp eq ptr %p1, %p2
  call void @free(ptr %a)
  ret i1 %eq
}
; CHECK-LABEL: define i1 @test_alloc_copied_via_memcpy()
; CHECK: %eq = icmp eq ptr %p1, %p2
; CHECK: ret i1 %eq

define i1 @test_alloc_via_select(i1 %cond, ptr %fallback) {
  %a = call noalias nonnull ptr @make_box()
  %p1 = load ptr, ptr %a, align 8
  %sel = select i1 %cond, ptr %a, ptr %fallback
  %p2 = load ptr, ptr %sel, align 8
  %eq = icmp eq ptr %p1, %p2
  call void @free(ptr %a)
  ret i1 %eq
}
; CHECK-LABEL: define i1 @test_alloc_via_select(
; CHECK: %eq = icmp eq ptr %p1, %p2
; CHECK: ret i1 %eq

define i1 @test_alloc_via_readonly_call() {
  %a = call noalias nonnull ptr @make_box()
  %p1 = load ptr, ptr %a, align 8
  %p2 = call ptr @read_ptr_readonly(ptr %a)
  %eq = icmp eq ptr %p1, %p2
  call void @free(ptr %a)
  ret i1 %eq
}
; CHECK-LABEL: define i1 @test_alloc_via_readonly_call()
; CHECK: %eq = icmp eq ptr %p1, %p2
; CHECK: ret i1 %eq

; 12. Positive tests: sound folds must still trigger.
define i1 @test_valid_distinct_allocas_eq() {
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %eq = icmp eq ptr %a, %b
  ret i1 %eq
}
; CHECK-LABEL: define i1 @test_valid_distinct_allocas_eq()
; CHECK: ret i1 false

define i1 @test_valid_distinct_allocas_ne() {
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %ne = icmp ne ptr %a, %b
  ret i1 %ne
}
; CHECK-LABEL: define i1 @test_valid_distinct_allocas_ne()
; CHECK: ret i1 true

define i1 @test_valid_same_ptr_zero_gep(ptr %p) {
  %q = getelementptr inbounds i8, ptr %p, i64 0
  %eq = icmp eq ptr %p, %q
  ret i1 %eq
}
; CHECK-LABEL: define i1 @test_valid_same_ptr_zero_gep(ptr %p)
; CHECK: ret i1 true

define i1 @test_valid_two_live_nonnull_mallocs() {
  %p = call noalias nonnull ptr @malloc(i64 16)
  %q = call noalias nonnull ptr @malloc(i64 16)
  %eq = icmp eq ptr %p, %q
  call void @free(ptr %p)
  call void @free(ptr %q)
  ret i1 %eq
}
; CHECK-LABEL: define i1 @test_valid_two_live_nonnull_mallocs()
; CHECK: call void @free(ptr %q)
; CHECK-NEXT: ret i1 false

define i1 @test_valid_alloca_vs_arg(ptr %arg) {
  %a = alloca i32, align 4
  %eq = icmp eq ptr %a, %arg
  ret i1 %eq
}
; CHECK-LABEL: define i1 @test_valid_alloca_vs_arg(ptr %arg)
; CHECK: ret i1 false

; End-to-end runtime harness verifying all runtime-observable properties.
define i32 @main() {
entry:
  %gslot = alloca ptr, align 8
  %fallback = alloca ptr, align 8
  store ptr null, ptr %fallback, align 8

  %r1 = call i1 @test_ordered_ptr_cmp()
  br i1 %r1, label %c2, label %fail

c2:
  %r2 = call i1 @test_int_zext_sext(i32 -1)
  br i1 %r2, label %fail, label %c3

c3:
  %r3 = call i1 @test_int_bitcast_fptosi(double 1.0)
  br i1 %r3, label %fail, label %c4

c4:
  %r4 = call i1 @test_const_gep_neq()
  br i1 %r4, label %fail, label %c5

c5:
  %hi = inttoptr i64 4294967297 to ptr
  %r5 = call i1 @test_lossy_ptr_roundtrip(ptr %hi)
  br i1 %r5, label %fail, label %c6

c6:
  %r6 = call i1 @test_noalias_args(ptr @G, ptr @G)
  br i1 %r6, label %c7, label %fail

c7:
  %r7 = call i1 @test_noalias_one_arg(ptr @G, ptr @G)
  br i1 %r7, label %c8, label %fail

c8:
  %r8 = call i1 @test_capture_before_alloc(ptr %gslot)
  br i1 %r8, label %c9, label %fail

c9:
  %r9 = call i1 @test_single_bb_loop_capture(ptr %gslot)
  br i1 %r9, label %c10, label %fail

c10:
  %r10 = call i1 @test_two_bb_loop_capture(ptr %gslot)
  br i1 %r10, label %c11, label %fail

c11:
  %r11 = call i1 @test_loop_alloc_load_capture(ptr %gslot)
  br i1 %r11, label %c12, label %fail

c12:
  %r12 = call i1 @test_calloc_load_null()
  br i1 %r12, label %c13, label %fail

c13:
  %r13 = call i1 @test_calloc_two_loads()
  br i1 %r13, label %c14, label %fail

c14:
  %r14 = call i1 @test_alloc_escaped_via_slot()
  br i1 %r14, label %c15, label %fail

c15:
  %r15 = call i1 @test_alloc_copied_via_memcpy()
  br i1 %r15, label %c16, label %fail

c16:
  %r16 = call i1 @test_alloc_via_select(i1 true, ptr %fallback)
  br i1 %r16, label %c17, label %fail

c17:
  %r17 = call i1 @test_alloc_via_readonly_call()
  br i1 %r17, label %c18, label %fail

c18:
  %r18 = call i1 @test_valid_distinct_allocas_eq()
  br i1 %r18, label %fail, label %c19

c19:
  %r19 = call i1 @test_valid_distinct_allocas_ne()
  br i1 %r19, label %c20, label %fail

c20:
  %r20 = call i1 @test_valid_same_ptr_zero_gep(ptr @G)
  br i1 %r20, label %c21, label %fail

c21:
  %r21 = call i1 @test_valid_two_live_nonnull_mallocs()
  br i1 %r21, label %fail, label %c22

c22:
  %r22 = call i1 @test_valid_alloca_vs_arg(ptr @G)
  br i1 %r22, label %fail, label %ok

ok:
  ret i32 0

fail:
  ret i32 1
}

attributes #0 = { "enzyme_allocation" }
