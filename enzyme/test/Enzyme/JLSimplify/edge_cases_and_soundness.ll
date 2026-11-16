; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="jl-inst-simplify,verify" -S | FileCheck %s; fi

declare noalias ptr @malloc(i64)
declare void @free(ptr)
declare noalias nonnull ptr @__rust_alloc_zeroed(i64, i64)
declare noalias nonnull ptr @swift_allocObject(ptr, i64, i64)
declare noalias nonnull ptr @custom_zeroed_alloc(i64) #1
declare ptr @jl_reshape_array(ptr, ptr, ptr)
declare void @use_bundle()
declare i32 @__gxx_personality_v0(...)

@U1 = unnamed_addr constant i32 42
@U2 = unnamed_addr constant i32 42
@W1 = extern_weak global i32
@W2 = extern_weak global i32
@Meta = global i8 0

; 1. jl_reshape_array returns a new wrapper object, not the input pointer.
define i1 @test_reshape_array_eq(ptr %T, ptr %arr, ptr %dims) {
  %r = call ptr @jl_reshape_array(ptr %T, ptr %arr, ptr %dims)
  %eq = icmp eq ptr %r, %arr
  ret i1 %eq
}
; CHECK-LABEL: define i1 @test_reshape_array_eq(
; CHECK: %eq = icmp eq ptr %r, %arr
; CHECK-NEXT: ret i1 %eq

; 2. One-past-the-end pointer equality must not fold to false.
define i1 @test_one_past_end() {
  %a = alloca i8, align 1
  %b = alloca i8, align 1
  %end = getelementptr inbounds i8, ptr %a, i64 1
  %eq = icmp eq ptr %end, %b
  ret i1 %eq
}
; CHECK-LABEL: define i1 @test_one_past_end()
; CHECK: %eq = icmp eq ptr %end, %b
; CHECK-NEXT: ret i1 %eq

; 3. unnamed_addr globals may be merged by the linker.
define i1 @test_unnamed_addr_globals() {
  %eq = icmp eq ptr @U1, @U2
  ret i1 %eq
}
; CHECK-LABEL: define i1 @test_unnamed_addr_globals()
; CHECK: %eq = icmp eq ptr @U1, @U2
; CHECK-NEXT: ret i1 %eq

; 4. extern_weak globals may both resolve to null.
define i1 @test_extern_weak_globals() {
  %eq = icmp eq ptr @W1, @W2
  ret i1 %eq
}
; CHECK-LABEL: define i1 @test_extern_weak_globals()
; CHECK: %eq = icmp eq ptr @W1, @W2
; CHECK-NEXT: ret i1 %eq

; 5. Zero-size allocas may share the same address.
define i1 @test_zero_size_allocas() {
  %a = alloca [0 x i8], align 1
  %b = alloca [0 x i8], align 1
  %eq = icmp eq ptr %a, %b
  ret i1 %eq
}
; CHECK-LABEL: define i1 @test_zero_size_allocas()
; CHECK: %eq = icmp eq ptr %a, %b
; CHECK-NEXT: ret i1 %eq

; 6. Non-nonnull malloc may return null on OOM.
define i1 @test_malloc_null() {
  %p = call noalias ptr @malloc(i64 16)
  %eq = icmp eq ptr %p, null
  ret i1 %eq
}
; CHECK-LABEL: define i1 @test_malloc_null()
; CHECK: %eq = icmp eq ptr %p, null
; CHECK-NEXT: ret i1 %eq

define i1 @test_two_nullable_mallocs() {
  %p = call noalias ptr @malloc(i64 16)
  %q = call noalias ptr @malloc(i64 16)
  %eq = icmp eq ptr %p, %q
  ret i1 %eq
}
; CHECK-LABEL: define i1 @test_two_nullable_mallocs()
; CHECK: %eq = icmp eq ptr %p, %q
; CHECK-NEXT: ret i1 %eq

; 7. Freed heap allocation address may be reused by a subsequent allocation.
define i1 @test_malloc_free_malloc() {
  %p = call noalias nonnull ptr @malloc(i64 16)
  call void @free(ptr %p)
  %q = call noalias nonnull ptr @malloc(i64 16)
  %eq = icmp eq ptr %p, %q
  ret i1 %eq
}
; CHECK-LABEL: define i1 @test_malloc_free_malloc()
; CHECK: %eq = icmp eq ptr %p, %q
; CHECK-NEXT: ret i1 %eq

define i1 @test_free_arg_malloc(ptr %arg) {
  call void @free(ptr %arg)
  %q = call noalias nonnull ptr @malloc(i64 16)
  %eq = icmp eq ptr %arg, %q
  ret i1 %eq
}
; CHECK-LABEL: define i1 @test_free_arg_malloc(ptr %arg)
; CHECK: %eq = icmp eq ptr %arg, %q
; CHECK-NEXT: ret i1 %eq

; 8. InvokeInst allocation must not crash notCapturedBefore.
define i1 @test_invoke_alloc(ptr %slot) personality ptr @__gxx_personality_v0 {
entry:
  %p = invoke noalias nonnull ptr @malloc(i64 16)
      to label %cont unwind label %lpad

cont:
  store ptr %p, ptr %slot, align 8
  %v = load ptr, ptr %slot, align 8
  %eq = icmp eq ptr %p, %v
  call void @free(ptr %p)
  ret i1 %eq

lpad:
  %lp = landingpad { ptr, i32 }
      cleanup
  ret i1 false
}
; CHECK-LABEL: define i1 @test_invoke_alloc(ptr %slot)
; CHECK: %eq = icmp eq ptr %p, %v
; CHECK: ret i1 %eq

; 9. Zero-initialized / pre-initialized allocators must not fold loads away.
define i1 @test_rust_alloc_zeroed_null() {
  %c = call noalias nonnull ptr @__rust_alloc_zeroed(i64 8, i64 8)
  %p = load ptr, ptr %c, align 8
  %eq = icmp eq ptr %p, null
  ret i1 %eq
}
; CHECK-LABEL: define i1 @test_rust_alloc_zeroed_null()
; CHECK: %eq = icmp eq ptr %p, null
; CHECK-NEXT: ret i1 %eq

define i1 @test_swift_alloc_metadata() {
  %obj = call noalias nonnull ptr @swift_allocObject(ptr @Meta, i64 32, i64 7)
  %m = load ptr, ptr %obj, align 8
  %eq = icmp eq ptr %m, @Meta
  ret i1 %eq
}
; CHECK-LABEL: define i1 @test_swift_alloc_metadata()
; CHECK: %eq = icmp eq ptr %m, @Meta
; CHECK-NEXT: ret i1 %eq

define i1 @test_allockind_zeroed_null() {
  %c = call noalias nonnull ptr @custom_zeroed_alloc(i64 8)
  %p = load ptr, ptr %c, align 8
  %eq = icmp eq ptr %p, null
  ret i1 %eq
}
; CHECK-LABEL: define i1 @test_allockind_zeroed_null()
; CHECK: %eq = icmp eq ptr %p, null
; CHECK-NEXT: ret i1 %eq

; 10. Operand bundle capture must be recognized by notCapturedBefore.
define i1 @test_operand_bundle_capture(ptr %slot) {
  %a = alloca i8, align 1
  call void @use_bundle() [ "deopt"(ptr %a) ]
  %p = load ptr, ptr %slot, align 8
  %eq = icmp eq ptr %a, %p
  ret i1 %eq
}
; CHECK-LABEL: define i1 @test_operand_bundle_capture(ptr %slot)
; CHECK: %eq = icmp eq ptr %a, %p
; CHECK-NEXT: ret i1 %eq

attributes #1 = { "enzyme_allocation" allockind("alloc,zeroed") }
