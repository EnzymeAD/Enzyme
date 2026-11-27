; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="simple-gvn,verify" -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="simple-gvn,verify" -S | %lli; fi

; 1. Loop with store AFTER load in the same block: backedge overwrites %p.
define double @test_loop_after_load(ptr noalias %p, i32 %n) {
entry:
  store double 1.0, ptr %p, align 8
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %v = load double, ptr %p, align 8
  store double 2.0, ptr %p, align 8
  %next = add i32 %i, 1
  %cmp = icmp slt i32 %next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret double %v
}

; 2. Multi-block loop with nocapture call AFTER load in loop header.
define internal void @clobber_double(ptr nocapture %p) noinline {
entry:
  store double 3.0, ptr %p, align 8
  ret void
}

define double @test_loop_call_after_load(ptr noalias %p, i32 %n) {
entry:
  store double 1.0, ptr %p, align 8
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %next, %latch ]
  %v = load double, ptr %p, align 8
  call void @clobber_double(ptr nocapture %p)
  br label %latch

latch:
  %next = add i32 %i, 1
  %cmp = icmp slt i32 %next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret double %v
}

; 3. Acyclic block with store AFTER load: forwarding from dominating store is still valid.
define double @test_acyclic_store_after_load(ptr noalias %p, i1 %c) {
entry:
  store double 5.0, ptr %p, align 8
  br i1 %c, label %mid, label %target

mid:
  br label %target

target:
  %v = load double, ptr %p, align 8
  store double 9.0, ptr %p, align 8
  ret double %v
}

; 4. Dominating store in intermediate block overrides earlier entry store in BFS.
define i32 @test_dom_override_bfs(ptr noalias %p, i1 %c) {
entry:
  store i32 1, ptr %p, align 4
  br label %mid

mid:
  store i32 2, ptr %p, align 4
  br i1 %c, label %left, label %right

left:
  br label %join

right:
  br label %join

join:
  %v = load i32, ptr %p, align 4
  ret i32 %v
}

; 5. Self-store `store ptr %p, ptr %p` escapes %p into memory.
define ptr @test_self_store_escape(ptr noalias %p) {
entry:
  store ptr %p, ptr %p, align 8
  %q = load ptr, ptr %p, align 8
  store ptr null, ptr %q, align 8
  %v = load ptr, ptr %p, align 8
  ret ptr %v
}

; 6. Volatile and atomic loads/stores must not be eliminated or forwarded.
define i32 @test_volatile_load(ptr noalias %p) {
entry:
  store i32 42, ptr %p, align 4
  %v = load volatile i32, ptr %p, align 4
  ret i32 %v
}

define i32 @test_atomic_load(ptr noalias %p) {
entry:
  store i32 42, ptr %p, align 4
  %v = load atomic i32, ptr %p seq_cst, align 4
  ret i32 %v
}

define i32 @test_volatile_store(ptr noalias %p) {
entry:
  store volatile i32 42, ptr %p, align 4
  %v = load i32, ptr %p, align 4
  ret i32 %v
}

; 7. ptrtoint/inttoptr truncation must not be treated as preserving offset.
define i32 @test_ptrtoint_trunc(ptr noalias %p) {
entry:
  %i = ptrtoint ptr %p to i32
  %q = inttoptr i32 %i to ptr
  store i32 99, ptr %q, align 4
  %v = load i32, ptr %p, align 4
  ret i32 %v
}

; 8. Failed conversion or unsafe non-noalias forwarding must not leave dead casts.
declare void @clobber_any()

define { i32, i32 } @test_no_orphan_bitcast(ptr noalias %p, double %x) {
entry:
  store double %x, ptr %p, align 8
  %v = load { i32, i32 }, ptr %p, align 4
  ret { i32, i32 } %v
}

define i32 @test_no_orphan_unsafe(ptr %p, i64 %x) {
entry:
  store i64 %x, ptr %p, align 8
  call void @clobber_any()
  %v = load i32, ptr %p, align 4
  ret i32 %v
}

; 9. Indirect callee use, empty aggregates, and zero-sized loads must not crash.
define void @test_callee_use(ptr noalias %p) {
entry:
  call void %p()
  ret void
}

define [0 x i8] @test_empty_agg(ptr noalias %p) {
entry:
  %a = load {}, ptr %p, align 1
  %b = load [0 x i8], ptr %p, align 1
  ret [0 x i8] %b
}

define {} @test_zero_load_offset(ptr noalias %p, i32 %x) {
entry:
  store i32 %x, ptr %p, align 4
  %g = getelementptr i8, ptr %p, i64 2
  %v = load {}, ptr %g, align 1
  ret {} %v
}

; E2E verification harness for %lli.
define i32 @main() {
entry:
  %buf = alloca double, align 8
  %r1 = call double @test_loop_after_load(ptr %buf, i32 2)
  %ok1 = fcmp oeq double %r1, 2.0
  br i1 %ok1, label %check2, label %fail

check2:
  %r2 = call double @test_loop_call_after_load(ptr %buf, i32 2)
  %ok2 = fcmp oeq double %r2, 3.0
  br i1 %ok2, label %check3, label %fail

check3:
  %r3 = call double @test_acyclic_store_after_load(ptr %buf, i1 true)
  %ok3 = fcmp oeq double %r3, 5.0
  br i1 %ok3, label %check4, label %fail

check4:
  %ibuf = alloca i32, align 4
  %r4 = call i32 @test_dom_override_bfs(ptr %ibuf, i1 false)
  %ok4 = icmp eq i32 %r4, 2
  br i1 %ok4, label %check5, label %fail

check5:
  %pbuf = alloca ptr, align 8
  %r5 = call ptr @test_self_store_escape(ptr %pbuf)
  %ok5 = icmp eq ptr %r5, null
  br i1 %ok5, label %pass, label %fail

pass:
  ret i32 0

fail:
  ret i32 1
}

; CHECK-LABEL: define double @test_loop_after_load(ptr noalias %p, i32 %n)
; CHECK: loop:
; CHECK:   %v = load double, ptr %p, align 8
; CHECK:   store double 2.000000e+00, ptr %p, align 8
; CHECK: exit:
; CHECK-NEXT:   ret double %v

; CHECK-LABEL: define double @test_loop_call_after_load(ptr noalias %p, i32 %n)
; CHECK: loop:
; CHECK:   %v = load double, ptr %p, align 8
; CHECK:   call void @clobber_double(ptr nocapture %p)
; CHECK: exit:
; CHECK-NEXT:   ret double %v

; CHECK-LABEL: define double @test_acyclic_store_after_load(ptr noalias %p, i1 %c)
; CHECK: target:
; CHECK-NEXT:   store double 9.000000e+00, ptr %p, align 8
; CHECK-NEXT:   ret double 5.000000e+00

; CHECK-LABEL: define i32 @test_dom_override_bfs(ptr noalias %p, i1 %c)
; CHECK: join:
; CHECK-NEXT:   ret i32 2

; CHECK-LABEL: define ptr @test_self_store_escape(ptr noalias %p)
; CHECK-NEXT: entry:
; CHECK-NEXT:   store ptr %p, ptr %p, align 8
; CHECK-NEXT:   %q = load ptr, ptr %p, align 8
; CHECK-NEXT:   store ptr null, ptr %q, align 8
; CHECK-NEXT:   %v = load ptr, ptr %p, align 8
; CHECK-NEXT:   ret ptr %v

; CHECK-LABEL: define i32 @test_volatile_load(ptr noalias %p)
; CHECK-NEXT: entry:
; CHECK-NEXT:   store i32 42, ptr %p, align 4
; CHECK-NEXT:   %v = load volatile i32, ptr %p, align 4
; CHECK-NEXT:   ret i32 %v

; CHECK-LABEL: define i32 @test_atomic_load(ptr noalias %p)
; CHECK-NEXT: entry:
; CHECK-NEXT:   store i32 42, ptr %p, align 4
; CHECK-NEXT:   %v = load atomic i32, ptr %p seq_cst, align 4
; CHECK-NEXT:   ret i32 %v

; CHECK-LABEL: define i32 @test_volatile_store(ptr noalias %p)
; CHECK-NEXT: entry:
; CHECK-NEXT:   store volatile i32 42, ptr %p, align 4
; CHECK-NEXT:   %v = load i32, ptr %p, align 4
; CHECK-NEXT:   ret i32 %v

; CHECK-LABEL: define i32 @test_ptrtoint_trunc(ptr noalias %p)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %i = ptrtoint ptr %p to i32
; CHECK-NEXT:   %q = inttoptr i32 %i to ptr
; CHECK-NEXT:   store i32 99, ptr %q, align 4
; CHECK-NEXT:   %v = load i32, ptr %p, align 4
; CHECK-NEXT:   ret i32 %v

; CHECK-LABEL: define { i32, i32 } @test_no_orphan_bitcast(ptr noalias %p, double %x)
; CHECK-NEXT: entry:
; CHECK-NEXT:   store double %x, ptr %p, align 8
; CHECK-NEXT:   %v = load { i32, i32 }, ptr %p, align 4
; CHECK-NEXT:   ret { i32, i32 } %v

; CHECK-LABEL: define i32 @test_no_orphan_unsafe(ptr %p, i64 %x)
; CHECK-NEXT: entry:
; CHECK-NEXT:   store i64 %x, ptr %p, align 8
; CHECK-NEXT:   call void @clobber_any()
; CHECK-NEXT:   %v = load i32, ptr %p, align 4
; CHECK-NEXT:   ret i32 %v

; CHECK-LABEL: define void @test_callee_use(ptr noalias %p)
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void %p()
; CHECK-NEXT:   ret void

; CHECK-LABEL: define [0 x i8] @test_empty_agg(ptr noalias %p)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %a = load {}, ptr %p, align 1
; CHECK-NEXT:   %b = load [0 x i8], ptr %p, align 1
; CHECK-NEXT:   ret [0 x i8] %b

; CHECK-LABEL: define {} @test_zero_load_offset(ptr noalias %p, i32 %x)
; CHECK-NEXT: entry:
; CHECK-NEXT:   store i32 %x, ptr %p, align 4
; CHECK-NEXT:   %g = getelementptr i8, ptr %p, i64 2
; CHECK-NEXT:   %v = load {}, ptr %g, align 1
; CHECK-NEXT:   ret {} %v
