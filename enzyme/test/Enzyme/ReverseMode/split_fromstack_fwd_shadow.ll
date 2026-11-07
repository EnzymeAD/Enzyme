; RUN: %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s

declare ptr @__enzyme_augmentfwd(...)
declare void @__enzyme_reverse(...)

define internal void @helper(ptr %dst, i64 %idx, ptr %active_ptr) {
entry:
  %tmp = alloca [2 x i64], align 8
  store i64 11, ptr %tmp, align 8
  %slot1 = getelementptr inbounds i64, ptr %tmp, i64 1
  store i64 22, ptr %slot1, align 8
  %sel = getelementptr inbounds i64, ptr %tmp, i64 %idx
  %v = load i64, ptr %sel, align 8
  store i64 %v, ptr %dst, align 8
  %p_slot = getelementptr inbounds i8, ptr %dst, i64 8
  store ptr %active_ptr, ptr %p_slot, align 8
  ret void
}

define void @caller(ptr %x) {
entry:
  %buf = alloca { i64, ptr }, align 8
  call void @helper(ptr %buf, i64 1, ptr %x)
  %p_slot = getelementptr inbounds i8, ptr %buf, i64 8
  %p = load ptr, ptr %p_slot, align 8
  %v = load double, ptr %p, align 8
  %sq = fmul double %v, %v
  store double %sq, ptr %p, align 8
  ret void
}

define void @test_derivative(ptr %x, ptr %dx) {
entry:
  %tape = call ptr (...) @__enzyme_augmentfwd(ptr @caller, metadata !"enzyme_dup", ptr %x, ptr %dx)
  call void (...) @__enzyme_reverse(ptr @caller, metadata !"enzyme_dup", ptr %x, ptr %dx, ptr %tape)
  ret void
}

; CHECK: define internal void @augmented_helper(
; CHECK-NOT: call {{.*}}@malloc(
; CHECK:   %[[ALLOC:.+]] = alloca [2 x i64]
; CHECK:   store i64 %v, ptr %"dst'"
; CHECK:   store i64 %v, ptr %dst
; CHECK:   ret void

; CHECK: define internal void @diffehelper(
; CHECK-NOT: call void @free(
; CHECK:   ret void
