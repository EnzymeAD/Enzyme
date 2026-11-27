; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="simple-gvn,verify" -S | FileCheck %s; fi

target datalayout = "e-m:e-p:64:64-p1:32:32-p3:32:32-i64:64-ni:10:11"

; Must not emit addrspacecast when storing a pointer in one address space and
; loading a pointer in another (except Julia AS 10 -> AS 0).
define ptr addrspace(1) @test_cross_as_ptr_diff_size(ptr noalias %p, ptr %x) {
entry:
  store ptr %x, ptr %p, align 8
  %v = load ptr addrspace(1), ptr %p, align 4
  ret ptr addrspace(1) %v
}

define ptr @test_cross_as_ptr_same_size(ptr noalias %p, ptr addrspace(2) %x) {
entry:
  store ptr addrspace(2) %x, ptr %p, align 8
  %v = load ptr, ptr %p, align 8
  ret ptr %v
}

; Must not emit inttoptr to a non-integral address space other than Julia AS 10.
define ptr addrspace(11) @test_non_integral_as11(ptr noalias %p, i64 %x) {
entry:
  store i64 %x, ptr %p, align 8
  %v = load ptr addrspace(11), ptr %p, align 8
  ret ptr addrspace(11) %v
}

; Must not crash on APInt bitwidth mismatch after addrspacecast to 32-bit AS.
define i32 @test_as_index_width_mismatch(ptr noalias %p) {
entry:
  store i32 1, ptr %p, align 4
  %q = addrspacecast ptr %p to ptr addrspace(3)
  %g = getelementptr i8, ptr addrspace(3) %q, i32 0
  %v = load i32, ptr addrspace(3) %g, align 4
  ret i32 %v
}

; Must not crash on scalable vector types.
define <vscale x 4 x float> @test_scalable(ptr noalias %p, <vscale x 4 x float> %x) {
entry:
  store <vscale x 4 x float> %x, ptr %p, align 16
  %v = load <vscale x 4 x float>, ptr %p, align 16
  ret <vscale x 4 x float> %v
}

; CHECK-LABEL: define ptr addrspace(1) @test_cross_as_ptr_diff_size(ptr noalias %p, ptr %x)
; CHECK-NEXT: entry:
; CHECK-NEXT:   store ptr %x, ptr %p, align 8
; CHECK-NEXT:   %v = load ptr addrspace(1), ptr %p, align 4
; CHECK-NEXT:   ret ptr addrspace(1) %v

; CHECK-LABEL: define ptr @test_cross_as_ptr_same_size(ptr noalias %p, ptr addrspace(2) %x)
; CHECK-NEXT: entry:
; CHECK-NEXT:   store ptr addrspace(2) %x, ptr %p, align 8
; CHECK-NEXT:   %v = load ptr, ptr %p, align 8
; CHECK-NEXT:   ret ptr %v

; CHECK-LABEL: define ptr addrspace(11) @test_non_integral_as11(ptr noalias %p, i64 %x)
; CHECK-NEXT: entry:
; CHECK-NEXT:   store i64 %x, ptr %p, align 8
; CHECK-NEXT:   %v = load ptr addrspace(11), ptr %p, align 8
; CHECK-NEXT:   ret ptr addrspace(11) %v

; CHECK-LABEL: define i32 @test_as_index_width_mismatch(ptr noalias %p)
; CHECK-NEXT: entry:
; CHECK-NEXT:   store i32 1, ptr %p, align 4
; CHECK-NEXT:   %q = addrspacecast ptr %p to ptr addrspace(3)
; CHECK-NEXT:   %g = getelementptr i8, ptr addrspace(3) %q, i32 0
; CHECK-NEXT:   %v = load i32, ptr addrspace(3) %g, align 4
; CHECK-NEXT:   ret i32 %v

; CHECK-LABEL: define <vscale x 4 x float> @test_scalable(ptr noalias %p, <vscale x 4 x float> %x)
; CHECK-NEXT: entry:
; CHECK-NEXT:   store <vscale x 4 x float> %x, ptr %p, align 16
; CHECK-NEXT:   %v = load <vscale x 4 x float>, ptr %p, align 16
; CHECK-NEXT:   ret <vscale x 4 x float> %v
