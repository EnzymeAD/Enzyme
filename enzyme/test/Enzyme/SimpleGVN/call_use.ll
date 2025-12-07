; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -simple-gvn -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="simple-gvn" -S | FileCheck %s

; Test that optimization is NOT applied when argument has non-load/store uses

declare void @external_func(ptr)

define i32 @test_call_use(ptr noalias nocapture %ptr) {
entry:
  store i32 42, ptr %ptr, align 4
  call void @external_func(ptr %ptr)
  %val = load i32, ptr %ptr, align 4
  ret i32 %val
}

; CHECK: define i32 @test_call_use(ptr noalias nocapture %ptr)
; CHECK-NEXT: entry:
; CHECK-NEXT:   store i32 42, ptr %ptr, align 4
; CHECK-NEXT:   call void @external_func(ptr %ptr)
; CHECK-NEXT:   %val = load i32, ptr %ptr, align 4
; CHECK-NEXT:   ret i32 %val
