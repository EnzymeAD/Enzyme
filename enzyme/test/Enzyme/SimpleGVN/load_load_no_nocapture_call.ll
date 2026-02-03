; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -opaque-pointers -simple-gvn -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -opaque-pointers -passes="simple-gvn" -S | FileCheck %s

; Test that optimization is NOT applied when call does not have nocapture attribute
; and that even with nocapture, we don't forward if there's an intermediate call

declare void @external_func(ptr)

define i32 @test_load_load_no_nocapture_call(ptr noalias nocapture %ptr) {
entry:
  call void @external_func(ptr %ptr)
  %val1 = load i32, ptr %ptr, align 4
  %val2 = load i32, ptr %ptr, align 4
  %sum = add i32 %val1, %val2
  ret i32 %sum
}

; CHECK: define i32 @test_load_load_no_nocapture_call(ptr noalias nocapture %ptr)
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @external_func(ptr %ptr)
; CHECK-NEXT:   %val1 = load i32, ptr %ptr, align 4
; CHECK-NEXT:   %val2 = load i32, ptr %ptr, align 4
; CHECK-NEXT:   %sum = add i32 %val1, %val2
; CHECK-NEXT:   ret i32 %sum
