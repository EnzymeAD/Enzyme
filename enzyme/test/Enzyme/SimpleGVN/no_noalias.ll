; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -opaque-pointers -simple-gvn -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -opaque-pointers -passes="simple-gvn" -S | FileCheck %s

; Test that optimization IS applied even when argument doesn't have noalias
; if there are no intervening memory-modifying instructions.

define i32 @test_no_noalias(ptr nocapture %ptr) {
entry:
  store i32 42, ptr %ptr, align 4
  %val = load i32, ptr %ptr, align 4
  ret i32 %val
}

; CHECK: define i32 @test_no_noalias(ptr nocapture %ptr)
; CHECK-NEXT: entry:
; CHECK-NEXT:   store i32 42, ptr %ptr, align 4
; CHECK-NEXT:   ret i32 42

