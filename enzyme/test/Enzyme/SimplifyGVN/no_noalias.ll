; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -simplify-gvn -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="simplify-gvn" -S | FileCheck %s

; Test that optimization is NOT applied when argument doesn't have noalias

define i32 @test_no_noalias(i32* nocapture %ptr) {
entry:
  store i32 42, i32* %ptr, align 4
  %val = load i32, i32* %ptr, align 4
  ret i32 %val
}

; CHECK: define i32 @test_no_noalias(i32* nocapture %ptr)
; CHECK-NEXT: entry:
; CHECK-NEXT:   store i32 42, i32* %ptr, align 4
; CHECK-NEXT:   %val = load i32, i32* %ptr, align 4
; CHECK-NEXT:   ret i32 %val
