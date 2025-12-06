; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -simplify-gvn -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="simplify-gvn" -S | FileCheck %s

; Test basic store-to-load forwarding with noalias nocapture argument

define i32 @test_basic(i32* noalias nocapture %ptr) {
entry:
  store i32 42, i32* %ptr, align 4
  %val = load i32, i32* %ptr, align 4
  ret i32 %val
}

; CHECK: define i32 @test_basic(i32* noalias nocapture %ptr)
; CHECK-NEXT: entry:
; CHECK-NEXT:   store i32 42, i32* %ptr, align 4
; CHECK-NEXT:   ret i32 42
