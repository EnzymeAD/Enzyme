; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -simple-gvn -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="simple-gvn" -S | FileCheck %s

; Test that optimization is applied even without nocapture attribute
; (since we verify no capturing uses inline)

define i32 @test_no_nocapture(ptr noalias %ptr) {
entry:
  store i32 42, ptr %ptr, align 4
  %val = load i32, ptr %ptr, align 4
  ret i32 %val
}

; CHECK: define i32 @test_no_nocapture(ptr noalias %ptr)
; CHECK-NEXT: entry:
; CHECK-NEXT:   store i32 42, ptr %ptr, align 4
; CHECK-NEXT:   ret i32 42
