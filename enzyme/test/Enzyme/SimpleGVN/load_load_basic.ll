; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -opaque-pointers -simple-gvn -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -opaque-pointers -passes="simple-gvn" -S | FileCheck %s

; Test basic load-to-load forwarding with noalias nocapture argument

define i32 @test_load_load_basic(ptr noalias nocapture %ptr) {
entry:
  %val1 = load i32, ptr %ptr, align 4
  %val2 = load i32, ptr %ptr, align 4
  %sum = add i32 %val1, %val2
  ret i32 %sum
}

; CHECK: define i32 @test_load_load_basic(ptr noalias nocapture %ptr)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %val1 = load i32, ptr %ptr, align 4
; CHECK-NEXT:   %sum = add i32 %val1, %val1
; CHECK-NEXT:   ret i32 %sum
