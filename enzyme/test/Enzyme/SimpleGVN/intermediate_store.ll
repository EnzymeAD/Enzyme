; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -opaque-pointers -simple-gvn -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -opaque-pointers -passes="simple-gvn" -S | FileCheck %s

; Test that load is not forwarded when there's an intermediate store

define i32 @test_intermediate_store(ptr noalias nocapture %ptr) {
entry:
  store i32 42, ptr %ptr, align 4
  store i32 99, ptr %ptr, align 4
  %val = load i32, ptr %ptr, align 4
  ret i32 %val
}

; CHECK: define i32 @test_intermediate_store(ptr noalias nocapture %ptr)
; CHECK-NEXT: entry:
; CHECK-NEXT:   store i32 42, ptr %ptr, align 4
; CHECK-NEXT:   store i32 99, ptr %ptr, align 4
; CHECK-NEXT:   ret i32 99
