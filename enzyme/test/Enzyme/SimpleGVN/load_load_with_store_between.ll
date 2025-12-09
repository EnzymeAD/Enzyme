; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -opaque-pointers -simple-gvn -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -opaque-pointers -passes="simple-gvn" -S | FileCheck %s

; Test that load-to-load forwarding does not happen when there's a store between loads

define i32 @test_load_load_with_store_between(ptr noalias nocapture %ptr) {
entry:
  %val1 = load i32, ptr %ptr, align 4
  store i32 100, ptr %ptr, align 4
  %val2 = load i32, ptr %ptr, align 4
  %sum = add i32 %val1, %val2
  ret i32 %sum
}

; The first load should not be forwarded to the third load (val2)
; But the store should be forwarded to val2
; CHECK: define i32 @test_load_load_with_store_between(ptr noalias nocapture %ptr)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %val1 = load i32, ptr %ptr, align 4
; CHECK-NEXT:   store i32 100, ptr %ptr, align 4
; CHECK-NEXT:   %sum = add i32 %val1, 100
; CHECK-NEXT:   ret i32 %sum
