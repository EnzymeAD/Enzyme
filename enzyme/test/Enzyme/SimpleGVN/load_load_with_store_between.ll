; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -opaque-pointers -simple-gvn -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -opaque-pointers -passes="simple-gvn" -S | FileCheck %s

; Test that load-to-load forwarding does not happen when there's a store between loads
; But does happen after the store when there's no intervening write

define i32 @test_load_load_with_store_between(ptr noalias nocapture %ptr) {
entry:
  %val1 = load i32, ptr %ptr, align 4
  %val2 = load i32, ptr %ptr, align 4
  store i32 %val1, ptr %ptr, align 4
  %val3 = load i32, ptr %ptr, align 4
  %val4 = load i32, ptr %ptr, align 4
  %sum1 = add i32 %val1, %val2
  %sum2 = add i32 %sum1, %val3
  %sum3 = add i32 %sum2, %val4
  ret i32 %sum3
}

; val1 and val2 should be forwarded (load-load before store)
; val3 and val4 should both be forwarded from the store (store-load forwarding)
; CHECK: define i32 @test_load_load_with_store_between(ptr noalias nocapture %ptr)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %val1 = load i32, ptr %ptr, align 4
; CHECK-NEXT:   store i32 %val1, ptr %ptr, align 4
; CHECK-NEXT:   %sum1 = add i32 %val1, %val1
; CHECK-NEXT:   %sum2 = add i32 %sum1, %val1
; CHECK-NEXT:   %sum3 = add i32 %sum2, %val1
; CHECK-NEXT:   ret i32 %sum3
