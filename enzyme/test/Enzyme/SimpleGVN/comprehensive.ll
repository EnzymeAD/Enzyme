; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -opaque-pointers -simple-gvn -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -opaque-pointers -passes="simple-gvn" -S | FileCheck %s

; Test comprehensive example with multiple stores and loads at different offsets

define i32 @test_comprehensive(ptr noalias nocapture %ptr) {
entry:
  ; Store values at different offsets
  %gep0 = getelementptr i32, ptr %ptr, i64 0
  %gep1 = getelementptr i32, ptr %ptr, i64 1
  %gep2 = getelementptr i32, ptr %ptr, i64 2
  
  store i32 10, ptr %gep0, align 4
  store i32 20, ptr %gep1, align 4
  store i32 30, ptr %gep2, align 4
  
  ; Load values back - should be forwarded
  %val0 = load i32, ptr %gep0, align 4
  %val1 = load i32, ptr %gep1, align 4
  %val2 = load i32, ptr %gep2, align 4
  
  ; Compute result
  %sum1 = add i32 %val0, %val1
  %sum2 = add i32 %sum1, %val2
  
  ret i32 %sum2
}

; CHECK: define i32 @test_comprehensive(ptr noalias nocapture %ptr)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %gep0 = getelementptr i32, ptr %ptr, i64 0
; CHECK-NEXT:   %gep1 = getelementptr i32, ptr %ptr, i64 1
; CHECK-NEXT:   %gep2 = getelementptr i32, ptr %ptr, i64 2
; CHECK-NEXT:   store i32 10, ptr %gep0, align 4
; CHECK-NEXT:   store i32 20, ptr %gep1, align 4
; CHECK-NEXT:   store i32 30, ptr %gep2, align 4
; CHECK-NEXT:   %sum1 = add i32 10, 20
; CHECK-NEXT:   %sum2 = add i32 %sum1, 30
; CHECK-NEXT:   ret i32 %sum2
