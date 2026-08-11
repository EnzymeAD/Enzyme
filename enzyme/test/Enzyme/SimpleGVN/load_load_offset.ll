; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -opaque-pointers -simple-gvn -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -opaque-pointers -passes="simple-gvn" -S | FileCheck %s

; Test load-to-load forwarding with GEP offsets

define i32 @test_load_load_offset(ptr noalias nocapture %ptr) {
entry:
  %gep = getelementptr i32, ptr %ptr, i64 1
  %val1 = load i32, ptr %gep, align 4
  %val2 = load i32, ptr %gep, align 4
  %sum = add i32 %val1, %val2
  ret i32 %sum
}

; CHECK: define i32 @test_load_load_offset(ptr noalias nocapture %ptr)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %gep = getelementptr i32, ptr %ptr, i64 1
; CHECK-NEXT:   %val1 = load i32, ptr %gep, align 4
; CHECK-NEXT:   %sum = add i32 %val1, %val1
; CHECK-NEXT:   ret i32 %sum
