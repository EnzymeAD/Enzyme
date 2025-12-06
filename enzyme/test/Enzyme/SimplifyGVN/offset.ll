; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -simplify-gvn -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="simplify-gvn" -S | FileCheck %s

; Test store-to-load forwarding with GEP offsets

define i32 @test_offset(ptr noalias nocapture %ptr) {
entry:
  %gep = getelementptr i32, ptr %ptr, i64 1
  store i32 123, ptr %gep, align 4
  %val = load i32, ptr %gep, align 4
  ret i32 %val
}

; CHECK: define i32 @test_offset(ptr noalias nocapture %ptr)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %gep = getelementptr i32, ptr %ptr, i64 1
; CHECK-NEXT:   store i32 123, ptr %gep, align 4
; CHECK-NEXT:   ret i32 123
