; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -opaque-pointers -simple-gvn -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -opaque-pointers -passes="simple-gvn" -S | FileCheck %s

; Test store-to-load forwarding with byte offset extraction

define i8 @test_byte_extraction(ptr noalias nocapture %ptr) {
entry:
  store i64 72623859790382856, ptr %ptr, align 8
  %gep = getelementptr i8, ptr %ptr, i64 1
  %val = load i8, ptr %gep, align 1
  ret i8 %val
}

; CHECK: define i8 @test_byte_extraction(ptr noalias nocapture %ptr)
; CHECK-NEXT: entry:
; CHECK-NEXT:   store i64 72623859790382856, ptr %ptr, align 8
; CHECK-NEXT:   %gep = getelementptr i8, ptr %ptr, i64 1
; CHECK-NEXT:   ret i8 7
