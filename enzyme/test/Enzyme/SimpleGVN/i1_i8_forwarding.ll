; RUN: %opt < %s %newLoadEnzyme -passes="simple-gvn" -S | FileCheck %s

define i8 @test_i1_i8_forwarding(i8* noalias nocapture %ptr, i1 %val) {
entry:
  %ptr_i1 = bitcast i8* %ptr to i1*
  store i1 %val, i1* %ptr_i1, align 1
  %v = load i8, i8* %ptr, align 1
  ret i8 %v
}

; CHECK: define i8 @test_i1_i8_forwarding(i8* noalias nocapture %ptr, i1 %val)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %ptr_i1 = bitcast i8* %ptr to i1*
; CHECK-NEXT:   store i1 %val, i1* %ptr_i1, align 1
; CHECK-NEXT:   %0 = zext i1 %val to i8
; CHECK-NEXT:   ret i8 %0
