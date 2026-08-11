; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -simple-gvn -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="simple-gvn" -S | FileCheck %s

define i8* @test_byte_extraction(i8** noalias nocapture %ptr, float* %x) {
entry:
  %c = bitcast i8** %ptr to float**
  store float* %x, float** %c
  %val = load i8*, i8** %ptr
  ret i8* %val
}

; CHECK: define i8* @test_byte_extraction(i8** noalias nocapture %ptr, float* %x) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %c = bitcast i8** %ptr to float**
; CHECK-NEXT:   store float* %x, float** %c, align 8
; CHECK-NEXT:   %0 = bitcast float* %x to i8*
; CHECK-NEXT:   ret i8* %0
; CHECK-NEXT: }
