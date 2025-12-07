; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -simple-gvn -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="simple-gvn" -S | FileCheck %s

; Test that optimization is NOT applied when the pointer value is stored (not stored to)

define i32 @test_store_pointer_value(ptr noalias nocapture %ptr, ptr %out) {
entry:
  store i32 42, ptr %ptr, align 4
  ; Store the pointer value itself to another location
  store ptr %ptr, ptr %out, align 8
  %val = load i32, ptr %ptr, align 4
  ret i32 %val
}

; CHECK: define i32 @test_store_pointer_value(ptr noalias nocapture %ptr, ptr %out)
; CHECK-NEXT: entry:
; CHECK-NEXT:   store i32 42, ptr %ptr, align 4
; CHECK-NEXT:   store ptr %ptr, ptr %out, align 8
; CHECK-NEXT:   %val = load i32, ptr %ptr, align 4
; CHECK-NEXT:   ret i32 %val
