; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -opaque-pointers -simple-gvn -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -opaque-pointers -passes="simple-gvn" -S | FileCheck %s

; Test that optimization is NOT applied when GEP has non-constant offset

define i32 @test_non_constant_gep(ptr noalias %ptr, i64 %idx) {
entry:
  %gep = getelementptr i32, ptr %ptr, i64 %idx
  store i32 42, ptr %gep, align 4
  %val = load i32, ptr %gep, align 4
  ret i32 %val
}

; CHECK: define i32 @test_non_constant_gep(ptr noalias %ptr, i64 %idx)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %gep = getelementptr i32, ptr %ptr, i64 %idx
; CHECK-NEXT:   store i32 42, ptr %gep, align 4
; CHECK-NEXT:   %val = load i32, ptr %gep, align 4
; CHECK-NEXT:   ret i32 %val
