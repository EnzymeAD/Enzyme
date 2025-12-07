; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -simple-gvn -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="simple-gvn" -S | FileCheck %s

; Test that load is NOT forwarded when there's a conditional store between

define i32 @test_conditional_store(ptr noalias %ptr, i1 %cond) {
entry:
  store i32 42, ptr %ptr, align 4
  br i1 %cond, label %then, label %merge

then:
  store i32 99, ptr %ptr, align 4
  br label %merge

merge:
  %val = load i32, ptr %ptr, align 4
  ret i32 %val
}

; CHECK: define i32 @test_conditional_store(ptr noalias %ptr, i1 %cond)
; CHECK: entry:
; CHECK-NEXT:   store i32 42, ptr %ptr, align 4
; CHECK-NEXT:   br i1 %cond, label %then, label %merge
; CHECK: then:
; CHECK-NEXT:   store i32 99, ptr %ptr, align 4
; CHECK-NEXT:   br label %merge
; CHECK: merge:
; CHECK-NEXT:   %val = load i32, ptr %ptr, align 4
; CHECK-NEXT:   ret i32 %val
