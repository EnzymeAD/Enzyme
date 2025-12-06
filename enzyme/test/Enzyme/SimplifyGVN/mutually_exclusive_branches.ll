; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -simplify-gvn -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="simplify-gvn" -S | FileCheck %s

; Test that load IS forwarded when store2 is in mutually exclusive branch

define i32 @test_mutually_exclusive_branches(ptr noalias %ptr, i1 %cond) {
entry:
  store i32 42, ptr %ptr, align 4
  br i1 %cond, label %then, label %else

then:
  %val = load i32, ptr %ptr, align 4
  ret i32 %val

else:
  store i32 99, ptr %ptr, align 4
  ret i32 99
}

; CHECK: define i32 @test_mutually_exclusive_branches(ptr noalias %ptr, i1 %cond)
; CHECK: entry:
; CHECK-NEXT:   store i32 42, ptr %ptr, align 4
; CHECK-NEXT:   br i1 %cond, label %then, label %else
; CHECK: then:
; CHECK-NEXT:   ret i32 42
; CHECK: else:
; CHECK-NEXT:   store i32 99, ptr %ptr, align 4
; CHECK-NEXT:   ret i32 99
