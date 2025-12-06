; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -simplify-gvn -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="simplify-gvn" -S | FileCheck %s

; Test that load is not forwarded when store does not dominate load

define i32 @test_no_dominance(i32* noalias nocapture %ptr, i1 %cond) {
entry:
  br i1 %cond, label %then, label %else

then:
  store i32 42, i32* %ptr, align 4
  br label %merge

else:
  br label %merge

merge:
  %val = load i32, i32* %ptr, align 4
  ret i32 %val
}

; CHECK: define i32 @test_no_dominance(i32* noalias nocapture %ptr, i1 %cond)
; CHECK: merge:
; CHECK-NEXT:   %val = load i32, i32* %ptr, align 4
; CHECK-NEXT:   ret i32 %val
