; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -opaque-pointers -simple-gvn -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -opaque-pointers -passes="simple-gvn" -S | FileCheck %s

; Test that optimization IS applied even when argument doesn't have noalias
; if there are no intervening memory-modifying instructions.

define i32 @test_no_noalias(ptr nocapture %ptr) {
entry:
  store i32 42, ptr %ptr, align 4
  %val = load i32, ptr %ptr, align 4
  ret i32 %val
}

; CHECK: define i32 @test_no_noalias(ptr nocapture %ptr)
; CHECK-NEXT: entry:
; CHECK-NEXT:   store i32 42, ptr %ptr, align 4
; CHECK-NEXT:   ret i32 42

declare void @some_func()

define i32 @test_no_noalias_intervening_call(ptr %ptr) {
entry:
  store i32 42, ptr %ptr, align 4
  call void @some_func()
  %val = load i32, ptr %ptr, align 4
  ret i32 %val
}

; CHECK: define i32 @test_no_noalias_intervening_call(ptr %ptr)
; CHECK-NEXT: entry:
; CHECK-NEXT:   store i32 42, ptr %ptr, align 4
; CHECK-NEXT:   call void @some_func()
; CHECK-NEXT:   %val = load i32, ptr %ptr, align 4
; CHECK-NEXT:   ret i32 %val

define i32 @test_noalias_intervening_call(ptr noalias %ptr) {
entry:
  store i32 42, ptr %ptr, align 4
  call void @some_func()
  %val = load i32, ptr %ptr, align 4
  ret i32 %val
}

; CHECK: define i32 @test_noalias_intervening_call(ptr noalias %ptr)
; CHECK-NEXT: entry:
; CHECK-NEXT:   store i32 42, ptr %ptr, align 4
; CHECK-NEXT:   call void @some_func()
; CHECK-NEXT:   ret i32 42


