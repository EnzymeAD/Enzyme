; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -opaque-pointers -simple-gvn -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -opaque-pointers -passes="simple-gvn" -S | FileCheck %s

; Test load-to-load forwarding with nocapture function call between loads
; The nocapture call should prevent forwarding

declare void @nocapture_func(ptr nocapture)

define i32 @test_load_load_nocapture_call(ptr noalias nocapture %ptr) {
entry:
  %val1 = load i32, ptr %ptr, align 4
  call void @nocapture_func(ptr nocapture %ptr)
  %val2 = load i32, ptr %ptr, align 4
  %sum = add i32 %val1, %val2
  ret i32 %sum
}

; CHECK: define i32 @test_load_load_nocapture_call(ptr noalias nocapture %ptr)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %val1 = load i32, ptr %ptr, align 4
; CHECK-NEXT:   call void @nocapture_func(ptr nocapture %ptr)
; CHECK-NEXT:   %val2 = load i32, ptr %ptr, align 4
; CHECK-NEXT:   %sum = add i32 %val1, %val2
; CHECK-NEXT:   ret i32 %sum
