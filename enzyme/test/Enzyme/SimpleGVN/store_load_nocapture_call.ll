; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -opaque-pointers -simple-gvn -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -opaque-pointers -passes="simple-gvn" -S | FileCheck %s

; Test load-to-load forwarding with nocapture function call between loads
; The nocapture call should prevent forwarding

declare void @nocapture_func(ptr nocapture)

define i32 @test_load_load_nocapture_call(ptr noalias nocapture %ptr) {
entry:
  store i32 0, ptr %ptr, align 4
  br label %next

next:
  call void @nocapture_func(ptr nocapture %ptr)
  %val2 = load i32, ptr %ptr, align 4
  ret i32 %val2
}

; CHECK: define i32 @test_load_load_nocapture_call(ptr noalias nocapture %ptr)
; CHECK-NEXT: entry:
; CHECK-NEXT:   store i32 0, ptr %ptr, align 4
; CHECK-NEXT:   br label %next

; CHECK: next:                                             ; preds = %entry
; CHECK-NEXT:   call void @nocapture_func(ptr nocapture %ptr)
; CHECK-NEXT:   %val2 = load i32, ptr %ptr, align 4
; CHECK-NEXT:   ret i32 %val2
; CHECK-NEXT: }
