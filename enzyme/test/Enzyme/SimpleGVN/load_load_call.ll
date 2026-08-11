; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -opaque-pointers -simple-gvn -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -opaque-pointers -passes="simple-gvn" -S | FileCheck %s

; Test load-to-load forwarding of a pointer returned by a call.
; The call is not noalias, but the intervening store is to a noalias argument.

define ptr @test_load_load_call(ptr noalias %roots) {
entry:
  %ptr = call ptr @get_ptr()
  %val1 = load ptr, ptr %ptr, align 8
  
  ; Intervening store to a noalias argument
  store ptr %val1, ptr %roots, align 8
  
  %val2 = load ptr, ptr %ptr, align 8
  ret ptr %val2
}

declare ptr @get_ptr()

; CHECK: define ptr @test_load_load_call(ptr noalias %roots)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %ptr = call ptr @get_ptr()
; CHECK-NEXT:   %val1 = load ptr, ptr %ptr, align 8
; CHECK-NEXT:   store ptr %val1, ptr %roots, align 8
; CHECK-NEXT:   ret ptr %val1

define ptr @test_load_load_call_intervening(ptr %roots) {
entry:
  %ptr = call ptr @get_ptr()
  %val1 = load ptr, ptr %ptr, align 8
  
  ; Intervening store to the pointer itself
  store ptr null, ptr %ptr, align 8
  
  %val2 = load ptr, ptr %ptr, align 8
  ret ptr %val2
}

; CHECK: define ptr @test_load_load_call_intervening(ptr %roots)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %ptr = call ptr @get_ptr()
; CHECK-NEXT:   %val1 = load ptr, ptr %ptr, align 8
; CHECK-NEXT:   store ptr null, ptr %ptr, align 8
; CHECK-NEXT:   ret ptr null

define ptr @test_load_load_call_intervening_alias(ptr %ptr2) {
entry:
  %ptr = call ptr @get_ptr()
  %val1 = load ptr, ptr %ptr, align 8
  
  ; Intervening store to a potentially aliasing pointer
  store ptr null, ptr %ptr2, align 8
  
  %val2 = load ptr, ptr %ptr, align 8
  ret ptr %val2
}

; CHECK: define ptr @test_load_load_call_intervening_alias(ptr %ptr2)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %ptr = call ptr @get_ptr()
; CHECK-NEXT:   %val1 = load ptr, ptr %ptr, align 8
; CHECK-NEXT:   store ptr null, ptr %ptr2, align 8
; CHECK-NEXT:   %val2 = load ptr, ptr %ptr, align 8
; CHECK-NEXT:   ret ptr %val2


