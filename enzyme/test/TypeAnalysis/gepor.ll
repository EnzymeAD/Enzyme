; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=callee -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=callee -S -o /dev/null | FileCheck %s

; Strength reduction turns `4 * i + 2` into `(i << 2) | 2`, so the getelementptr
; index must still be recognized as the constant offset 16 for the double type
; of the final load to flow back into the swapped i64 slots.

define double @callee(ptr %d) {
entry:
  %i = or i64 0, 2
  %a = getelementptr [8 x i8], ptr %d, i64 %i
  %b = getelementptr [8 x i8], ptr %d, i64 4
  %va = load i64, ptr %a, align 8
  %vb = load i64, ptr %b, align 8
  store i64 %vb, ptr %a, align 8
  store i64 %va, ptr %b, align 8
  %p = getelementptr i8, ptr %d, i64 16
  %x = load double, ptr %p, align 8
  ret double %x
}

; CHECK: callee - {[-1]:Float@double} |{[-1]:Pointer}:{}
; CHECK-NEXT: ptr %d: {[-1]:Pointer, [-1,16]:Float@double, [-1,32]:Float@double}
; CHECK-NEXT: entry
; CHECK-NEXT:   %i = or i64 0, 2: {[-1]:Anything}
; CHECK-NEXT:   %a = getelementptr [8 x i8], ptr %d, i64 %i: {[-1]:Pointer, [-1,0]:Float@double, [-1,16]:Float@double}
; CHECK-NEXT:   %b = getelementptr [8 x i8], ptr %d, i64 4: {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT:   %va = load i64, ptr %a, align 8: {[-1]:Float@double}
; CHECK-NEXT:   %vb = load i64, ptr %b, align 8: {[-1]:Float@double}
; CHECK-NEXT:   store i64 %vb, ptr %a, align 8: {}
; CHECK-NEXT:   store i64 %va, ptr %b, align 8: {}
; CHECK-NEXT:   %p = getelementptr i8, ptr %d, i64 16: {[-1]:Pointer, [-1,0]:Float@double, [-1,16]:Float@double}
; CHECK-NEXT:   %x = load double, ptr %p, align 8: {[-1]:Float@double}
; CHECK-NEXT:   ret double %x: {}
