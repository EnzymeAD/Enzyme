; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=f -S -o /dev/null | FileCheck %s

@g = internal global [0 x double] zeroinitializer

define double @f(double %x) {
entry:
  %p = getelementptr [0 x double], ptr @g, i64 0, i64 0
  store double %x, ptr %p, align 8
  %v = load double, ptr %p, align 8
  ret double %v
}

; CHECK: f - {[-1]:Float@double} |{[-1]:Float@double}:{}
; CHECK-NEXT: double %x: {[-1]:Float@double}
; CHECK-NEXT: entry
; CHECK-NEXT:   %p = getelementptr [0 x double], ptr @g, i64 0, i64 0: {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT:   store double %x, ptr %p, align 8: {}
; CHECK-NEXT:   %v = load double, ptr %p, align 8: {[-1]:Float@double}
; CHECK-NEXT:   ret double %v: {}
