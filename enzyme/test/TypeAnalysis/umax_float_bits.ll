; RUN: if [ %llvmver -gt 11 ]; then %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=f -S -o /dev/null | FileCheck %s; fi

; Integer min/max of the bits of floats propagate the float type both ways.

define double @f(double %x, double %y, ptr %p, ptr %q, ptr %r) {
entry:
  %xi = bitcast double %x to i64
  %yi = bitcast double %y to i64
  %m = call i64 @llvm.umax.i64(i64 %xi, i64 %yi)
  store i64 %m, ptr %p, align 8
  %a = load i64, ptr %q, align 8
  %b = load i64, ptr %r, align 8
  %n = call i64 @llvm.smin.i64(i64 %a, i64 %b)
  %d = bitcast i64 %n to double
  ret double %d
}

declare i64 @llvm.umax.i64(i64, i64)
declare i64 @llvm.smin.i64(i64, i64)

; CHECK: f - {[-1]:Float@double} |{[-1]:Float@double}:{} {[-1]:Float@double}:{} {[-1]:Pointer, [-1,0]:Float@double}:{} {[-1]:Pointer, [-1,0]:Float@double}:{} {[-1]:Pointer, [-1,0]:Float@double}:{}
; CHECK-NEXT: double %x: {[-1]:Float@double}
; CHECK-NEXT: double %y: {[-1]:Float@double}
; CHECK-NEXT: ptr %p: {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT: ptr %q: {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT: ptr %r: {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT: entry
; CHECK-NEXT:   %xi = bitcast double %x to i64: {[-1]:Float@double}
; CHECK-NEXT:   %yi = bitcast double %y to i64: {[-1]:Float@double}
; CHECK-NEXT:   %m = call i64 @llvm.umax.i64(i64 %xi, i64 %yi): {[-1]:Float@double}
; CHECK-NEXT:   store i64 %m, ptr %p, align 8: {}
; CHECK-NEXT:   %a = load i64, ptr %q, align 8: {[-1]:Float@double}
; CHECK-NEXT:   %b = load i64, ptr %r, align 8: {[-1]:Float@double}
; CHECK-NEXT:   %n = call i64 @llvm.smin.i64(i64 %a, i64 %b): {[-1]:Float@double}
; CHECK-NEXT:   %d = bitcast i64 %n to double: {[-1]:Float@double}
; CHECK-NEXT:   ret double %d: {}
