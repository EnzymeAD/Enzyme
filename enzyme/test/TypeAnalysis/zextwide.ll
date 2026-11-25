; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="print-type-analysis" -type-analysis-func=caller -S -o /dev/null | FileCheck %s; fi

; The caller truncates the call's result, so four of its eight bytes are known
; to be integers, and the callee's zext has to hand its two-byte operand no
; more than the operand's own bytes.

define i64 @callee(ptr %p) {
entry:
  %l = load i16, ptr %p, align 2
  %z = zext i16 %l to i64
  ret i64 %z
}

define double @caller(ptr %p, ptr %q) {
entry:
  %c = call i64 @callee(ptr %p)
  %t = trunc i64 %c to i32
  %g = getelementptr inbounds double, ptr %q, i32 %t
  %v = load double, ptr %g, align 8
  ret double %v
}

; CHECK: callee - {[0]:Integer, [1]:Integer, [2]:Integer, [3]:Integer} |{[-1]:Pointer}:{}
; CHECK-NEXT: ptr %p: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %l = load i16, ptr %p, align 2: {[-1]:Integer}
; CHECK-NEXT:   %z = zext i16 %l to i64: {[0]:Integer, [1]:Integer, [2]:Integer, [3]:Integer}
; CHECK-NEXT:   ret i64 %z: {}

; CHECK: caller - {[-1]:Float@double} |{[-1]:Pointer}:{} {[-1]:Pointer}:{}
; CHECK-NEXT: ptr %p: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer}
; CHECK-NEXT: ptr %q: {[-1]:Pointer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %c = call i64 @callee(ptr %p): {[0]:Integer, [1]:Integer, [2]:Integer, [3]:Integer}
; CHECK-NEXT:   %t = trunc i64 %c to i32: {[-1]:Integer}
; CHECK-NEXT:   %g = getelementptr inbounds double, ptr %q, i32 %t: {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT:   %v = load double, ptr %g, align 8: {[-1]:Float@double}
; CHECK-NEXT:   ret double %v: {}
