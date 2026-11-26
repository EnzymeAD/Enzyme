; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-type-analysis" -type-analysis-func=callee -S -o /dev/null | FileCheck %s

; x(i) of a 1-based array, i = 1..n, after LICM hoisted the step back by one
; element out of the loop: a GEP of minus one element is the base of the
; scaled GEP of the index. The types of the elements reach the base pointers.
define void @callee(ptr %x, ptr %t, i64 %n) {
entry:
  %x.lo = getelementptr i8, ptr %x, i64 -4
  %t.lo = getelementptr i8, ptr %t, i64 -4
  br label %loop

loop:
  %i = phi i64 [ 1, %entry ], [ %i.next, %loop ]
  %xi = getelementptr float, ptr %x.lo, i64 %i
  %v = load float, ptr %xi, align 4
  %w = fmul float %v, %v
  %ti = getelementptr float, ptr %t.lo, i64 %i
  store float %w, ptr %ti, align 4
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; CHECK: callee - {} |{[-1]:Pointer}:{} {[-1]:Pointer}:{} {[-1]:Integer}:{}
; CHECK-NEXT: ptr %x: {[-1]:Pointer, [-1,0]:Float@float}
; CHECK-NEXT: ptr %t: {[-1]:Pointer, [-1,0]:Float@float}
; CHECK-NEXT: i64 %n: {[-1]:Integer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %x.lo = getelementptr i8, ptr %x, i64 -4: {[-1]:Pointer, [-1,4]:Float@float}
; CHECK-NEXT:   %t.lo = getelementptr i8, ptr %t, i64 -4: {[-1]:Pointer, [-1,4]:Float@float}
