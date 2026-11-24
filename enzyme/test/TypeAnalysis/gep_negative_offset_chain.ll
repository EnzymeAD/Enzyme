; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-type-analysis" -type-analysis-func=callee -S -o /dev/null | FileCheck %s

; x(i) of a 1-based array, i = 1..n, as InstCombine canonicalizes &x[i - 1]:
; a scaled GEP of the index followed by a GEP of minus one element. The types
; of the elements reach the base pointers.
define void @callee(ptr %x, ptr %t, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 1, %entry ], [ %i.next, %loop ]
  %xi.hi = getelementptr [4 x i8], ptr %x, i64 %i
  %xi = getelementptr i8, ptr %xi.hi, i64 -4
  %v = load float, ptr %xi, align 4
  %w = fmul float %v, %v
  %ti.hi = getelementptr [4 x i8], ptr %t, i64 %i
  %ti = getelementptr i8, ptr %ti.hi, i64 -4
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
; CHECK: %xi.hi = getelementptr [4 x i8], ptr %x, i64 %i: {[-1]:Pointer}
; CHECK-NEXT: %xi = getelementptr i8, ptr %xi.hi, i64 -4: {[-1]:Pointer, [-1,0]:Float@float}
