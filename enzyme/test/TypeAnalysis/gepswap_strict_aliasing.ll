; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -print-type-analysis -type-analysis-func=callee -enzyme-strict-aliasing=0 -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-type-analysis" -type-analysis-func=callee -enzyme-strict-aliasing=0 -S -o /dev/null | FileCheck %s

; A single-index byte GEP (`getelementptr i8, ptr %d, i64 %.idx`) whose initial
; induction value is 0 must not clip the pointee TypeTree to the 1-byte `i8`
; element size, so subsequent chained GEPs (`+16`, `+32`) still receive the
; double types from `%d` under `-enzyme-strict-aliasing=0`.

define void @callee(i64 %n, ptr %d) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %.idx = shl i64 %iv, 5
  %base = getelementptr i8, ptr %d, i64 %.idx
  %a = getelementptr inbounds i8, ptr %base, i64 16
  %b = getelementptr inbounds i8, ptr %base, i64 32
  %va = load i64, ptr %a, align 8
  %vb = load i64, ptr %b, align 8
  store i64 %vb, ptr %a, align 8
  store i64 %va, ptr %b, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %cond = icmp eq i64 %iv.next, %n
  br i1 %cond, label %exit, label %loop

exit:
  %p2 = getelementptr inbounds i8, ptr %d, i64 16
  %v2 = load double, ptr %p2, align 8
  %v0 = load double, ptr %d, align 8
  %add = fadd double %v0, %v2
  store double %add, ptr %d, align 8
  ret void
}

; CHECK: callee - {} |{[-1]:Integer}:{} {[-1]:Pointer}:{}
; CHECK-NEXT: i64 %n: {[-1]:Integer}
; CHECK-NEXT: ptr %d: {[-1]:Pointer, [-1,0]:Float@double, [-1,16]:Float@double, [-1,32]:Float@double}
; CHECK-NEXT: entry
; CHECK-NEXT:   br label %loop: {}
; CHECK-NEXT: loop
; CHECK-NEXT:   %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]: {[-1]:Integer}
; CHECK-NEXT:   %.idx = shl i64 %iv, 5: {[-1]:Integer}
; CHECK-NEXT:   %base = getelementptr i8, ptr %d, i64 %.idx: {[-1]:Pointer, [-1,0]:Float@double, [-1,16]:Float@double, [-1,32]:Float@double}
; CHECK-NEXT:   %a = getelementptr inbounds i8, ptr %base, i64 16: {[-1]:Pointer, [-1,0]:Float@double, [-1,16]:Float@double}
; CHECK-NEXT:   %b = getelementptr inbounds i8, ptr %base, i64 32: {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT:   %va = load i64, ptr %a, align 8: {[-1]:Float@double}
; CHECK-NEXT:   %vb = load i64, ptr %b, align 8: {[-1]:Float@double}
; CHECK-NEXT:   store i64 %vb, ptr %a, align 8: {}
; CHECK-NEXT:   store i64 %va, ptr %b, align 8: {}
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1: {[-1]:Integer}
; CHECK-NEXT:   %cond = icmp eq i64 %iv.next, %n: {[-1]:Integer}
; CHECK-NEXT:   br i1 %cond, label %exit, label %loop: {}
; CHECK-NEXT: exit
; CHECK-NEXT:   %p2 = getelementptr inbounds i8, ptr %d, i64 16: {[-1]:Pointer, [-1,0]:Float@double, [-1,16]:Float@double}
; CHECK-NEXT:   %v2 = load double, ptr %p2, align 8: {[-1]:Float@double}
; CHECK-NEXT:   %v0 = load double, ptr %d, align 8: {[-1]:Float@double}
; CHECK-NEXT:   %add = fadd double %v0, %v2: {[-1]:Float@double}
; CHECK-NEXT:   store double %add, ptr %d, align 8: {}
; CHECK-NEXT:   ret void: {}
