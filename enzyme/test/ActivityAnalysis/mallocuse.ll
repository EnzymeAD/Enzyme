; RUN: %opt < %s %newLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=_take -opaque-pointers -S -o /dev/null | FileCheck %s

declare ptr @malloc(i64)

define double @_take(ptr %a0, i1 %a1) {
entry:
  %a3 = tail call ptr @malloc(i64 10)
  %a4 = tail call ptr @malloc(i64 10)
  %a5 = ptrtoint ptr %a4 to i64
  %a6 = or i64 %a5, 1
  %a7 = inttoptr i64 %a6 to ptr
  %a8 = load double, ptr %a7, align 8
  store double %a8, ptr %a0, align 8
  br i1 %a1, label %.lr.ph, label %.lr.ph1.peel.next

.lr.ph1.peel.next:                                ; preds = %2
  %.pre = load double, ptr %a4, align 8
  ret double %.pre

.lr.ph:                                           ; preds = %.lr.ph, %2
  %a9 = load double, ptr %a3, align 4
  store double %a9, ptr %a4, align 8
  br label %.lr.ph
}

; CHECK: ptr %a0: icv:0
; CHECK-NEXT: i1 %a1: icv:1
; CHECK-NEXT: entry
; CHECK-NEXT:   %a3 = tail call ptr @malloc(i64 10): icv:1 ici:1
; CHECK-NEXT:   %a4 = tail call ptr @malloc(i64 10): icv:1 ici:1
; CHECK-NEXT:   %a5 = ptrtoint ptr %a4 to i64: icv:1 ici:1
; CHECK-NEXT:   %a6 = or i64 %a5, 1: icv:1 ici:1
; CHECK-NEXT:   %a7 = inttoptr i64 %a6 to ptr: icv:1 ici:1
; CHECK-NEXT:   %a8 = load double, ptr %a7, align 8: icv:1 ici:1
; CHECK-NEXT:   store double %a8, ptr %a0, align 8: icv:1 ici:1
; CHECK-NEXT:   br i1 %a1, label %.lr.ph, label %.lr.ph1.peel.next: icv:1 ici:1
; CHECK-NEXT: .lr.ph1.peel.next
; CHECK-NEXT:   %.pre = load double, ptr %a4, align 8: icv:1 ici:1
; CHECK-NEXT:   ret double %.pre: icv:1 ici:1
; CHECK-NEXT: .lr.ph
; CHECK-NEXT:   %a9 = load double, ptr %a3, align 4: icv:1 ici:1
; CHECK-NEXT:   store double %a9, ptr %a4, align 8: icv:1 ici:1
; CHECK-NEXT:   br label %.lr.ph: icv:1 ici:1
