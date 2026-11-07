; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -print-type-analysis -type-analysis-func=advance_cursor -enzyme-max-type-offset=16 -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-type-analysis" -type-analysis-func=advance_cursor -enzyme-max-type-offset=16 -S -o /dev/null | FileCheck %s

define void @advance_cursor(ptr %slot) {
entry:
  %cur = load ptr, ptr %slot, align 8
  %next = getelementptr inbounds ptr, ptr %cur, i64 1
  store ptr %next, ptr %slot, align 8
  %chunk = load ptr, ptr %cur, align 8
  store float 1.000000e+00, ptr %chunk, align 4
  ret void
}

; CHECK-LABEL: advance_cursor - {}
; CHECK-NEXT: ptr %slot: {[-1]:Pointer, [-1,0]:Pointer, [-1,0,0]:Pointer, [-1,0,0,0]:Float@float, [-1,0,8]:Pointer, [-1,0,8,0]:Float@float, [-1,0,16]:Pointer, [-1,0,16,0]:Float@float}
; CHECK-NEXT: entry
; CHECK-NEXT:   %cur = load ptr, ptr %slot, align 8: {[-1]:Pointer, [-1,0]:Pointer, [-1,0,0]:Float@float, [-1,8]:Pointer, [-1,8,0]:Float@float, [-1,16]:Pointer, [-1,16,0]:Float@float}
; CHECK-NEXT:   %next = getelementptr inbounds ptr, ptr %cur, i64 1: {[-1]:Pointer, [-1,0]:Pointer, [-1,0,0]:Float@float, [-1,8]:Pointer, [-1,8,0]:Float@float, [-1,16]:Pointer, [-1,16,0]:Float@float}
; CHECK-NEXT:   store ptr %next, ptr %slot, align 8: {}
; CHECK-NEXT:   %chunk = load ptr, ptr %cur, align 8: {[-1]:Pointer, [-1,0]:Float@float}
