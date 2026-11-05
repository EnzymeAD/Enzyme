; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=foo -S | FileCheck %s

declare i32 @use_int(i32) readnone nounwind

define i32 @foo() {
entry:
  %slot = alloca i32, align 4
  store i32 42, ptr %slot, align 4
  %old = atomicrmw add ptr %slot, i32 1 acq_rel
  %res = call i32 @use_int(i32 %old)
  ret i32 %res
}

; CHECK: entry
; CHECK-NEXT:   %slot = alloca i32, align 4: icv:1 ici:1
; CHECK-NEXT:   store i32 42, ptr %slot, align 4: icv:1 ici:1
; CHECK-NEXT:   %old = atomicrmw add ptr %slot, i32 1 acq_rel, align 4: icv:1 ici:1
; CHECK-NEXT:   %res = call i32 @use_int(i32 %old): icv:1 ici:1
; CHECK-NEXT:   ret i32 %res: icv:1 ici:1
