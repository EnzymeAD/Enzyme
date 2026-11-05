; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=foo -S | FileCheck %s

declare void @read_and_escape(ptr readonly nocapture, ptr)

define void @foo(double %x) {
entry:
  %buf = alloca double, align 8
  store double %x, ptr %buf, align 8
  call void @read_and_escape(ptr %buf, ptr %buf)
  ret void
}

; CHECK: entry
; CHECK-NEXT:   %buf = alloca double, align 8: icv:0 ici:1
; CHECK-NEXT:   store double %x, ptr %buf, align 8: icv:1 ici:0
; CHECK-NEXT:   call void @read_and_escape(ptr %buf, ptr %buf): icv:1 ici:0
