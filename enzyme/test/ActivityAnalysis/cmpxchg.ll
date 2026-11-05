; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=foo -S | FileCheck %s

define void @foo() {
entry:
  %lock = alloca i32, align 4
  store i32 0, ptr %lock, align 4
  %cx = cmpxchg ptr %lock, i32 0, i32 1 acquire monotonic
  %ok = extractvalue { i32, i1 } %cx, 1
  ret void
}

; CHECK: entry
; CHECK-NEXT:   %lock = alloca i32, align 4: icv:1 ici:1
; CHECK-NEXT:   store i32 0, ptr %lock, align 4: icv:1 ici:1
; CHECK-NEXT:   %cx = cmpxchg ptr %lock, i32 0, i32 1 acquire monotonic, align 4: icv:1 ici:1
; CHECK-NEXT:   %ok = extractvalue { i32, i1 } %cx, 1: icv:1 ici:1
; CHECK-NEXT:   ret void: icv:1 ici:1
