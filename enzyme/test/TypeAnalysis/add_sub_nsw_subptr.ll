; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-type-analysis" -type-analysis-func=caller -S -o /dev/null | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @caller(ptr %slot) {
entry:
  %v = load i32, ptr %slot, align 4
  %inc = add nsw i32 %v, 1
  store i32 %inc, ptr %slot, align 4
  ret void
}

; CHECK: caller - {} |
; CHECK-NEXT: ptr %slot: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %v = load i32, ptr %slot, align 4: {[-1]:Integer}
; CHECK-NEXT:   %inc = add nsw i32 %v, 1: {[-1]:Integer}
; CHECK-NEXT:   store i32 %inc, ptr %slot, align 4: {}
; CHECK-NEXT:   ret void: {}
