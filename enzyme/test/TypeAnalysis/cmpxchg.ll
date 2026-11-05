; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-type-analysis" -type-analysis-func=caller -S -o /dev/null | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @caller(ptr %lock) {
entry:
  %cx = cmpxchg ptr %lock, i32 0, i32 1 acquire monotonic
  %old = extractvalue { i32, i1 } %cx, 0
  %ok = extractvalue { i32, i1 } %cx, 1
  %sel = select i1 %ok, i32 %old, i32 0
  ret i32 %sel
}

; CHECK: caller - {[-1]:Integer} |
; CHECK-NEXT: ptr %lock: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %cx = cmpxchg ptr %lock, i32 0, i32 1 acquire monotonic, align 4: {[0]:Integer, [1]:Integer, [2]:Integer, [3]:Integer, [4]:Integer}
; CHECK-NEXT:   %old = extractvalue { i32, i1 } %cx, 0: {[-1]:Integer}
; CHECK-NEXT:   %ok = extractvalue { i32, i1 } %cx, 1: {[-1]:Integer}
