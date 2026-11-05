; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-type-analysis" -type-analysis-func=caller -S -o /dev/null | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @sink(i64)

define void @caller(ptr %unk_ptr, i32 %idx) {
entry:
  %unk = load i64, ptr %unk_ptr, align 8
  %known_int = zext i32 %idx to i64
  %prod = mul i64 %unk, %known_int
  call void @sink(i64 %prod)
  ret void
}

; CHECK: caller - {} |
; CHECK: entry
; CHECK-NEXT:   %unk = load i64, ptr %unk_ptr, align 8: {}
; CHECK-NEXT:   %known_int = zext i32 %idx to i64: {[-1]:Integer}
; CHECK-NEXT:   %prod = mul i64 %unk, %known_int: {[-1]:Integer}
; CHECK-NEXT:   call void @sink(i64 %prod): {}
; CHECK-NEXT:   ret void: {}
