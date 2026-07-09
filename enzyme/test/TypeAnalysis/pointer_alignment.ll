; RUN: %opt < %s %newLoadEnzyme -opaque-pointers -passes="print-type-analysis" -type-analysis-func=foo -S -o /dev/null | FileCheck %s

source_filename = "pointer_alignment.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define float @foo(ptr %x, i64 %n) {
entry:
  %buf = alloca [128 x i8], align 16
  %intptr = ptrtoint ptr %buf to i64
  %add = add i64 %intptr, 63
  %and = and i64 %add, -64
  %aligned = inttoptr i64 %and to ptr
  %check = icmp eq i64 %and, %n
  br i1 %check, label %then, label %else

then:
  %val = load float, ptr %x, align 4
  store float %val, ptr %aligned, align 4
  %out = load float, ptr %aligned, align 4
  ret float %out

else:
  ret float 0.0
}

; CHECK: foo - {} |{[-1]:Pointer, [-1,0]:Float@float}:{} {[-1]:Integer}:{}
; CHECK-NEXT: ptr %x: {[-1]:Pointer, [-1,0]:Float@float}
; CHECK-NEXT: i64 %n: {[-1]:Integer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %buf = alloca [128 x i8], align 16: {[-1]:Pointer, [-1,0]:Float@float}
; CHECK-NEXT:   %intptr = ptrtoint ptr %buf to i64: {[-1]:Pointer, [-1,0]:Float@float}
; CHECK-NEXT:   %add = add i64 %intptr, 63: {[-1]:Pointer, [-1,0]:Float@float}
; CHECK-NEXT:   %and = and i64 %add, -64: {[-1]:Pointer, [-1,0]:Float@float}
; CHECK-NEXT:   %aligned = inttoptr i64 %and to ptr: {[-1]:Pointer, [-1,0]:Float@float}
; CHECK-NEXT:   %check = icmp eq i64 %and, %n: {[-1]:Integer}
; CHECK-NEXT:   br i1 %check, label %then, label %else: {}
; CHECK: then
; CHECK-NEXT:   %val = load float, ptr %x, align 4: {[-1]:Float@float}
; CHECK-NEXT:   store float %val, ptr %aligned, align 4: {}
; CHECK-NEXT:   %out = load float, ptr %aligned, align 4: {[-1]:Float@float}
; CHECK-NEXT:   ret float %out: {}
; CHECK: else
; CHECK-NEXT:   ret float 0.0{{.*}}: {}
