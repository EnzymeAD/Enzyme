; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="print-type-analysis" -type-analysis-func=caller -S -o /dev/null | FileCheck %s; fi

; A phi of derived pointers split into a base phi and an offset phi that are
; correlated per incoming edge: %mem is a {i64 length, ptr data} object indexed
; at 8, %arr a {ptr data, ...} object indexed at 0. The pointee type of the gep
; must be propagated to each incoming object at its own offset; applying both
; offsets to both objects would claim the length of %mem is a pointer, which
; conflicts with its integer use.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define float @caller(ptr %mem, ptr %arr, i1 %c) {
entry:
  %len = load i64, ptr %mem, align 8
  %lenf = uitofp i64 %len to float
  br i1 %c, label %a, label %b

a:
  br label %merge

b:
  br label %merge

merge:
  %base = phi ptr [ %mem, %a ], [ %arr, %b ]
  %off = phi i64 [ 8, %a ], [ 0, %b ]
  %p = getelementptr i8, ptr %base, i64 %off
  %data = load ptr, ptr %p, align 8
  %v = load float, ptr %data, align 4
  %r = fadd float %v, %lenf
  ret float %r
}

; CHECK: caller - {[-1]:Float@float} |{[-1]:Pointer}:{} {[-1]:Pointer}:{} {[-1]:Integer}:{}
; CHECK-NEXT: ptr %mem: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer, [-1,8]:Pointer, [-1,8,0]:Float@float}
; CHECK-NEXT: ptr %arr: {[-1]:Pointer, [-1,0]:Pointer, [-1,0,0]:Float@float}
; CHECK-NEXT: i1 %c: {[-1]:Integer}
; CHECK: %base = phi ptr [ %mem, %a ], [ %arr, %b ]: {[-1]:Pointer}
; CHECK-NEXT:   %off = phi i64 [ 8, %a ], [ 0, %b ]: {[-1]:Integer}
; CHECK-NEXT:   %p = getelementptr i8, ptr %base, i64 %off: {[-1]:Pointer, [-1,0]:Pointer, [-1,0,0]:Float@float}
; CHECK-NEXT:   %data = load ptr, ptr %p, align 8: {[-1]:Pointer, [-1,0]:Float@float}
; CHECK-NEXT:   %v = load float, ptr %data, align 4: {[-1]:Float@float}
