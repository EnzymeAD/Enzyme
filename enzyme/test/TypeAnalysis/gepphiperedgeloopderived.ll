; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="print-type-analysis" -type-analysis-func=caller -S -o /dev/null | FileCheck %s; fi

; As gepphiperedgeloop.ll, with the backedge of the base phi carrying a value
; derived from it rather than the phi itself. The loop-carried edge must not
; push the offsets of the backedge into %base.next, which would reach the merged
; phi and again claim the length of %mem is a pointer.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define float @caller(ptr %mem, ptr %arr, i1 %c, i64 %n) {
entry:
  %len = load i64, ptr %mem, align 8
  %lenf = uitofp i64 %len to float
  br i1 %c, label %a, label %b

a:
  br label %loop

b:
  br label %loop

loop:
  %base = phi ptr [ %mem, %a ], [ %arr, %b ], [ %base.next, %loop ]
  %off = phi i64 [ 8, %a ], [ 0, %b ], [ %off, %loop ]
  %i = phi i64 [ 0, %a ], [ 0, %b ], [ %inc, %loop ]
  %acc = phi float [ %lenf, %a ], [ %lenf, %b ], [ %r, %loop ]
  %p = getelementptr i8, ptr %base, i64 %off
  %data = load ptr, ptr %p, align 8
  %v = load float, ptr %data, align 4
  %r = fadd float %v, %acc
  %base.next = getelementptr i8, ptr %base, i64 0
  %inc = add i64 %i, 1
  %done = icmp eq i64 %inc, %n
  br i1 %done, label %exit, label %loop

exit:
  ret float %r
}

; CHECK: caller - {[-1]:Float@float} |{[-1]:Pointer}:{} {[-1]:Pointer}:{} {[-1]:Integer}:{} {[-1]:Integer}:{}
; CHECK-NEXT: ptr %mem: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer, [-1,8]:Pointer, [-1,8,0]:Float@float}
; CHECK-NEXT: ptr %arr: {[-1]:Pointer, [-1,0]:Pointer, [-1,0,0]:Float@float}
; CHECK: %base = phi ptr [ %mem, %a ], [ %arr, %b ], [ %base.next, %loop ]: {[-1]:Pointer}
; CHECK-NEXT:   %off = phi i64 [ 8, %a ], [ 0, %b ], [ %off, %loop ]: {[-1]:Integer}
; CHECK: %p = getelementptr i8, ptr %base, i64 %off: {[-1]:Pointer, [-1,0]:Pointer, [-1,0,0]:Float@float}
; CHECK-NEXT:   %data = load ptr, ptr %p, align 8: {[-1]:Pointer, [-1,0]:Float@float}
; CHECK-NEXT:   %v = load float, ptr %data, align 4: {[-1]:Float@float}
