; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -print-type-analysis -type-analysis-func=f -enzyme-strict-aliasing=0 -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-type-analysis" -type-analysis-func=f -enzyme-strict-aliasing=0 -S -o /dev/null | FileCheck %s

; Rust's Vec uses NonNull::dangling() (the integer 8, the alignment) as the data
; pointer of an empty allocation, giving a phi of a ptrtoint and a constant int.
; Without strict aliasing the stores/loads in %use (which does not post-dominate
; %join) do not push Pointer back into %ptr, so the PHI hypothesis pass guesses
; %p is an Integer. The DOWN-only hypothesis analyzer then forwarded Integer
; through the inttoptr to the address of a load, which asserted in
; TypeTree::Lookup. The load through a non-pointer must instead invalidate the
; hypothesis.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare noalias ptr @calloc(i64, i64)

define double @f(i64 %n, ptr %out) {
entry:
  %cmp = icmp eq i64 %n, 0
  br i1 %cmp, label %join, label %alloc

alloc:
  %m = tail call noalias ptr @calloc(i64 %n, i64 8)
  %mi = ptrtoint ptr %m to i64
  br label %join

join:
  %p = phi i64 [ %mi, %alloc ], [ 8, %entry ]
  %ptr = inttoptr i64 %p to ptr
  br i1 %cmp, label %exit, label %use

use:
  store double 1.000000e+00, ptr %ptr, align 8
  %ld = load double, ptr %ptr, align 8
  store double %ld, ptr %out, align 8
  br label %exit

exit:
  ret double 0.000000e+00
}

; CHECK: f - {{.*}} |{[-1]:Integer}:{} {[-1]:Pointer}:{}
; CHECK-NEXT: i64 %n: {[-1]:Integer}
; CHECK-NEXT: ptr %out: {[-1]:Pointer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %cmp = icmp eq i64 %n, 0: {[-1]:Integer}
; CHECK-NEXT:   br i1 %cmp, label %join, label %alloc: {}
; CHECK-NEXT: alloc
; CHECK-NEXT:   %m = tail call noalias ptr @calloc(i64 %n, i64 8): {[-1]:Pointer}
; CHECK-NEXT:   %mi = ptrtoint ptr %m to i64: {[-1]:Pointer}
; CHECK-NEXT:   br label %join: {}
; CHECK-NEXT: join
; CHECK-NEXT:   %p = phi i64 [ %mi, %alloc ], [ 8, %entry ]: {}
; CHECK-NEXT:   %ptr = inttoptr i64 %p to ptr: {}
; CHECK-NEXT:   br i1 %cmp, label %exit, label %use: {}
; CHECK-NEXT: use
; CHECK-NEXT:   store double 1.000000e+00, ptr %ptr, align 8: {}
; CHECK-NEXT:   %ld = load double, ptr %ptr, align 8: {}
; CHECK-NEXT:   store double %ld, ptr %out, align 8: {}
; CHECK-NEXT:   br label %exit: {}
; CHECK-NEXT: exit
; CHECK-NEXT:   ret double 0.000000e+00: {}
