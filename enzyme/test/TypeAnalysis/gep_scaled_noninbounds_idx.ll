; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-type-analysis" -type-analysis-func=callee -S -o /dev/null | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define double @callee(ptr %src, ptr %stride_slot, i64 %j) {
entry:
  %base_val = load double, ptr %src, align 8
  %stride = load i64, ptr %stride_slot, align 8
  %mul = mul i64 %stride, %j
  %p0 = getelementptr [8 x i8], ptr %src, i64 %mul
  %val = load double, ptr %p0, align 8
  %res = fadd double %base_val, %val
  ret double %res
}

; CHECK: callee - {[-1]:Float@double} |{[-1]:Pointer}:{} {[-1]:Pointer}:{} {[-1]:Integer}:{}
; CHECK-NEXT: ptr %src: {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT: ptr %stride_slot: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer}
; CHECK-NEXT: i64 %j: {[-1]:Integer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %base_val = load double, ptr %src, align 8: {[-1]:Float@double}
; CHECK-NEXT:   %stride = load i64, ptr %stride_slot, align 8: {[-1]:Integer}
; CHECK-NEXT:   %mul = mul i64 %stride, %j: {[-1]:Integer}
; CHECK-NEXT:   %p0 = getelementptr [8 x i8], ptr %src, i64 %mul: {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT:   %val = load double, ptr %p0, align 8: {[-1]:Float@double}
; CHECK-NEXT:   %res = fadd double %base_val, %val: {[-1]:Float@double}
; CHECK-NEXT:   ret double %res: {}
