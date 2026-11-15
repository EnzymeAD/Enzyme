; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-type-analysis" -type-analysis-func=caller -S -o /dev/null | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @caller(ptr %p_fmax, ptr %p_fmin, float %v) {
entry:
  %r_fmax = atomicrmw fmax ptr %p_fmax, double 1.000000e+00 seq_cst
  %r_fmin = atomicrmw fmin ptr %p_fmin, float %v seq_cst
  ret void
}

; CHECK: caller - {} |{[-1]:Pointer, [-1,0]:Float@double}:{} {[-1]:Pointer, [-1,0]:Float@float}:{} {[-1]:Float@float}:{}
; CHECK: entry
; CHECK-NEXT:   %r_fmax = atomicrmw fmax ptr %p_fmax, double 1.000000e+00 seq_cst, align 8: {[-1]:Float@double}
; CHECK-NEXT:   %r_fmin = atomicrmw fmin ptr %p_fmin, float %v seq_cst, align 4: {[-1]:Float@float}
; CHECK-NEXT:   ret void: {}
