; RUN: if [ %llvmver -ge 15 ]; then not %opt < %s %OPnewLoadEnzyme -passes="enzyme" -S 2>&1 | FileCheck %s; fi

target datalayout = "e-m:e-p:64:64-i64:64-n8:16:32:64-S128"

declare float @__enzyme_fwddiff(...)

; Float bit-hacking (bitcast float -> i32; add i32; bitcast i32 -> float) has
; known Float@float type and must NOT be treated as a constant narrow integer op.
define float @nextafter_like(float %x) {
entry:
  %i = bitcast float %x to i32
  %j = add i32 %i, 1
  %y = bitcast i32 %j to float
  ret float %y
}

define float @fwd(float %x, float %dx) {
  %r = call float (...) @__enzyme_fwddiff(ptr @nextafter_like, float %x, float %dx)
  ret float %r
}

; CHECK: cannot handle unknown binary operator:   %j = add i32 %i, 1
