; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=caller -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=caller -S -o /dev/null | FileCheck %s

; Test that extractvalue from a call returning a struct with float arrays
; properly propagates type information even when the aggregate operand has
; no known type (e.g., from an opaque function call result).

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.c_v = type { [2 x float], [2 x [3 x float]] }

declare %struct.c_v @pre_work()

define void @caller() {
entry:
  %cv = call %struct.c_v @pre_work()
  %f2 = extractvalue %struct.c_v %cv, 0
  %f2x3 = extractvalue %struct.c_v %cv, 1
  %f0 = extractvalue [2 x float] %f2, 0
  %f1 = extractvalue [2 x float] %f2, 1
  %f00 = extractvalue [2 x [3 x float]] %f2x3, 0, 0
  ret void
}

; CHECK-LABEL: caller
; CHECK:   %f2 = extractvalue %struct.c_v %cv, 0: {[-1]:Float@float}
; CHECK:   %f2x3 = extractvalue %struct.c_v %cv, 1: {[-1]:Float@float}
; CHECK:   %f0 = extractvalue [2 x float] %f2, 0: {[-1]:Float@float}
; CHECK:   %f1 = extractvalue [2 x float] %f2, 1: {[-1]:Float@float}
; CHECK:   %f00 = extractvalue [2 x [3 x float]] %f2x3, 0, 0: {[-1]:Float@float}
