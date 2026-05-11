; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=compute -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=compute -S -o /dev/null | FileCheck %s

; Regression test for EnzymeAD/Enzyme#2630 (and #2463): TypeAnalysis must
; recover float type info for extractvalue results whose source aggregate
; comes from an opaque external call (no body to interprocedurally analyse).
; Pre-fix, %a/%b/%a0.../%b12 all had empty type info ("{}"), and the
; AdjointGenerator emitted "Cannot deduce type of extract" when reverse-mode
; AD tried to consume them. Post-fix, the LLVM type of each extractvalue
; result (uniform-FP aggregate or single FP) seeds Float@float into the
; result's TypeTree, which UP propagation then back-fills onto the source.

target triple = "x86_64-unknown-linux-gnu"

%struct.CV = type { [2 x float], [2 x [3 x float]] }

declare %struct.CV @pre_work(float)

define float @compute(float %seed) {
entry:
  %r = call %struct.CV @pre_work(float %seed)
  %a = extractvalue %struct.CV %r, 0
  %b = extractvalue %struct.CV %r, 1
  %a0 = extractvalue [2 x float] %a, 0
  %a1 = extractvalue [2 x float] %a, 1
  %b00 = extractvalue [2 x [3 x float]] %b, 0, 0
  %b01 = extractvalue [2 x [3 x float]] %b, 0, 1
  %b02 = extractvalue [2 x [3 x float]] %b, 0, 2
  %b10 = extractvalue [2 x [3 x float]] %b, 1, 0
  %b11 = extractvalue [2 x [3 x float]] %b, 1, 1
  %b12 = extractvalue [2 x [3 x float]] %b, 1, 2
  %m = fmul float %a0, %seed
  ret float %m
}

; CHECK: compute - {[-1]:Float@float} |{[-1]:Float@float}:{}
; CHECK-NEXT: float %seed: {[-1]:Float@float}
; CHECK-NEXT: entry
; CHECK-NEXT:   %r = call %struct.CV @pre_work(float %seed): {[0]:Float@float, [4]:Float@float, [8]:Float@float, [12]:Float@float, [16]:Float@float, [20]:Float@float, [24]:Float@float, [28]:Float@float}
; CHECK-NEXT:   %a = extractvalue %struct.CV %r, 0: {[-1]:Float@float}
; CHECK-NEXT:   %b = extractvalue %struct.CV %r, 1: {[-1]:Float@float}
; CHECK-NEXT:   %a0 = extractvalue [2 x float] %a, 0: {[-1]:Float@float}
; CHECK-NEXT:   %a1 = extractvalue [2 x float] %a, 1: {[-1]:Float@float}
; CHECK-NEXT:   %b00 = extractvalue [2 x [3 x float]] %b, 0, 0: {[-1]:Float@float}
; CHECK-NEXT:   %b01 = extractvalue [2 x [3 x float]] %b, 0, 1: {[-1]:Float@float}
; CHECK-NEXT:   %b02 = extractvalue [2 x [3 x float]] %b, 0, 2: {[-1]:Float@float}
; CHECK-NEXT:   %b10 = extractvalue [2 x [3 x float]] %b, 1, 0: {[-1]:Float@float}
; CHECK-NEXT:   %b11 = extractvalue [2 x [3 x float]] %b, 1, 1: {[-1]:Float@float}
; CHECK-NEXT:   %b12 = extractvalue [2 x [3 x float]] %b, 1, 2: {[-1]:Float@float}
; CHECK-NEXT:   %m = fmul float %a0, %seed: {[-1]:Float@float}
; CHECK-NEXT:   ret float %m: {}

; Negative test: mixed aggregate (float + i32) must NOT seed Float on the i32 field.
; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=compute_mixed -o /dev/null | FileCheck %s --check-prefix=MIXED; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=compute_mixed -S -o /dev/null | FileCheck %s --check-prefix=MIXED

%struct.Mixed = type { float, i32 }

declare %struct.Mixed @opaque_mixed(float)

define float @compute_mixed(float %seed) {
entry:
  %r = call %struct.Mixed @opaque_mixed(float %seed)
  %f = extractvalue %struct.Mixed %r, 0
  %i = extractvalue %struct.Mixed %r, 1
  ret float %f
}

; MIXED: compute_mixed
; MIXED: %f = extractvalue %struct.Mixed %r, 0: {[-1]:Float@float}
; MIXED-NOT: %i = extractvalue %struct.Mixed %r, 1: {[-1]:Float@float}
