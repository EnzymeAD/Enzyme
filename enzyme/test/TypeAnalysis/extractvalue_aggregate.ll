; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=compute -enzyme-loose-types -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=compute -enzyme-loose-types -S -o /dev/null | FileCheck %s

; Regression test for EnzymeAD/Enzyme#2630 (and #2463): with looseTypeAnalysis,
; TypeAnalysis seeds float type info for extractvalue results whose source
; aggregate comes from an opaque external call. The function takes i64 (not
; float) so the only source of float info is the extractvalue LLVM type seeding.

target triple = "x86_64-unknown-linux-gnu"

%struct.CV = type { [2 x float], [2 x [3 x float]] }

declare %struct.CV @pre_work(i64)

define void @compute(i64 %opaque) {
entry:
  %r = call %struct.CV @pre_work(i64 %opaque)
  %a = extractvalue %struct.CV %r, 0
  %a0 = extractvalue [2 x float] %a, 0
  ret void
}

; CHECK: compute - {} |{[-1]:Integer}:{}
; CHECK-NEXT: i64 %opaque: {[-1]:Integer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %r = call %struct.CV @pre_work(i64 %opaque): {[0]:Float@float, [4]:Float@float, [8]:Float@float, [12]:Float@float, [16]:Float@float, [20]:Float@float, [24]:Float@float, [28]:Float@float}
; CHECK-NEXT:   %a = extractvalue %struct.CV %r, 0: {[-1]:Float@float}
; CHECK-NEXT:   %a0 = extractvalue [2 x float] %a, 0: {[-1]:Float@float}
; CHECK-NEXT:   ret void: {}
