; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=compute -enzyme-loose-types -enzyme-type-warning=0 -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=compute -enzyme-loose-types -enzyme-type-warning=0 -S -o /dev/null | FileCheck %s

; End-to-end check: with loose-types, the full enzyme pass succeeds (no
; "Cannot deduce type of extract" diagnostic) and emits a reverse-mode
; entry point. Pre-fix this run failed because the extracted [2 x float]
; aggregates had empty TypeTrees; the AdjointGenerator's loose-types
; fallback (AdjointGenerator.h:1894) only handles primitive FP/int and
; cannot fill aggregate types.
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-loose-types -enzyme-type-warning=0 -enzyme-preopt=false --enzyme-assume-unknown-nofree=1 -S | FileCheck %s --check-prefix=ENZYME

; Warning check: when the LLVM-type fallback fires, an "assuming
; extractvalue result is..." warning must be emitted (gated on the
; default-on -enzyme-type-warning) so users can see when this potentially
; lossy assumption is being made.
; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=compute -enzyme-loose-types -S -o /dev/null 2>&1 | FileCheck %s --check-prefix=WARN

; Regression test for EnzymeAD/Enzyme#2630 (and #2463): with looseTypeAnalysis,
; TypeAnalysis seeds float type info for extractvalue results whose source
; aggregate comes from an opaque external call. The function takes i64 (not
; float) so the only source of float info is the extractvalue LLVM type seeding.

target triple = "x86_64-unknown-linux-gnu"

%struct.CV = type { [2 x float], [2 x [3 x float]] }

declare %struct.CV @pre_work(i64) #0
declare void @opaque_sink(float, float) #0
declare float @__enzyme_autodiff(...)

attributes #0 = { "enzyme_inactive" }

define void @compute(i64 %opaque) {
entry:
  %r = call %struct.CV @pre_work(i64 %opaque)
  %a = extractvalue %struct.CV %r, 0
  %b = extractvalue %struct.CV %r, 1
  %a0 = extractvalue [2 x float] %a, 0
  %a1 = extractvalue [2 x float] %a, 1
  %b0 = extractvalue [2 x [3 x float]] %b, 0, 0
  %b1 = extractvalue [2 x [3 x float]] %b, 1, 2
  ret void
}

; Active reverse-mode entry. Mirrors #2630's pattern: an opaque struct
; return whose elements feed an active fmul. Without the fix the enzyme
; pass aborts with "Cannot deduce type of extract" on the [2 x float]
; aggregates extracted from %r.
define float @ad_compute(float %seed) {
entry:
  %r = call %struct.CV @pre_work(i64 0)
  %a = extractvalue %struct.CV %r, 0
  %b = extractvalue %struct.CV %r, 1
  %a0 = extractvalue [2 x float] %a, 0
  %a1 = extractvalue [2 x float] %a, 1
  %b00 = extractvalue [2 x [3 x float]] %b, 0, 0
  call void @opaque_sink(float %a0, float %a1)
  %m1 = fmul float %a0, %seed
  %m2 = fmul float %m1, %b00
  ret float %m2
}

define float @caller(float %seed) {
  %d = call float (...) @__enzyme_autodiff(float (float)* @ad_compute, float %seed)
  ret float %d
}

; CHECK: compute - {} |{[-1]:Integer}:{}
; CHECK-NEXT: i64 %opaque: {[-1]:Integer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %r = call %struct.CV @pre_work(i64 %opaque): {[0]:Float@float, [4]:Float@float, [8]:Float@float, [12]:Float@float, [16]:Float@float, [20]:Float@float, [24]:Float@float, [28]:Float@float}
; CHECK-NEXT:   %a = extractvalue %struct.CV %r, 0: {[-1]:Float@float}
; CHECK-NEXT:   %b = extractvalue %struct.CV %r, 1: {[-1]:Float@float}
; CHECK-NEXT:   %a0 = extractvalue [2 x float] %a, 0: {[-1]:Float@float}
; CHECK-NEXT:   %a1 = extractvalue [2 x float] %a, 1: {[-1]:Float@float}
; CHECK-NEXT:   %b0 = extractvalue [2 x [3 x float]] %b, 0, 0: {[-1]:Float@float}
; CHECK-NEXT:   %b1 = extractvalue [2 x [3 x float]] %b, 1, 2: {[-1]:Float@float}
; CHECK-NEXT:   ret void: {}

; ENZYME: define internal {{.*}} @diffead_compute
; ENZYME-NOT: Cannot deduce type of extract

; WARN: Enzyme: assuming extractvalue result is float from LLVM type (looseTypeAnalysis):
; WARN-SAME: %a = extractvalue %struct.CV %r, 0
