; End-to-end check: with loose-types, the full enzyme pass succeeds (no
; "Cannot deduce type of extract" diagnostic) and emits a reverse-mode
; entry point. Pre-fix this run failed because the extracted [2 x float]
; aggregates had empty TypeTrees; the AdjointGenerator's loose-types
; fallback (AdjointGenerator.h:1894) only handled primitive FP/int and
; couldn't fill aggregate types. Fix extends that fallback to walk
; aggregate types via uniformFPLeafType (same file scope).
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-loose-types -enzyme-type-warning=0 -enzyme-preopt=false --enzyme-assume-unknown-nofree=1 -S | FileCheck %s --check-prefix=ENZYME

; Regression test for EnzymeAD/Enzyme#2630 (and #2463): with looseTypeAnalysis,
; the AdjointGenerator's loose-types fallback walks aggregate types so
; extractvalue results whose source aggregate comes from an opaque external
; call get seeded with the leaf FP type.

target triple = "x86_64-unknown-linux-gnu"

%struct.CV = type { [2 x float], [2 x [3 x float]] }

; pre_work has a trivial body (returns undef) instead of being an opaque
; declaration. Enzyme needs its primal struct return available in the reverse
; pass and can only synthesize the augmented forward pass for a function with
; a body — a bodyless declaration aborts with "No augmented forward pass found
; for pre_work", which is orthogonal to the type-deduction path under test.
; Returning undef keeps the result untyped, so the extractvalue results still
; have empty TypeTrees and hit the loose-types aggregate fallback. The leaves
; that are only consumed by the inactive opaque_sink (e.g. %a1, and all but
; one leaf of %b) stay unconstrained, which is what makes the aggregate type
; undeducible without the fix.
define %struct.CV @pre_work(i64 %n) #0 {
  ret %struct.CV undef
}
declare void @opaque_sink(float, float) #0
declare float @__enzyme_autodiff(...)

attributes #0 = { "enzyme_inactive" }

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

; ENZYME: define internal {{.*}} @diffead_compute
; ENZYME-NOT: Cannot deduce type of extract
