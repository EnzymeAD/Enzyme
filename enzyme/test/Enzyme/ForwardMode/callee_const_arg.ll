; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -enzyme-detect-readthrow=0 -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -enzyme-detect-readthrow=0 -S | FileCheck %s

; @inner has no custom rule and never touches %a0, so @outer's readnone on
; the call is accurate -- this does not reproduce the original bug by itself
; (that needs a custom rule whose behavior diverges from the primal; see
; ReverseMode/custom_const_state.ll). It tests shouldDisableNoWrite's
; strategy: once a callee is resolved and has a body Enzyme will analyze, it
; cannot tell an accurate readnone from one hiding a custom rule, so it never
; lets either prune a Const pointer via write-only/no-capture attributes.
; Before the fix, this accurate readnone alone let Enzyme drop %a1. See
; EnzymeAD/Enzyme.jl#3570.

declare double @__enzyme_fwddiff(...)

define double @dsquare(double %x) {
entry:
  %0 = tail call double (...) @__enzyme_fwddiff(ptr @outer, metadata !"enzyme_dup", double %x, double 1.0, metadata !"enzyme_const", ptr null)
  ret double %0
}

define dso_local double @inner(ptr nocapture %a0, double %a1) {
top:
  %a3 = fmul double %a1, %a1
  ret double %a3
}

; Function Attrs: nosync readnone
define double @outer(double %a0, ptr %a1) {
entry:
  %a7 = call double @inner(ptr %a1, double %a0) readnone
  ret double %a7
}

; CHECK: define internal double @fwddiffeouter(double %a0, double %"a0'", ptr %a1)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call fast double @fwddiffeinner(ptr %a1, double %a0, double %"a0'")
; CHECK-NEXT:   ret double %0
; CHECK-NEXT: }
