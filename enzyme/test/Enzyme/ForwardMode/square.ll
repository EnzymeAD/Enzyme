; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -early-cse -S | FileCheck %s

; source code
; double square(double x) {
;     return x * x;
; }
; 
; double dsquare(double x) {
;     return __builtin_autodiff(square, x);
; }

define double @square(double %x) {
entry:
  %mul = fmul fast double %x, %x
  ret double %mul
}

define double @dsquare(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @square, double %x, double 1.0)
  ret double %0
}

declare double @__enzyme_fwddiff(double (double)*, ...) 

; CHECK: define internal {{(dso_local )?}}{ double } @diffesquare(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = fmul fast double %"x'", %x
; CHECK-NEXT:   %1 = fadd fast double %0, %0
; CHECK-NEXT:   %2 = insertvalue { double } undef, double %1, 0
; CHECK-NEXT:   ret { double } %2
; CHECK-NEXT: }