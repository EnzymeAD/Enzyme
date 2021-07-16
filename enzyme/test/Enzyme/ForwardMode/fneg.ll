; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s


; extern double __enzyme_fwddiff(void*, double, double);
;
; double fneg(double x) {
;     return -x;
; }
; 
; double dfneg(double x) {
;     return __enzyme_fwddiff((void*)fneg, x, 1.0);
; }


define double @fneg(double %x) {
  %fneg = fneg double %x
  ret double %fneg
}

define dso_local double @_Z5dfnegd(double %0) {
  %2 = call double @__enzyme_fwddiff(double (double)* @fneg, double %0, double 1.0)
  ret double %2
}

declare double @__enzyme_fwddiff(double (double)*, double, double)


; CHECK: define internal { double } @diffefneg(double %x, double %"x'") {
; CHECK-NEXT:   %1 = fneg fast double %"x'"
; CHECK-NEXT:   %2 = insertvalue { double } undef, double %1, 0
; CHECK-NEXT:   ret { double } %2
; CHECK-NEXT: }