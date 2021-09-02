; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -simplifycfg -adce -S | FileCheck %s

declare double @erfi(double)

define double @tester(double %x) {
entry:
  %call = call double @erfi(double %x)
  ret double %call
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @tester, double %x, double 1.0)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_fwddiff(double (double)*, ...)

; CHECK: define internal { double } @diffetester(double %x, double %"x'") {
; CHECK-NEXT: entry:
; CHECK-NEXT:    %0 = fmul fast double %x, %x
; CHECK-NEXT:    %1 = call fast double @llvm.exp.f64(double %0)
; CHECK-NEXT:    %2 = fmul fast double %1, 0x3FF20DD750429B6D
; CHECK-NEXT:    %3 = fmul fast double %2, %"x'"
; CHECK-NEXT:    %4 = insertvalue { double } undef, double %3, 0
; CHECK-NEXT:    ret { double } %4
; CHECK-NEXT: }
