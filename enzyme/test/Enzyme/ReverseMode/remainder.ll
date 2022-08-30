; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -O3 -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %0 = tail call fast double @remainder(double %x, double %y)
  ret double %0
}

define double @test_derivative1(double %x, double %y) {
entry:
  %0 = call double (double (double, double)*, ...) @__enzyme_autodiff(double (double, double)* nonnull @tester, metadata !"enzyme_const", double %x, double %y)
  ret double %0
}

define double @test_derivative2(double %x, double %y) {
entry:
  %0 = call double (double (double, double)*, ...) @__enzyme_autodiff(double (double, double)* nonnull @tester, double %x, metadata !"enzyme_const", double %y)
  ret double %0
}

; Function Attrs: nounwind readnone speculatable
declare double @remainder(double, double)

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double, double)*, ...)

; CHECK: define double @test_derivative1(double %x, double %y)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = fdiv fast double %x, %y
; CHECK-NEXT:   %1 = tail call fast double @llvm.round.f64(double %0)
; CHECK-NEXT:   ret double %1
; CHECK-NEXT: }

; CHECK: define double @test_derivative2(double %x, double %y)
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret double 1.000000e+00
; CHECK-NEXT: }
