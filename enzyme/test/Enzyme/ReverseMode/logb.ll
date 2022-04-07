; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -O3 -S | FileCheck %s

declare double @__enzyme_autodiff(i8*, ...)
declare double @logb(double)

define double @test(double %x) {
entry:
  %call = call double @logb(double %x)
  ret double %call
}

define double @test_derivative(double %x) {
entry:
  %call = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (double)* @test to i8*), double %x)
  ret double %call
}

; CHECK: define double @test_derivative(double %x)
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret double 0.000000e+00
; CHECK-NEXT: }