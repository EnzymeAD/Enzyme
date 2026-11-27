; RUN: if [ %llvmver -ge 15 ]; then not %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S 2>&1 | FileCheck %s; fi

define double @f(double %x, double %y) {
  %r = fmul double %x, %y
  ret double %r
}

declare ptr @__enzyme_autodiff(...)

define ptr @test(double %x, double %y) {
  %r = call ptr (...) @__enzyme_autodiff(ptr @f, double %x, double %y)
  ret ptr %r
}

; CHECK: Enzyme: Cannot cast return type of gradient { double, double }
; CHECK-SAME: to desired type ptr
