; RUN: if [ %llvmver -ge 15 ]; then not %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S 2>&1 | FileCheck %s; fi

define double @f(double %x, double %y) {
  %r = fmul double %x, %y
  ret double %r
}

declare void @__enzyme_autodiff(ptr sret(double), ...)

define void @test(ptr %out, double %x, double %y) {
  call void (ptr, ...) @__enzyme_autodiff(ptr sret(double) %out, ptr @f, double %x, double %y)
  ret void
}

; CHECK: Enzyme: Cannot cast return type of gradient { double, double }
; CHECK-SAME: to desired type double
