; RUN: if [ %llvmver -ge 15 ]; then not %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S 2>&1 | FileCheck %s; fi

define double @f(double %x) {
  %r = fmul double %x, %x
  ret double %r
}

declare ptr @__enzyme_augmentfwd(...)

define ptr @test(double %x) {
  %r = call ptr (...) @__enzyme_augmentfwd(ptr @f, double %x)
  ret ptr %r
}

; CHECK: Enzyme: Cannot cast return type of gradient { ptr, double }
; CHECK-SAME: to desired type ptr
