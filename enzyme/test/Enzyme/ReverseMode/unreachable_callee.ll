; RUN: %opt < %s %newLoadEnzyme -passes=enzyme -S -o %t
; RUN: %opt < %t -passes=verify -disable-output
; RUN: FileCheck %s < %t

define double @callee(double %x) {
  unreachable
}

define double @caller(double %x) {
  %y = call double @callee(double %x)
  %result = call double @llvm.sin.f64(double %y)
  ret double %result
}

define double @caller_vjp(double %x) {
  %dx = call double @__enzyme_autodiff(ptr @caller, double %x)
  ret double %dx
}

declare double @__enzyme_autodiff(ptr, double)
declare double @llvm.sin.f64(double)

; CHECK: define internal { double } @diffecaller(
; CHECK: call fast double @llvm.cos.f64(
; CHECK: define internal { double } @diffecallee(
; CHECK-NEXT: unreachable
