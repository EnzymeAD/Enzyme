; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s

; Test that on NVPTX targets, the derivative of __nv_tanh uses __nv_cosh (not
; plain "cosh" which doesn't exist on the CUDA device).

target triple = "nvptx64-nvidia-cuda"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"

; Declare __nv_tanh as the CUDA device implementation of tanh
declare double @__nv_tanh(double) #1
attributes #1 = { nounwind "enzyme_math"="tanh" "implements"="llvm.tanh.f64" "implements2"="tanh" }

; Declare __nv_cosh needed for the derivative of tanh
declare double @__nv_cosh(double) #2
attributes #2 = { nounwind "enzyme_math"="cosh" "implements"="llvm.cosh.f64" "implements2"="cosh" }

define void @foo(double* %x_in, double* %x_out) {
entry:
  %x = load double, double* %x_in
  %r = call double @__nv_tanh(double %x)
  store double %r, double* %x_out
  ret void
}

declare void @__enzyme_autodiff(...)

define void @test(double* %x, double* %d_x, double* %y, double* %d_y) {
entry:
  call void (...) @__enzyme_autodiff(void (double*, double*)* @foo,
                                     double* %x, double* %d_x,
                                     double* %y, double* %d_y)
  ret void
}

; Derivative of tanh(x) is 1/cosh(x)^2; on NVPTX must use __nv_cosh not cosh
; CHECK: define internal void @diffefoo(
; CHECK: call double @__nv_cosh(
