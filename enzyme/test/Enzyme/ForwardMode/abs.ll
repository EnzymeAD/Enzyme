; RUN: %opt < %s %loadEnzyme -enzyme -S | FileCheck %s

define double @tester(i32 %x) {
entry:
  %0 = tail call i32 @llvm.abs.i32(i32 %x, i1 false)
  %1 = sitofp i32 %0 to double
  ret double %1
}

define double @test_derivative(i32 %x) {
entry:
  %0 = tail call double (double (i32)*, ...) @__enzyme_fwddiff(double (i32)* nonnull @tester, i32 %x)

  ret double %0
}

declare i32 @llvm.abs.i32(i32, i1 immarg)
declare double @__enzyme_fwddiff(double (i32)*, ...)

; CHECK: define internal double @fwddiffetester(i32 %x)
; CHECK: ret double 0.000000e+00

