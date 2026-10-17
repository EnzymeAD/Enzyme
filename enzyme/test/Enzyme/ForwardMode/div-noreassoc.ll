; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -instcombine -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(instcombine)" -enzyme-preopt=false -S | FileCheck %s

; The derivative of c / y is emitted as (-dy * (c / y)) / y. It must not be
; reassociated into -c * dy / (y * y): y * y overflows for large |y| (in single
; precision already for |y| > ~1.8e19), giving 0 or NaN derivatives.
define float @tester(float %y) {
entry:
  %0 = fdiv float 6.500000e+01, %y
  ret float %0
}

define float @test_derivative(float %y) {
entry:
  %0 = tail call float (float (float)*, ...) @__enzyme_fwddiff(float (float)* nonnull @tester, float %y, float 1.0)
  ret float %0
}

declare float @__enzyme_fwddiff(float (float)*, ...)

; CHECK: define internal {{(dso_local )?}}float @fwddiffetester(float %y, float %"y'")
; CHECK-NOT: fmul {{.*}}float %y, %y
; CHECK: ret float
