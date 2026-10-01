; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

; tanh'(x) = 4u / (1 + u)^2 with u = exp(-2|x|), see ReverseMode/tanh.ll.

define double @tester(double %x) {
entry:
  %0 = tail call fast double @tanh(double %x)
  ret double %0
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @tester, double %x, double 1.0)
  ret double %0
}

declare double @tanh(double)
declare double @__enzyme_fwddiff(double (double)*, ...)

; CHECK: define internal double @fwddiffetester(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call fast double @llvm.fabs.f64(double %x)
; CHECK-NEXT:   %1 = fmul fast double -2.000000e+00, %0
; CHECK-NEXT:   %2 = call fast double @llvm.exp.f64(double %1)
; CHECK-NEXT:   %3 = fmul fast double 4.000000e+00, %2
; CHECK-NEXT:   %4 = fmul fast double %"x'", %3
; CHECK-NEXT:   %5 = fadd fast double 1.000000e+00, %2
; CHECK-NEXT:   %6 = fmul fast double %5, %5
; CHECK-NEXT:   %7 = fdiv fast double %4, %6
; CHECK-NEXT:   ret double %7
; CHECK-NEXT: }
