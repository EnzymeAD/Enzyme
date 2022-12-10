; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -early-cse -simplifycfg -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,mem2reg,early-cse,simplifycfg" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %0 = tail call fast double @llvm.log10.f64(double %x)
  ret double %0
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @tester, double %x, double 1.0)
  ret double %0
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.log10.f64(double)

; Function Attrs: nounwind
declare double @__enzyme_fwddiff(double (double)*, ...)

; CHECK: define internal double @fwddiffetester(double %x, double %"x'") 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = fmul fast double 0x40026BB1BBB55516, %x
; CHECK-NEXT:   %1 = fdiv fast double %"x'", %0
; CHECK-NEXT:   ret double %1
; CHECK-NEXT: }
