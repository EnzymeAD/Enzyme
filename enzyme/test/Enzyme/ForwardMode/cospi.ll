; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %0 = tail call double @cospi(double %x)
  ret double %0
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (...) @__enzyme_fwddiff(double (double)* nonnull @tester, double %x, double 1.0)
  ret double %0
}

declare double @cospi(double) readnone

; Function Attrs: nounwind
declare double @__enzyme_fwddiff(...)

; CHECK: define internal double @fwddiffetester(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = fadd fast double %x, 5.000000e-01
; CHECK-NEXT:   %1 = call fast double @cospi(double %0)
; CHECK-NEXT:   %2 = fmul fast double %1, %"x'"
; CHECK-NEXT:   %3 = fmul fast double 0x400921FB54442D1F, %2
; CHECK-NEXT:   ret double %3
; CHECK-NEXT: }
