; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -early-cse -instcombine -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(early-cse,instcombine)" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define double @tester(double %y, double %x) {
entry:
  %call = call double @atan2(double %y, double %x)
  ret double %call
}

define double @test_derivative(double %y, double %x) {
entry:
  %0 = tail call double (...) @__enzyme_fwddiff(double (double, double)* nonnull @tester, double %y, double 1.000000e+00, double %x, double 1.000000e+00)
  ret double %0
}

declare double @atan2(double, double)

; Function Attrs: nounwind
declare double @__enzyme_fwddiff(...)

; CHECK-LABEL: define internal double @fwddiffetester(
; CHECK-NEXT: entry:
; CHECK-DAG:   %[[a3:.+]] = fmul double %"y'", %x
; CHECK-DAG:    %[[a1:.+]] = fmul double %x, %x
; CHECK-DAG:    %[[a0:.+]] = fmul double %y, %y
; CHECK-DAG:   %[[a2:.+]] = fadd double %[[a1]], %[[a0]]
; CHECK-DAG:   %[[a7:.+]] = fdiv double %[[a3]], %[[a2]]
; CHECK-DAG:   %[[a4:.+]] = fmul double %"x'", %y
; CHECK-DAG:   %[[a8:.+]] = fdiv double %[[a4]], %[[a2]]
; CHECK-DAG:   %[[a5:.+]] = fsub double %[[a7]], %[[a8]]
; CHECK-NEXT:   ret double %[[a5]]
; CHECK-NEXT: }

