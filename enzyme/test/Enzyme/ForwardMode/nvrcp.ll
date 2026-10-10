; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -instsimplify -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(instsimplify)" -enzyme-preopt=false -S | FileCheck %s

declare double @__nv_drcp_rn(double)

define double @tester(double %x) {
entry:
  %0 = call double @__nv_drcp_rn(double %x)
  ret double %0
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @tester, double %x, double 1.0)
  ret double %0
}

declare double @__enzyme_fwddiff(double (double)*, ...)

; The divisor is never squared, so the tangent cannot overflow or underflow for large or small x.
; CHECK: define internal {{(dso_local )?}}double @fwddiffetester(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = fdiv fast double %"x'", %x
; CHECK-NEXT:   %1 = fdiv fast double %0, %x
; CHECK-NEXT:   %2 = fneg fast double %1
; CHECK-NEXT:   ret double %2
; CHECK-NEXT: }
