; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %0 = tail call fast double @llvm.sqrt.f64(double %x)
  ret double %0
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @tester, double %x, double 1.0)
  ret double %0
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.sqrt.f64(double)

; Function Attrs: nounwind
declare double @__enzyme_fwddiff(double (double)*, ...)

; CHECK: define internal double @fwddiffetester(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[i3:.+]] = fcmp fast ueq double %x, 0.000000e+00
; CHECK-NEXT:   %[[i0:.+]] = call fast double @llvm.sqrt.f64(double %x)
; CHECK-NEXT:   %[[i1:.+]] = fmul fast double 2.000000e+00, %[[i0]]
; CHECK-NEXT:   %[[i2:.+]] = fdiv fast double %"x'", %[[i1]]
; CHECK-NEXT:   %[[i4:.+]] = select{{( fast)?}} i1 %[[i3]], double 0.000000e+00, double %[[i2]]
; CHECK-NEXT:   ret double %[[i4]]
; CHECK-NEXT: }
