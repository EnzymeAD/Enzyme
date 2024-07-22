; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %0 = fadd double %x, %y
  ret double %0
}

define double @test_derivative(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__enzyme_error_estimate(double (double, double)* nonnull @tester, double %x, double 0.0, double %y, double 0.0)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_error_estimate(double (double, double)*, ...)


; CHECK: define internal double @fwddiffetester(double %x, double %"x'", double %y, double %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[i0:.+]] = fadd double %x, %y
; CHECK-NEXT:   %[[i1:.+]] = fmul fast double %"x'", %x
; CHECK-NEXT:   %[[i2:.+]] = fdiv fast double %[[i1]], %[[i0]]
; CHECK-NEXT:   %[[i3:.+]] = call fast double @llvm.fabs.f64(double %[[i2]])
; CHECK-NEXT:   %[[i4:.+]] = fmul fast double %"y'", %y
; CHECK-NEXT:   %[[i5:.+]] = fdiv fast double %[[i4]], %[[i0]]
; CHECK-NEXT:   %[[i6:.+]] = call fast double @llvm.fabs.f64(double %[[i5]])
; CHECK-NEXT:   %[[i7:.+]] = fadd fast double %[[i3]], %[[i6]]
; CHECK-NEXT:   %[[i8:.+]] = bitcast double %[[i0]] to i64
; CHECK-NEXT:   %[[i9:.+]] = xor i64 %[[i8]], 1
; CHECK-NEXT:   %[[i10:.+]] = bitcast i64 %[[i9]] to double
; CHECK-NEXT:   %[[i11:.+]] = fsub fast double %[[i0]], %[[i10]]
; CHECK-NEXT:   %[[i12:.+]] = call fast double @llvm.fabs.f64(double %[[i11]])
; CHECK-NEXT:   %[[i13:.+]] = call fast double @llvm.maxnum.f64(double %[[i12]], double %[[i7]])
; CHECK-NEXT:   ret double %[[i13]]
; CHECK-NEXT: }
