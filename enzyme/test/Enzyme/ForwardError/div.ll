; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %0 = fdiv double %x, %y
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
; CHECK-NEXT:   %[[i0:.+]] = fdiv double %x, %y
; CHECK-NEXT:   %[[i1:.+]] = fmul fast double %"x'", %x
; CHECK-NEXT:   %[[i2:.+]] = fdiv fast double %[[i1]], %[[i0]]
; CHECK-NEXT:   %[[i3:.+]] = fdiv fast double %[[i2]], %y
; CHECK-NEXT:   %[[i4:.+]] = call fast double @llvm.fabs.f64(double %[[i3]])
; CHECK-NEXT:   %[[i5:.+]] = fmul fast double %"y'", %y
; CHECK-NEXT:   %[[i6:.+]] = fdiv fast double %[[i5]], %[[i0]]
; CHECK-NEXT:   %[[i7:.+]] = fdiv fast double %[[i6]], %y
; CHECK-NEXT:   %[[i8:.+]] = fdiv fast double %x, %y
; CHECK-NEXT:   %[[i9:.+]] = fmul fast double %[[i7]], %[[i8]]
; CHECK-NEXT:   %[[i10:.+]] = fneg fast double %[[i9]]
; CHECK-NEXT:   %[[i11:.+]] = call fast double @llvm.fabs.f64(double %[[i10]])
; CHECK-NEXT:   %[[i12:.+]] = fadd fast double %[[i4]], %[[i11]]
; CHECK-NEXT:   %[[i13:.+]] = bitcast double %[[i0]] to i64
; CHECK-NEXT:   %[[i14:.+]] = xor i64 %[[i13]], 1
; CHECK-NEXT:   %[[i15:.+]] = bitcast i64 %[[i14]] to double
; CHECK-NEXT:   %[[i16:.+]] = fsub fast double %[[i0]], %[[i15]]
; CHECK-NEXT:   %[[i17:.+]] = call fast double @llvm.fabs.f64(double %[[i16]])
; CHECK-NEXT:   %[[i18:.+]] = call fast double @llvm.maxnum.f64(double %[[i17]], double %[[i12]])
; CHECK-NEXT:   ret double %[[i18]]
; CHECK-NEXT: }
