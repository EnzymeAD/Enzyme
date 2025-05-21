; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %0 = tail call fast double @llvm.cos.f64(double %x)
  ret double %0
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_error_estimate(double (double)* nonnull @tester, double %x, double 1.0)
  ret double %0
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.cos.f64(double)

; Function Attrs: nounwind readnone speculatable
declare double @llvm.sin.f64(double)

; Function Attrs: nounwind
declare double @__enzyme_error_estimate(double (double)*, ...)


; CHECK: define internal double @fwddiffetester(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[i0:.+]] = tail call fast double @llvm.cos.f64(double %x)
; CHECK-NEXT:   %[[i1:.+]] = fmul fast double %"x'", %x
; CHECK-NEXT:   %[[i2:.+]] = fdiv fast double %[[i1]], %[[i0]]
; CHECK-NEXT:   %[[i3:.+]] = call fast double @llvm.sin.f64(double %x)
; CHECK-NEXT:   %[[i4:.+]] = fneg fast double %[[i3]]
; CHECK-NEXT:   %[[i5:.+]] = fmul fast double %[[i2]], %[[i4]]
; CHECK-NEXT:   %[[i6:.+]] = call fast double @llvm.fabs.f64(double %[[i5]])
; CHECK-NEXT:   %[[i7:.+]] = bitcast double %[[i0]] to i64
; CHECK-NEXT:   %[[i8:.+]] = xor i64 %[[i7]], 1
; CHECK-NEXT:   %[[i9:.+]] = bitcast i64 %[[i8]] to double
; CHECK-NEXT:   %[[i10:.+]] = fsub fast double %[[i0]], %[[i9]]
; CHECK-NEXT:   %[[i11:.+]] = call fast double @llvm.fabs.f64(double %[[i10]])
; CHECK-NEXT:   %[[i12:.+]] = call fast double @llvm.maxnum.f64(double %[[i11]], double %[[i6]])
; CHECK-NEXT:   ret double %[[i12]]
; CHECK-NEXT: }