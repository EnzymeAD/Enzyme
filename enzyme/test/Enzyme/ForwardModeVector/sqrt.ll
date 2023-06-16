; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -O3 -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,default<O3>" -enzyme-preopt=false -S | FileCheck %s

%struct.Gradients = type { double, double, double }

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %0 = tail call fast double @llvm.sqrt.f64(double %x)
  ret double %0
}

define %struct.Gradients @test_derivative(double %x) {
entry:
  %0 = tail call %struct.Gradients (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @tester, metadata !"enzyme_width", i64 3, double %x, double 1.0, double 2.0, double 3.0)
  ret %struct.Gradients %0
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.sqrt.f64(double)

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_fwddiff(double (double)*, ...)


; CHECK: define %struct.Gradients @test_derivative(double %x)
; CHECK-NEXT: entry
; CHECK-NEXT:   %[[i2:.+]] = fcmp fast ueq double %x, 0.000000e+00
; CHECK-NEXT:   %[[i0:.+]] = tail call fast double @llvm.sqrt.f64(double %x)
; CHECK-NEXT:   %[[i1:.+]] = fdiv fast double 5.000000e-01, %[[i0]]
; CHECK-NEXT:   %[[i4:.+]] = fdiv fast double 1.000000e+00, %[[i0]]
; CHECK-NEXT:   %[[i6:.+]] = fdiv fast double 1.500000e+00, %[[i0]]
; CHECK-NEXT:   %[[i3:.+]] = select {{(fast )?}}i1 %[[i2]], double 0.000000e+00, double %[[i1]]
; CHECK-NEXT:   %[[i5:.+]] = select {{(fast )?}}i1 %[[i2]], double 0.000000e+00, double %[[i4]]
; CHECK-NEXT:   %[[i7:.+]] = select {{(fast )?}}i1 %[[i2]], double 0.000000e+00, double %[[i6]]
; CHECK-NEXT:   %[[i8:.+]] = insertvalue %struct.Gradients zeroinitializer, double %[[i3]], 0
; CHECK-NEXT:   %[[i9:.+]] = insertvalue %struct.Gradients %[[i8]], double %[[i5]], 1
; CHECK-NEXT:   %[[i10:.+]] = insertvalue %struct.Gradients %[[i9]], double %[[i7]], 2
; CHECK-NEXT:   ret %struct.Gradients %[[i10]]
; CHECK-NEXT: }
