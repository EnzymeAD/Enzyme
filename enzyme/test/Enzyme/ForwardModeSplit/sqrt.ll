; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -O3 -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,default<O3>" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %0 = tail call fast double @llvm.sqrt.f64(double %x)
  ret double %0
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_fwdsplit(double (double)* nonnull @tester, double %x, double 1.0, i8* null)
  ret double %0
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.sqrt.f64(double)

; Function Attrs: nounwind
declare double @__enzyme_fwdsplit(double (double)*, ...)

; CHECK: define double @test_derivative(double %x)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = tail call fast double @llvm.sqrt.f64(double %x)
; CHECK-NEXT:   %1 = fdiv fast double 5.000000e-01, %0
; CHECK-NEXT:   %2 = fcmp fast ueq double %x, 0.000000e+00
; CHECK-NEXT:   %3 = select{{( fast)?}} i1 %2, double 0.000000e+00, double %1
; CHECK-NEXT:   ret double %3
; CHECK-NEXT: }
