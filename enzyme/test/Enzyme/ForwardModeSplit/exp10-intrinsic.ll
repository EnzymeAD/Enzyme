; RUN: if [ %llvmver -ge 18 ]; then %opt < %s %newLoadEnzyme -passes="enzyme,default<O3>" -enzyme-preopt=false -S | FileCheck %s; fi

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %0 = tail call fast double @llvm.exp10.f64(double %x)
  ret double %0
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_fwdsplit(double (double)* nonnull @tester, double %x, double 1.0, i8* null)
  ret double %0
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.exp10.f64(double)

; Function Attrs: nounwind
declare double @__enzyme_fwdsplit(double (double)*, ...)

; equivalent to log(10) * exp10(x)
; CHECK: define double @test_derivative(double %x)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = tail call fast double @llvm.exp10.f64(double %x)
; CHECK-NEXT:   %1 = fmul fast double %0, 0x40026BB1BBB55516
; CHECK-NEXT:   ret double %1
; CHECK-NEXT: }
