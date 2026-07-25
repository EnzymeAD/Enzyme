; RUN: if [ %llvmver -ge 18 ]; then %opt < %s %newLoadEnzyme -passes="enzyme,default<O3>" -enzyme-preopt=false -S | FileCheck %s; fi

%struct.Gradients = type { double, double }

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_fwddiff(double (double)*, ...)

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %0 = tail call fast double @llvm.exp10.f64(double %x)
  ret double %0
}

define %struct.Gradients @test_derivative(double %x) {
entry:
  %0 = tail call %struct.Gradients (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @tester, metadata !"enzyme_width", i64 2, double %x, double 1.0, double 2.5)
  ret %struct.Gradients %0
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.exp10.f64(double)

; equivalent to log(10) * exp10(x) scaled by each seed
; CHECK: define %struct.Gradients @test_derivative(double %x)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = tail call fast double @llvm.exp10.f64(double %x)
; CHECK-NEXT:   %1 = fmul fast double %0, 0x40026BB1BBB55516
; CHECK-NEXT:   %2 = fmul fast double %0, 0x4017069E2AA2AA5C
; CHECK-NEXT:   %3 = insertvalue %struct.Gradients zeroinitializer, double %1, 0
; CHECK-NEXT:   %4 = insertvalue %struct.Gradients %3, double %2, 1
; CHECK-NEXT:   ret %struct.Gradients %4
; CHECK-NEXT: }
