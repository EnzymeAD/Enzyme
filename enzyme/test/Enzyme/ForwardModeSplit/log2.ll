; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false  -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme"  -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %0 = tail call fast double @llvm.log2.f64(double %x)
  ret double %0
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_fwdsplit(double (double)* nonnull @tester, double %x, double 1.0, i8* null)
  ret double %0
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.log2.f64(double)

; Function Attrs: nounwind
declare double @__enzyme_fwdsplit(double (double)*, ...)

; CHECK: define internal double @fwddiffetester(double %x, double %"x'", i8* %tapeArg)
; CHECK-NEXT: entry:
; equivalent to 1/log(2) / x
; CHECK-NEXT:   %0 = fmul fast double 0x3FE62E42FEFA39EF, %x
; CHECK-NEXT:   %1 = fdiv fast double %"x'", %0
; CHECK-NEXT:   ret double %1
; CHECK-NEXT: }
