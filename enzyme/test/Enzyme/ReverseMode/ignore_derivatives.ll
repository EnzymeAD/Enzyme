; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %y = tail call fast double @__enzyme_ignore_derivatives(double %x)
  %y2 = fmul double %y, %y
  %res = fmul add %y2, %x
  ret double %res
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @tester, double %x)
  ret double %0
}

declare double @__enzyme_ignore_derivatives(double)

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double)*, ...)

; CHECK: define internal { double } @diffetester(double %x, double %differeturn) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = fmul fast double %x, %x
; CHECK-NEXT:   %1 = fsub fast double 1.000000e+00, %0
; CHECK-NEXT:   %2 = call fast double @llvm.sqrt.f64(double %1)
; CHECK-NEXT:   %3 = fdiv fast double %differeturn, %2
; CHECK-NEXT:   %4 = insertvalue { double } undef, double %3, 0
; CHECK-NEXT:   ret { double } %4
; CHECK-NEXT: }
