; RUN: if [ %llvmver -ge 18 ]; then %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %0 = tail call fast double @llvm.exp10.f64(double %x)
  ret double %0
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @tester, double %x)
  ret double %0
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.exp10.f64(double)

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double)*, ...)

; equivalent to log(10) * exp10(x)
; CHECK: define internal { double } @diffetester(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call fast double @llvm.exp10.f64(double %x)
; CHECK-NEXT:   %1 = fmul fast double 0x40026BB1BBB55516, %0
; CHECK-NEXT:   %2 = fmul fast double %differeturn, %1
; CHECK-NEXT:   %3 = insertvalue { double } undef, double %2, 0
; CHECK-NEXT:   ret { double } %3
; CHECK-NEXT: }
