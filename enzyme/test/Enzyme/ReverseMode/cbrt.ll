; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -mem2reg -sroa -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,sroa,instsimplify,%simplifycfg)" -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %call = call double @cbrt(double %x)
  ret double %call
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @tester, double %x)
  ret double %0
}

declare double @cbrt(double)

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double)*, ...)

; CHECK: define internal { double } @diffetester(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call double @cbrt(double %x)
; CHECK-DAG:    [[REG1:%[0-9]+]] = fmul double 3.000000e+00, %x
; CHECK-DAG:    [[REG2:%[0-9]+]] = fmul double %differeturn, %0
; CHECK-NEXT:   %3 = fdiv double [[REG2]], [[REG1]]
; CHECK-NEXT:   %4 = fadd double 0.000000e+00, %3
; CHECK-NEXT:   %5 = insertvalue { double } undef, double %4, 0
; CHECK-NEXT:   ret { double } %5
; CHECK-NEXT: }
