; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -simplifycfg -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,sroa,instsimplify,%simplifycfg,adce)" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: nounwind readnone willreturn
declare double @cabs([2 x double]) #7

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %agg0 = insertvalue [2 x double] undef, double %x, 0
  %agg1 = insertvalue [2 x double] %agg0, double %y, 1
  %call = call double @cabs([2 x double] %agg1)
  ret double %call
}

define double @test_derivative(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__enzyme_autodiff(double (double, double)* nonnull @tester, double %x, double %y)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double, double)*, ...)


; CHECK: define internal { double, double } @diffetester(double %x, double %y, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %agg0 = insertvalue [2 x double] undef, double %x, 0
; CHECK-NEXT:   %agg1 = insertvalue [2 x double] %agg0, double %y, 1
; CHECK-NEXT:   %0 = call double @cabs([2 x double] %agg1)
; CHECK-NEXT:   %1 = fdiv double %x, %0
; CHECK-NEXT:   %2 = fmul double %differeturn, %1
; CHECK-NEXT:   %3 = fdiv double %y, %0
; CHECK-NEXT:   %4 = fmul double %differeturn, %3
; CHECK-NEXT:   %5 = fadd double 0.000000e+00, %2
; CHECK-NEXT:   %6 = fadd double 0.000000e+00, %4
; CHECK-NEXT:   %7 = insertvalue { double, double } undef, double %5, 0
; CHECK-NEXT:   %8 = insertvalue { double, double } %7, double %6, 1
; CHECK-NEXT:   ret { double, double } %8
; CHECK-NEXT: }
