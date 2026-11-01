; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -sroa -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,sroa,instsimplify,%simplifycfg)" -S | FileCheck %s

declare double @xlogy_jl(double, double)

define double @tester(double %x, double %y) {
entry:
  %e = tail call double @xlogy_jl(double %x, double %y)
  ret double %e
}

define { double, double } @test_derivative(double %x, double %y) {
entry:
  %0 = call { double, double } (...) @__enzyme_autodiff(double (double, double)* nonnull @tester, double %x, double %y)
  ret { double, double } %0
}


; Function Attrs: nounwind
declare { double, double } @__enzyme_autodiff(...)

; CHECK: define internal { double, double } @diffetester(double %x, double %y, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call double @llvm.log.f64(double %y)
; CHECK-NEXT:   %1 = fmul double %0, %differeturn
; CHECK-NEXT:   %2 = fadd double 0.000000e+00, %1
; CHECK-NEXT:   %3 = fcmp oeq double %x, 0.000000e+00
; CHECK-NEXT:   %4 = fdiv double %x, %y
; CHECK-NEXT:   %5 = fmul double %4, %differeturn
; CHECK-NEXT:   %6 = fadd double 0.000000e+00, %5
; CHECK-NEXT:   %7 = select i1 %3, double 0.000000e+00, double %6
; CHECK-NEXT:   %8 = insertvalue { double, double } undef, double %2, 0
; CHECK-NEXT:   %9 = insertvalue { double, double } %8, double %7, 1
; CHECK-NEXT:   ret { double, double } %9
; CHECK-NEXT: }
