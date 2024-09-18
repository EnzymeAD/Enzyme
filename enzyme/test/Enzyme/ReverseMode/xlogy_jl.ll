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
; CHECK-NEXT:   %0 = call fast double @llvm.log.f64(double %y)
; CHECK-NEXT:   %1 = fmul fast double %0, %differeturn
; CHECK-NEXT:   %2 = fcmp fast oeq double %x, 0.000000e+00
; CHECK-NEXT:   %3 = fdiv fast double %x, %y
; CHECK-NEXT:   %4 = fmul fast double %3, %differeturn
; CHECK-NEXT:   %5 = select fast i1 %2, double 0.000000e+00, double %4
; CHECK-NEXT:   %6 = insertvalue { double, double } undef, double %1, 0
; CHECK-NEXT:   %7 = insertvalue { double, double } %6, double %5, 1
; CHECK-NEXT:   ret { double, double } %7
; CHECK-NEXT: }
