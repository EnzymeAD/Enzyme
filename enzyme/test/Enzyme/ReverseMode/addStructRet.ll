; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %0 = fadd fast double %x, %y
  ret double %0
}

define { double, double } @test_derivative(double %x, double %y) {
entry:
  %0 = tail call { double, double } (double (double, double)*, ...) @__enzyme_autodiff(double (double, double)* nonnull @tester, double %x, double %y)
  ret { double, double } %0
}

; Function Attrs: nounwind
declare { double, double } @__enzyme_autodiff(double (double, double)*, ...)

; CHECK: define internal { double, double } @diffetester(double %x, double %y, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = insertvalue { double, double } undef, double %differeturn, 0
; CHECK-NEXT:   %1 = insertvalue { double, double } %0, double %differeturn, 1
; CHECK-NEXT:   ret { double, double } %1
; CHECK-NEXT: }
