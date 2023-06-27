; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %0 = fmul fast double %x, %y
  ret double %0
}

define double @test_derivative(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__enzyme_autodiff(double (double, double)* nonnull @tester, double %x, double %y)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double, double)*, ...)

; CHECK: define internal {{(dso_local )?}}{ double, double } @diffetester(double %x, double %y, double %[[differet:.+]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[diffex:.+]] = fmul fast double %[[differet]], %y
; CHECK-NEXT:   %[[diffey:.+]] = fmul fast double %[[differet]], %x
; CHECK-NEXT:   %[[i0:.+]] = insertvalue { double, double } undef, double %[[diffex]], 0
; CHECK-NEXT:   %[[i1:.+]] = insertvalue { double, double } %[[i0]], double %[[diffey]], 1
; CHECK-NEXT:   ret { double, double } %[[i1]]
; CHECK-NEXT: }
