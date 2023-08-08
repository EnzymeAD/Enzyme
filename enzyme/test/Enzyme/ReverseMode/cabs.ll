; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -early-cse -instsimplify -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,%simplifycfg,early-cse,instsimplify)"  -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %call = call double @cabs(double %x, double %y)
  ret double %call
}

define double @test_derivative(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__enzyme_autodiff(double (double, double)* nonnull @tester, double %x, double %y)
  ret double %0
}

declare double @cabs(double, double)

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double, double)*, ...)

; CHECK: define internal { double, double } @diffetester(double %x, double %y, double {{(noundef )?}}%differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call fast double @cabs(double %x, double %y)
; CHECK-NEXT:   %1 = fdiv fast double %x, %0
; CHECK-NEXT:   %2 = fmul fast double %differeturn, %1
; CHECK-NEXT:   %3 = fdiv fast double %y, %0
; CHECK-NEXT:   %4 = fmul fast double %differeturn, %3
; CHECK-NEXT:   %5 = insertvalue { double, double } undef, double %2, 0
; CHECK-NEXT:   %6 = insertvalue { double, double } %5, double %4, 1
; CHECK-NEXT:   ret { double, double } %6
; CHECK-NEXT: }
