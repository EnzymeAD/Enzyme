; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -simplifycfg -instsimplify -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,%simplifycfg,instsimplify,adce)" -S | FileCheck %s

; C lgamma returns a scalar: d/dx lgamma(x) = digamma(x)

declare double @lgamma(double)

define double @tester(double %x) {
entry:
  %call = call double @lgamma(double %x)
  ret double %call
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @tester, double %x)
  ret double %0
}

declare double @__enzyme_autodiff(double (double)*, ...)

; CHECK: define internal { double } @diffetester(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[V0:.+]] = call fast double @digamma(double %x)
; CHECK-NEXT:   %[[V1:.+]] = fmul fast double %[[V0]], %differeturn
; CHECK-NEXT:   %[[V2:.+]] = insertvalue { double } undef, double %[[V1]], 0
; CHECK-NEXT:   ret { double } %[[V2]]
; CHECK-NEXT: }
