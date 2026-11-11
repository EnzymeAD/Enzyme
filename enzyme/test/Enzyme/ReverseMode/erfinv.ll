; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -simplifycfg -instsimplify -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,%simplifycfg,instsimplify,adce)" -S | FileCheck %s

; d/dx erfinv(x) = sqrt(pi)/2 * exp(erfinv(x)^2)

declare double @erfinv(double)

define double @tester(double %x) {
entry:
  %call = call double @erfinv(double %x)
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
; CHECK-NEXT:   %[[V0:.+]] = call fast double @erfinv(double %x)
; CHECK-NEXT:   %[[V1:.+]] = fmul fast double %[[V0]], %[[V0]]
; CHECK-NEXT:   %[[V2:.+]] = call fast double @llvm.exp.f64(double %[[V1]])
; CHECK-NEXT:   %[[V3:.+]] = fmul fast double 0x3FEC5BF891B4EF6B, %[[V2]]
; CHECK-NEXT:   %[[V4:.+]] = fmul fast double %[[V3]], %differeturn
; CHECK-NEXT:   %[[V5:.+]] = insertvalue { double } undef, double %[[V4]], 0
; CHECK-NEXT:   ret { double } %[[V5]]
; CHECK-NEXT: }
