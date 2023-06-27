; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %0 = tail call fast double @llvm.sqrt.f64(double %x)
  ret double %0
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @tester, double %x)
  ret double %0
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.sqrt.f64(double)

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double)*, ...)

; CHECK: define internal { double } @diffetester(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[i3:.+]] = fcmp fast ueq double %x, 0.000000e+00
; CHECK-NEXT:   %[[i0:.+]] = call fast double @llvm.sqrt.f64(double %x)
; CHECK-NEXT:   %[[i1:.+]] = fmul fast double 2.000000e+00, %[[i0]]
; CHECK-NEXT:   %[[i2:.+]] = fdiv fast double %differeturn, %[[i1]]
; CHECK-NEXT:   %[[i4:.+]] = select{{( fast)?}} i1 %[[i3]], double 0.000000e+00, double %[[i2]]
; CHECK-NEXT:   %[[i5:.+]] = insertvalue { double } undef, double %[[i4]], 0
; CHECK-NEXT:   ret { double } %[[i5]]
