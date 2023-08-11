; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %0 = tail call fast double @sincn(double %x)
  ret double %0
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @tester, double %x)
  ret double %0
}

; Function Attrs: nounwind readnone speculatable
declare double @sincn(double)

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double)*, ...)

; CHECK: define internal { double } @diffetester(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[cmp:.+]] = fcmp fast oeq double %x, 0.000000e+00
; CHECK-NEXT:   %[[a0:.+]] = fmul fast double 0x400921FB54442D1F, %x
; CHECK-NEXT:   %[[i0:.+]] = call fast double @llvm.cos.f64(double %[[a0]])
; CHECK-NEXT:   %[[i1:.+]] = call fast double @sincn(double %x)
; CHECK-NEXT:   %[[i2:.+]] = fsub fast double %[[i0]], %[[i1]]
; CHECK-NEXT:   %[[i3:.+]] = fdiv fast double %[[i2]], %x
; CHECK-NEXT:   %[[i4:.+]] = fmul fast double %differeturn, %[[i3]]
; CHECK-NEXT:   %[[sel:.+]] = select {{(fast )?}}i1 %[[cmp]], double 0.000000e+00, double %[[i4]]
; CHECK-NEXT:   %[[i5:.+]] = insertvalue { double } undef, double %[[sel]], 0
; CHECK-NEXT:   ret { double } %[[i5]]
; CHECK-NEXT: }
