; RUN: opt < %s -load=%llvmshlibdir/LLVMEnzyme%shlibext -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x) {
entry:
  ret double 1.000000e+00
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @llvm.autodiff.p0f_f64f64f(double (double)* nonnull @tester, double %x)
  ret double %0
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.cos.f64(double)

; Function Attrs: nounwind readnone speculatable
declare double @llvm.sin.f64(double)

; Function Attrs: nounwind
declare double @llvm.autodiff.p0f_f64f64f(double (double)*, ...)

; CHECK: define internal { double } @diffetester(double %x, double %[[differet:.+]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret { double } zeroinitializer
; CHECK-NEXT: }
