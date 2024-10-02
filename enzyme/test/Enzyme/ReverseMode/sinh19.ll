; RUN: if [ %llvmver -ge 19 ]; then %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.sinh.f64(double) #14

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %0 = call double @llvm.sinh.f64(double %x)
  ret double %0
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @tester, double %x)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double)*, ...)

; CHECK: define internal { double } @diffetester(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:  %0 = call fast double @llvm.cosh.f64(double %x)
; CHECK-NEXT:  %1 = fmul fast double %differeturn, %0
; CHECK-NEXT:  %2 = insertvalue { double } undef, double %1, 0
; CHECK-NEXT:  ret { double } %2
; CHECK-NEXT: }
