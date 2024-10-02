; RUN: if [ %llvmver -ge 19 ]; then %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.tanh.f64(double) #14

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %0 = call double @llvm.tanh.f64(double %x)
  ret double %0
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (ptr, ...) @__enzyme_autodiff(ptr nonnull @tester, double %x)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(ptr, ...)

; CHECK: define internal { double } @diffetester(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:  %0 = call fast double @llvm.cosh.f64(double %x)
; CHECK-NEXT:  %1 = fmul fast double %0, %0
; CHECK-NEXT:  %2 = fdiv fast double %differeturn, %1
; CHECK-NEXT:  %3 = insertvalue { double } undef, double %2, 0
; CHECK-NEXT:  ret { double } %3
; CHECK-NEXT: }
