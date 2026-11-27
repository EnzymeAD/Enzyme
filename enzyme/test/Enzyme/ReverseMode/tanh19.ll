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
; CHECK-NEXT:  %0 = call {{(fast )?}}double @llvm.fabs.f64(double %x)
; CHECK-NEXT:  %1 = fmul {{(fast )?}}double -2.000000e+00, %0
; CHECK-NEXT:  %2 = call {{(fast )?}}double @llvm.exp.f64(double %1)
; CHECK-NEXT:  %3 = fmul {{(fast )?}}double 4.000000e+00, %2
; CHECK-NEXT:  %4 = fmul {{(fast )?}}double %differeturn, %3
; CHECK-NEXT:  %5 = fadd {{(fast )?}}double 1.000000e+00, %2
; CHECK-NEXT:  %6 = fmul {{(fast )?}}double %5, %5
; CHECK-NEXT:  %7 = fdiv {{(fast )?}}double %4, %6
; CHECK:       ret { double }
; CHECK-NEXT: }
