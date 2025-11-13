; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define double @tester(double %x0) #0 {
entry:
  %x2 = alloca double, align 8
  %x3 = alloca fp128, align 16
  store double %x0, double* %x2, align 8
  %x4 = load double, double* %x2, align 8
  %x5 = fpext double %x4 to fp128
  store fp128 %x5, fp128* %x3, align 16
  %x6 = load fp128, fp128* %x3, align 16
  %x7 = fptrunc fp128 %x6 to double
  ret double %x7
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @tester, double %x)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double)*, ...)

; CHECK: define internal { double } @diffetester(double %x0, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = insertvalue { double } undef, double %differeturn, 0
; CHECK-NEXT:   ret { double } %0
; CHECK-NEXT: }