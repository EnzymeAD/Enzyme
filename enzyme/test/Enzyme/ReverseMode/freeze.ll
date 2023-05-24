; RUN: if [ %llvmver -lt 16 ] && [ %llvmver -ge 10 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -simplifycfg -instsimplify -adce -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 10 ]; then %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,%simplifycfg,instsimplify,adce)" -S | FileCheck %s; fi

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %out = freeze double %x
  ret double %out
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @tester, double %x)
  ret double %0
}

declare double @cbrt(double)

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double)*, ...)

; CHECK: define internal { double } @diffetester(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = freeze double %differeturn
; CHECK-NEXT:   %1 = insertvalue { double } undef, double %0, 0
; CHECK-NEXT:   ret { double } %1
; CHECK-NEXT: }
