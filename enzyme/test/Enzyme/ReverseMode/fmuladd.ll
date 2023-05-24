; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -simplifycfg -instsimplify -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,%simplifycfg,instsimplify,adce)" -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x, double %y, double %z) {
entry:
  %0 = tail call fast double @llvm.fmuladd.f64(double %x, double %y, double %z)
  ret double %0
}

define double @test_derivative(double %x, double %y, double %z) {
entry:
  %0 = tail call double (double (double, double, double)*, ...) @__enzyme_autodiff(double (double, double, double)* nonnull @tester, double %x, double %y, double %z)
  ret double %0
}

declare double @llvm.fmuladd.f64(double %a, double %b, double %c)

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double, double, double)*, ...)

; CHECK: define internal { double, double, double } @diffetester(double %x, double %y, double %z, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = fmul fast double %differeturn, %y
; CHECK-NEXT:   %1 = fmul fast double %differeturn, %x
; CHECK-NEXT:   %2 = insertvalue { double, double, double } undef, double %0, 0
; CHECK-NEXT:   %3 = insertvalue { double, double, double } %2, double %1, 1
; CHECK-NEXT:   %4 = insertvalue { double, double, double } %3, double %differeturn, 2
; CHECK-NEXT:   ret { double, double, double } %4
; CHECK-NEXT: }
