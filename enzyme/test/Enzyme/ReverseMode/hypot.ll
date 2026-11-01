; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -early-cse -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,early-cse,instsimplify,%simplifycfg)" -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %call = call double @hypot(double %x, double %y)
  ret double %call
}

define double @tester2(double %x) {
entry:
  %call = call double @hypot(double %x, double 2.000000e+00)
  ret double %call
}

define double @test_derivative(double %x, double %y) {
entry:
  %0 = tail call double (...) @__enzyme_autodiff(double (double, double)* nonnull @tester, double %x, double %y)
  %1 = tail call double (...) @__enzyme_autodiff(double (double)* nonnull @tester2, double %x)
  ret double %0
}

declare double @hypot(double, double)

; Function Attrs: nounwind
declare double @__enzyme_autodiff(...)

; CHECK: define internal { double, double } @diffetester(double %x, double %y, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-DAG:   %[[a1:.+]] = fmul double %differeturn, %x
; CHECK-DAG:   %[[a0:.+]] = call double @hypot(double %x, double %y)
; CHECK-DAG:   %[[a2:.+]] = fdiv double %[[a1]], %[[a0]]
; CHECK-DAG:   %[[a3:.+]] = fadd double 0.000000e+00, %[[a2]]
; CHECK-DAG:   %[[a4:.+]] = fmul double %differeturn, %y
; CHECK-DAG:   %[[a5:.+]] = fdiv double %[[a4]], %[[a0]]
; CHECK-DAG:   %[[a6:.+]] = fadd double 0.000000e+00, %[[a5]]
; CHECK-DAG:   %[[a7:.+]] = insertvalue { double, double } undef, double %[[a3]], 0
; CHECK-DAG:   %[[a8:.+]] = insertvalue { double, double } %[[a7]], double %[[a6]], 1
; CHECK-NEXT:   ret { double, double } %[[a8]]
; CHECK-NEXT: }

; CHECK: define internal { double } @diffetester2(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-DAG:   %[[a1:.+]] = fmul double %differeturn, %x
; CHECK-DAG:   %[[a0:.+]] = call double @hypot(double %x, double 2.000000e+00)
; CHECK-DAG:   %[[a2:.+]] = fdiv double %[[a1]], %[[a0]]
; CHECK-DAG:   %[[a3:.+]] = fadd double 0.000000e+00, %[[a2]]
; CHECK-DAG:   %[[a4:.+]] = insertvalue { double } undef, double %[[a3]], 0
; CHECK-NEXT:   ret { double } %[[a4]]
; CHECK-NEXT: }
