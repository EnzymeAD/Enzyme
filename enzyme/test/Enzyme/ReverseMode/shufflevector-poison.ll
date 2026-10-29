; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -early-cse -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,%simplifycfg,early-cse)" -enzyme-preopt=false -S | FileCheck %s

; Lane 1 of the shuffle mask is poison: it reads neither operand, so the
; reverse pass must not propagate its adjoint into %x or %y.

declare { <2 x double>, <2 x double> } @__enzyme_autodiff(...)

define double @tester(<2 x double> %x, <2 x double> %y) {
entry:
  %r = shufflevector <2 x double> %x, <2 x double> %y, <3 x i32> <i32 0, i32 poison, i32 3>
  %a = extractelement <3 x double> %r, i32 0
  %b = extractelement <3 x double> %r, i32 2
  %s = fadd double %a, %b
  ret double %s
}

define { <2 x double>, <2 x double> } @test_derivative(<2 x double> %x, <2 x double> %y) {
entry:
  %0 = tail call { <2 x double>, <2 x double> } (...) @__enzyme_autodiff(double (<2 x double>, <2 x double>)* nonnull @tester, <2 x double> %x, <2 x double> %y)
  ret { <2 x double>, <2 x double> } %0
}

; CHECK-LABEL: define internal { <2 x double>, <2 x double> } @diffetester(<2 x double> %x, <2 x double> %y, double %differeturn)
; CHECK:         %[[dr:.+]] = load <3 x double>, {{.*}}%"r'de"
; CHECK-NEXT:    %[[d0:.+]] = extractelement <3 x double> %[[dr]], i64 0
; CHECK-NOT:     extractelement <3 x double> %[[dr]], i64 1
; CHECK-NOT:     i32 -3
; CHECK:         %[[d2:.+]] = extractelement <3 x double> %[[dr]], i64 2
; CHECK-NEXT:    %[[yp:.+]] = getelementptr inbounds <2 x double>, {{.*}}%"y'de", i32 0, i32 1
; CHECK-NEXT:    %[[yv:.+]] = load double, {{.*}}%[[yp]]
; CHECK-NEXT:    %[[ya:.+]] = fadd fast double %[[yv]], %[[d2]]
; CHECK-NEXT:    store double %[[ya]], {{.*}}%[[yp]]
