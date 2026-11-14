; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,early-cse,%simplifycfg)" -S | FileCheck %s; fi

declare double @llvm.minimum.f64(double, double)
declare <2 x double> @llvm.minimum.v2f64(<2 x double>, <2 x double>)
declare { double, double } @__enzyme_autodiff(...)
declare { <2 x double>, <2 x double> } @__enzyme_autodiff_v2(...)

define double @tester(double %x, double %y) {
entry:
  %0 = tail call double @llvm.minimum.f64(double %x, double %y)
  ret double %0
}

define <2 x double> @tester_vec(<2 x double> %x, <2 x double> %y) {
entry:
  %0 = tail call <2 x double> @llvm.minimum.v2f64(<2 x double> %x, <2 x double> %y)
  ret <2 x double> %0
}

define { double, double } @test_derivative(double %x, double %y) {
entry:
  %0 = tail call { double, double } (...) @__enzyme_autodiff(ptr nonnull @tester, double %x, double %y)
  ret { double, double } %0
}

define { <2 x double>, <2 x double> } @test_derivative_vec(<2 x double> %x, <2 x double> %y) {
entry:
  %0 = tail call { <2 x double>, <2 x double> } (...) @__enzyme_autodiff_v2(ptr nonnull @tester_vec, <2 x double> %x, <2 x double> %y)
  ret { <2 x double>, <2 x double> } %0
}

; CHECK: define internal {{(dso_local )?}}{ double, double } @diffetester(double %x, double %y, double %[[differet:.+]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[ORD:.+]] = fcmp ord double %y, %y
; CHECK-NEXT:   %[[ULT:.+]] = fcmp ult double %x, %y
; CHECK-NEXT:   %[[OEQ:.+]] = fcmp oeq double %x, %y
; CHECK-NEXT:   %[[XI:.+]] = bitcast double %x to i64
; CHECK-NEXT:   %[[YI:.+]] = bitcast double %y to i64
; CHECK-NEXT:   %[[NOTYI:.+]] = xor i64 %[[YI]], -1
; CHECK-NEXT:   %[[MASK:.+]] = and i64 %[[XI]], %[[NOTYI]]
; CHECK-NEXT:   %[[SIGN:.+]] = icmp slt i64 %[[MASK]], 0
; CHECK-NEXT:   %[[NPZ:.+]] = and i1 %[[OEQ]], %[[SIGN]]
; CHECK-NEXT:   %[[OR:.+]] = or i1 %[[ULT]], %[[NPZ]]
; CHECK-NEXT:   %[[cmp:.+]] = and i1 %[[ORD]], %[[OR]]
; CHECK-NEXT:   %[[diffex:.+]] = select {{(fast )?}}i1 %[[cmp]], double %[[differet]], double 0.000000e+00
; CHECK-NEXT:   %[[diffey:.+]] = select {{(fast )?}}i1 %[[cmp]], double 0.000000e+00, double %[[differet]]
; CHECK-NEXT:   %[[iv0:.+]] = insertvalue { double, double } undef, double %[[diffex]], 0
; CHECK-NEXT:   %[[iv1:.+]] = insertvalue { double, double } %[[iv0]], double %[[diffey]], 1
; CHECK-NEXT:   ret { double, double } %[[iv1]]
; CHECK-NEXT: }

; CHECK: define internal {{(dso_local )?}}{ <2 x double>, <2 x double> } @diffetester_vec(<2 x double> %x, <2 x double> %y, <2 x double> %[[differetv:.+]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[ORDV:.+]] = fcmp ord <2 x double> %y, %y
; CHECK-NEXT:   %[[ULTV:.+]] = fcmp ult <2 x double> %x, %y
; CHECK-NEXT:   %[[OEQV:.+]] = fcmp oeq <2 x double> %x, %y
; CHECK-NEXT:   %[[XIV:.+]] = bitcast <2 x double> %x to <2 x i64>
; CHECK-NEXT:   %[[YIV:.+]] = bitcast <2 x double> %y to <2 x i64>
; CHECK-NEXT:   %[[NOTYIV:.+]] = xor <2 x i64> %[[YIV]], <i64 -1, i64 -1>
; CHECK-NEXT:   %[[MASKV:.+]] = and <2 x i64> %[[XIV]], %[[NOTYIV]]
; CHECK-NEXT:   %[[SIGNV:.+]] = icmp slt <2 x i64> %[[MASKV]], zeroinitializer
; CHECK-NEXT:   %[[NPZV:.+]] = and <2 x i1> %[[OEQV]], %[[SIGNV]]
; CHECK-NEXT:   %[[ORV:.+]] = or <2 x i1> %[[ULTV]], %[[NPZV]]
; CHECK-NEXT:   %[[CMPV:.+]] = and <2 x i1> %[[ORDV]], %[[ORV]]
; CHECK-NEXT:   %[[DIFFEXV:.+]] = select {{(fast )?}}<2 x i1> %[[CMPV]], <2 x double> %[[differetv]], <2 x double> zeroinitializer
; CHECK-NEXT:   %[[DIFFEYV:.+]] = select {{(fast )?}}<2 x i1> %[[CMPV]], <2 x double> zeroinitializer, <2 x double> %[[differetv]]
; CHECK-NEXT:   %[[IVV0:.+]] = insertvalue { <2 x double>, <2 x double> } undef, <2 x double> %[[DIFFEXV]], 0
; CHECK-NEXT:   %[[IVV1:.+]] = insertvalue { <2 x double>, <2 x double> } %[[IVV0]], <2 x double> %[[DIFFEYV]], 1
; CHECK-NEXT:   ret { <2 x double>, <2 x double> } %[[IVV1]]
; CHECK-NEXT: }
