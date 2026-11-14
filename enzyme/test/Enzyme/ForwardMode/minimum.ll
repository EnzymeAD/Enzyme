; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,early-cse,%simplifycfg)" -S | FileCheck %s; fi

declare double @llvm.minimum.f64(double, double)
declare <2 x double> @llvm.minimum.v2f64(<2 x double>, <2 x double>)
declare double @__enzyme_fwddiff(...)
declare { double, double } @__enzyme_fwddiff_w2(...)
declare <2 x double> @__enzyme_fwddiff_v2(...)

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

define double @test_derivative(double %x, double %y) {
entry:
  %0 = tail call double (...) @__enzyme_fwddiff(ptr nonnull @tester, double %x, double 1.0, double %y, double 1.0)
  ret double %0
}

define { double, double } @test_derivative_w2(double %x, double %y) {
entry:
  %0 = tail call { double, double } (...) @__enzyme_fwddiff_w2(ptr nonnull @tester, metadata !"enzyme_width", i64 2, double %x, double 1.0, double 0.0, double %y, double 0.0, double 1.0)
  ret { double, double } %0
}

define <2 x double> @test_derivative_vec(<2 x double> %x, <2 x double> %dx, <2 x double> %y, <2 x double> %dy) {
entry:
  %0 = tail call <2 x double> (...) @__enzyme_fwddiff_v2(ptr nonnull @tester_vec, <2 x double> %x, <2 x double> %dx, <2 x double> %y, <2 x double> %dy)
  ret <2 x double> %0
}

; CHECK: define internal {{(dso_local )?}}double @fwddiffetester(double %x, double %"x'", double %y, double %"y'")
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
; CHECK-NEXT:   %[[CMP:.+]] = and i1 %[[ORD]], %[[OR]]
; CHECK-NEXT:   %[[RES:.+]] = select {{(fast )?}}i1 %[[CMP]], double %"x'", double %"y'"
; CHECK-NEXT:   ret double %[[RES]]
; CHECK-NEXT: }

; CHECK: define internal {{(dso_local )?}}[2 x double] @fwddiffe2tester(double %x, [2 x double] %"x'", double %y, [2 x double] %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[ORD2:.+]] = fcmp ord double %y, %y
; CHECK-NEXT:   %[[ULT2:.+]] = fcmp ult double %x, %y
; CHECK-NEXT:   %[[OEQ2:.+]] = fcmp oeq double %x, %y
; CHECK-NEXT:   %[[XI2:.+]] = bitcast double %x to i64
; CHECK-NEXT:   %[[YI2:.+]] = bitcast double %y to i64
; CHECK-NEXT:   %[[NOTYI2:.+]] = xor i64 %[[YI2]], -1
; CHECK-NEXT:   %[[MASK2:.+]] = and i64 %[[XI2]], %[[NOTYI2]]
; CHECK-NEXT:   %[[SIGN2:.+]] = icmp slt i64 %[[MASK2]], 0
; CHECK-NEXT:   %[[NPZ2:.+]] = and i1 %[[OEQ2]], %[[SIGN2]]
; CHECK-NEXT:   %[[OR2:.+]] = or i1 %[[ULT2]], %[[NPZ2]]
; CHECK-NEXT:   %[[CMP2:.+]] = and i1 %[[ORD2]], %[[OR2]]
; CHECK-DAG:   %[[EX0:.+]] = extractvalue [2 x double] %"x'", 0
; CHECK-DAG:   %[[EY0:.+]] = extractvalue [2 x double] %"y'", 0
; CHECK-DAG:   %[[S0:.+]] = select {{(fast )?}}i1 %[[CMP2]], double %[[EX0]], double %[[EY0]]
; CHECK-DAG:   %[[IV0:.+]] = insertvalue [2 x double] undef, double %[[S0]], 0
; CHECK-DAG:   %[[EX1:.+]] = extractvalue [2 x double] %"x'", 1
; CHECK-DAG:   %[[EY1:.+]] = extractvalue [2 x double] %"y'", 1
; CHECK-DAG:   %[[S1:.+]] = select {{(fast )?}}i1 %[[CMP2]], double %[[EX1]], double %[[EY1]]
; CHECK-DAG:   %[[IV1:.+]] = insertvalue [2 x double] %[[IV0]], double %[[S1]], 1
; CHECK-NEXT:   ret [2 x double] %[[IV1]]
; CHECK-NEXT: }

; CHECK: define internal {{(dso_local )?}}<2 x double> @fwddiffetester_vec(<2 x double> %x, <2 x double> %"x'", <2 x double> %y, <2 x double> %"y'")
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
; CHECK-NEXT:   %[[RESV:.+]] = select {{(fast )?}}<2 x i1> %[[CMPV]], <2 x double> %"x'", <2 x double> %"y'"
; CHECK-NEXT:   ret <2 x double> %[[RESV]]
; CHECK-NEXT: }
