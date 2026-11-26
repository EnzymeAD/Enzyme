; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -early-cse -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,%simplifycfg,early-cse)" -enzyme-preopt=false -S | FileCheck %s

; Lane 1 of the shuffle mask is poison: it reads neither operand, so the
; reverse pass must not propagate its adjoint into %x or %y.

%struct.Gradients = type { [2 x <2 x double>], [2 x <2 x double>] }

declare %struct.Gradients @__enzyme_autodiff(...)

define <3 x double> @tester(<2 x double> %x, <2 x double> %y) {
entry:
  %r = shufflevector <2 x double> %x, <2 x double> %y, <3 x i32> <i32 1, i32 poison, i32 3>
  ret <3 x double> %r
}

define %struct.Gradients @test_derivative(<2 x double> %x, <2 x double> %y) {
entry:
  %0 = tail call %struct.Gradients (...) @__enzyme_autodiff(<3 x double> (<2 x double>, <2 x double>)* nonnull @tester, metadata !"enzyme_width", i64 2, <2 x double> %x, <2 x double> %y)
  ret %struct.Gradients %0
}

; CHECK-LABEL: define internal { [2 x <2 x double>], [2 x <2 x double>] } @diffe2tester(<2 x double> %x, <2 x double> %y, [2 x <3 x double>] %differeturn)
; CHECK:         %[[r0:.+]] = extractvalue [2 x <3 x double>] %differeturn, 0
; CHECK-NEXT:    %[[d00:.+]] = extractelement <3 x double> %[[r0]], i64 0
; CHECK-NEXT:    %[[xp0:.+]] = getelementptr inbounds [2 x <2 x double>], {{.*}}%"x'de", i32 0, i32 0, i32 1
; CHECK-NEXT:    %[[xv0:.+]] = load double, {{.*}}%[[xp0]]
; CHECK-NEXT:    %[[xa0:.+]] = fadd fast double %[[xv0]], %[[d00]]
; CHECK-NEXT:    store double %[[xa0]], {{.*}}%[[xp0]]
; CHECK-NEXT:    %[[r1:.+]] = extractvalue [2 x <3 x double>] %differeturn, 1
; CHECK-NEXT:    %[[d10:.+]] = extractelement <3 x double> %[[r1]], i64 0
; CHECK-NEXT:    %[[xp1:.+]] = getelementptr inbounds [2 x <2 x double>], {{.*}}%"x'de", i32 0, i32 1, i32 1
; CHECK-NEXT:    %[[xv1:.+]] = load double, {{.*}}%[[xp1]]
; CHECK-NEXT:    %[[xa1:.+]] = fadd fast double %[[xv1]], %[[d10]]
; CHECK-NEXT:    store double %[[xa1]], {{.*}}%[[xp1]]
; CHECK-NEXT:    %[[d02:.+]] = extractelement <3 x double> %[[r0]], i64 2
; CHECK-NEXT:    %[[yp0:.+]] = getelementptr inbounds [2 x <2 x double>], {{.*}}%"y'de", i32 0, i32 0, i32 1
; CHECK-NEXT:    %[[yv0:.+]] = load double, {{.*}}%[[yp0]]
; CHECK-NEXT:    %[[ya0:.+]] = fadd fast double %[[yv0]], %[[d02]]
; CHECK-NEXT:    store double %[[ya0]], {{.*}}%[[yp0]]
; CHECK-NEXT:    %[[d12:.+]] = extractelement <3 x double> %[[r1]], i64 2
; CHECK-NEXT:    %[[yp1:.+]] = getelementptr inbounds [2 x <2 x double>], {{.*}}%"y'de", i32 0, i32 1, i32 1
; CHECK-NEXT:    %[[yv1:.+]] = load double, {{.*}}%[[yp1]]
; CHECK-NEXT:    %[[ya1:.+]] = fadd fast double %[[yv1]], %[[d12]]
; CHECK-NEXT:    store double %[[ya1]], {{.*}}%[[yp1]]
; CHECK-NEXT:    %{{.+}} = load [2 x <2 x double>], {{.*}}%"x'de"
