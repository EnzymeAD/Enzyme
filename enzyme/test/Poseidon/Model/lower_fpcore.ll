; RUN: %opt %s %loadPoseidon -load-pass-plugin=%poseidon_test_programs/poseidon-test-model%shlibext -passes=lower-fpcore -S | FileCheck %s

; Herbie's constants lower to the binary64 constants the accuracy model
; evaluates them as, NAN among them, and a comparison to the fcmp with its IEEE
; meaning: != to une, the others to their ordered predicates. LLVM 24 prints a double constant as +qnan,
; +inf or f0x<bits>, earlier versions as 0x<bits>.

!poseidon.test.fpcore = !{!0, !1, !2, !3, !4, !5}

; CHECK-LABEL: define double @fpcore.0(double %x, double %y)
; CHECK-NEXT: entry:
; CHECK-NEXT: ret double {{\+qnan|0x7FF8000000000000}}
!0 = !{!"(NAN.f64)"}
; CHECK-LABEL: define double @fpcore.1(double %x, double %y)
; CHECK-NEXT: entry:
; CHECK-NEXT: ret double {{\+inf|0x7FF0000000000000}}
!1 = !{!"(INFINITY.f64)"}
; CHECK-LABEL: define double @fpcore.2(double %x, double %y)
; CHECK-NEXT: entry:
; CHECK-NEXT: ret double {{f?}}0x400921FB54442D18
!2 = !{!"(PI.f64)"}
; CHECK-LABEL: define double @fpcore.3(double %x, double %y)
; CHECK-NEXT: entry:
; CHECK-NEXT: ret double {{f?}}0x4005BF0A8B145769
!3 = !{!"(E.f64)"}
; CHECK-LABEL: define double @fpcore.4(double %x, double %y)
; CHECK: fcmp une double %x, %y
!4 = !{!"(if (!=.f64 x y) x y)"}
; CHECK-LABEL: define double @fpcore.5(double %x, double %y)
; CHECK: fcmp ole double %x, %y
!5 = !{!"(if (<=.f64 x y) x y)"}
