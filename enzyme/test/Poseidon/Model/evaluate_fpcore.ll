; RUN: %opt %s %loadPoseidon -load-pass-plugin=%poseidon_test_programs/poseidon-test-model%shlibext -passes=evaluate-fpcore -disable-output | FileCheck %s

; The MPFR reference and the binary64 evaluation of the accuracy model give a
; Herbie expression the same value: FPCore's E is e and PI is pi. Both evaluate
; every math function Herbie's platforms offer, erf, lgamma (the logarithm of
; |Gamma|, here of Gamma(-2.5) < 0) and tgamma among them; binary64 is libm's
; value, which may differ from the reference in its last place.

!poseidon.test.fpcore = !{!0, !1, !2, !3, !4, !5}

; CHECK: (E.f64): reference 2.7182818284590451 binary64 2.7182818284590451
!0 = !{!"(E.f64)"}
; CHECK-NEXT: (log.f64 (E.f64)): reference 1 binary64 1
!1 = !{!"(log.f64 (E.f64))"}
; CHECK-NEXT: (PI.f64): reference 3.1415926535897931 binary64 3.1415926535897931
!2 = !{!"(PI.f64)"}
; CHECK-NEXT: (erf.f64 #s(literal 1/2 binary64)): reference 0.52049987781304652 binary64 0.5204998778130{{[0-9]+}}
!3 = !{!"(erf.f64 #s(literal 1/2 binary64))"}
; CHECK-NEXT: (lgamma.f64 #s(literal -5/2 binary64)): reference -0.056243716497674054 binary64 -0.05624371649767{{[0-9]+}}
!4 = !{!"(lgamma.f64 #s(literal -5/2 binary64))"}
; CHECK-NEXT: (tgamma.f64 #s(literal 9/2 binary64)): reference 11.631728396567448 binary64 11.63172839656{{[0-9]+}}
!5 = !{!"(tgamma.f64 #s(literal 9/2 binary64))"}
