; RUN: %opt %s %loadPoseidon -load-pass-plugin=%poseidon_test_programs/poseidon-test-model%shlibext -passes=evaluate-fpcore -disable-output | FileCheck %s

; The MPFR reference and the binary64 evaluation of the accuracy model give a
; Herbie expression the same value: FPCore's E is e and PI is pi.

!poseidon.test.fpcore = !{!0, !1, !2}

; CHECK: (E.f64): reference 2.7182818284590451 binary64 2.7182818284590451
!0 = !{!"(E.f64)"}
; CHECK-NEXT: (log.f64 (E.f64)): reference 1 binary64 1
!1 = !{!"(log.f64 (E.f64))"}
; CHECK-NEXT: (PI.f64): reference 3.1415926535897931 binary64 3.1415926535897931
!2 = !{!"(PI.f64)"}
