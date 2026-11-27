; RUN: rm -rf %t && split-file %s %t
; RUN: %opt %t/evaluate.ll %loadPoseidon -load-pass-plugin=%poseidon_test_programs/poseidon-test-model%shlibext -passes=evaluate-fpcore -disable-output | FileCheck --check-prefix=EVAL %s
; RUN: %opt %t/lower.ll %loadPoseidon -load-pass-plugin=%poseidon_test_programs/poseidon-test-model%shlibext -passes=lower-fpcore -S | FileCheck --check-prefix=LOWER %s

;--- evaluate.ll
; The MPFR reference and the binary64 evaluation agree on Herbie's constants
; and math functions (binary64 is libm's, to the last place); comparisons with
; a NaN are IEEE's (all false but !=), and so are logical operations over them.

!poseidon.test.fpcore = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9, !10, !11, !12, !13, !14}

; EVAL: (E.f64): reference 2.7182818284590451 binary64 2.7182818284590451
!0 = !{!"(E.f64)"}
; EVAL-NEXT: (log.f64 (E.f64)): reference 1 binary64 1
!1 = !{!"(log.f64 (E.f64))"}
; EVAL-NEXT: (PI.f64): reference 3.1415926535897931 binary64 3.1415926535897931
!2 = !{!"(PI.f64)"}
; EVAL-NEXT: (erf.f64 #s(literal 1/2 binary64)): reference 0.52049987781304652 binary64 0.5204998778130{{[0-9]+}}
!3 = !{!"(erf.f64 #s(literal 1/2 binary64))"}
; EVAL-NEXT: (lgamma.f64 #s(literal -5/2 binary64)): reference -0.056243716497674054 binary64 -0.05624371649767{{[0-9]+}}
!4 = !{!"(lgamma.f64 #s(literal -5/2 binary64))"}
; EVAL-NEXT: (tgamma.f64 #s(literal 9/2 binary64)): reference 11.631728396567448 binary64 11.63172839656{{[0-9]+}}
!5 = !{!"(tgamma.f64 #s(literal 9/2 binary64))"}
; EVAL-NEXT: (if (==.f64 (NAN.f64) #s(literal 1 binary64)) #s(literal 1 binary64) #s(literal 0 binary64)): reference 0 binary64 0
!6 = !{!"(if (==.f64 (NAN.f64) #s(literal 1 binary64)) #s(literal 1 binary64) #s(literal 0 binary64))"}
; EVAL-NEXT: (if (!=.f64 (NAN.f64) #s(literal 1 binary64)) #s(literal 1 binary64) #s(literal 0 binary64)): reference 1 binary64 1
!7 = !{!"(if (!=.f64 (NAN.f64) #s(literal 1 binary64)) #s(literal 1 binary64) #s(literal 0 binary64))"}
; EVAL-NEXT: (if (<.f64 (NAN.f64) #s(literal 1 binary64)) #s(literal 1 binary64) #s(literal 0 binary64)): reference 0 binary64 0
!8 = !{!"(if (<.f64 (NAN.f64) #s(literal 1 binary64)) #s(literal 1 binary64) #s(literal 0 binary64))"}
; EVAL-NEXT: (if (>.f64 (NAN.f64) #s(literal 1 binary64)) #s(literal 1 binary64) #s(literal 0 binary64)): reference 0 binary64 0
!9 = !{!"(if (>.f64 (NAN.f64) #s(literal 1 binary64)) #s(literal 1 binary64) #s(literal 0 binary64))"}
; EVAL-NEXT: (if (<=.f64 (NAN.f64) #s(literal 1 binary64)) #s(literal 1 binary64) #s(literal 0 binary64)): reference 0 binary64 0
!10 = !{!"(if (<=.f64 (NAN.f64) #s(literal 1 binary64)) #s(literal 1 binary64) #s(literal 0 binary64))"}
; EVAL-NEXT: (if (>=.f64 (NAN.f64) #s(literal 1 binary64)) #s(literal 1 binary64) #s(literal 0 binary64)): reference 0 binary64 0
!11 = !{!"(if (>=.f64 (NAN.f64) #s(literal 1 binary64)) #s(literal 1 binary64) #s(literal 0 binary64))"}
; EVAL-NEXT: (if (and (!=.f64 (NAN.f64) (NAN.f64)) (not (==.f64 (NAN.f64) (NAN.f64)))) #s(literal 1 binary64) #s(literal 0 binary64)): reference 1 binary64 1
!12 = !{!"(if (and (!=.f64 (NAN.f64) (NAN.f64)) (not (==.f64 (NAN.f64) (NAN.f64)))) #s(literal 1 binary64) #s(literal 0 binary64))"}
; EVAL-NEXT: (if (or (<=.f64 (NAN.f64) (NAN.f64)) (>=.f64 (NAN.f64) (NAN.f64))) #s(literal 1 binary64) #s(literal 0 binary64)): reference 0 binary64 0
!13 = !{!"(if (or (<=.f64 (NAN.f64) (NAN.f64)) (>=.f64 (NAN.f64) (NAN.f64))) #s(literal 1 binary64) #s(literal 0 binary64))"}
; EVAL-NEXT: (if (not (<.f64 (NAN.f64) #s(literal 1 binary64))) #s(literal 1 binary64) #s(literal 0 binary64)): reference 1 binary64 1
!14 = !{!"(if (not (<.f64 (NAN.f64) #s(literal 1 binary64))) #s(literal 1 binary64) #s(literal 0 binary64))"}

;--- lower.ll
; Herbie's constants lower to their binary64 values and a comparison to its
; IEEE fcmp (!= une, others ordered). LLVM 24 prints +qnan, +inf, f0x<bits>.

!poseidon.test.fpcore = !{!0, !1, !2, !3, !4, !5}

; LOWER-LABEL: define double @fpcore.0(double %x, double %y)
; LOWER-NEXT: entry:
; LOWER-NEXT: ret double {{\+qnan|0x7FF8000000000000}}
!0 = !{!"(NAN.f64)"}
; LOWER-LABEL: define double @fpcore.1(double %x, double %y)
; LOWER-NEXT: entry:
; LOWER-NEXT: ret double {{\+inf|0x7FF0000000000000}}
!1 = !{!"(INFINITY.f64)"}
; LOWER-LABEL: define double @fpcore.2(double %x, double %y)
; LOWER-NEXT: entry:
; LOWER-NEXT: ret double {{f?}}0x400921FB54442D18
!2 = !{!"(PI.f64)"}
; LOWER-LABEL: define double @fpcore.3(double %x, double %y)
; LOWER-NEXT: entry:
; LOWER-NEXT: ret double {{f?}}0x4005BF0A8B145769
!3 = !{!"(E.f64)"}
; LOWER-LABEL: define double @fpcore.4(double %x, double %y)
; LOWER: fcmp une double %x, %y
!4 = !{!"(if (!=.f64 x y) x y)"}
; LOWER-LABEL: define double @fpcore.5(double %x, double %y)
; LOWER: fcmp ole double %x, %y
!5 = !{!"(if (<=.f64 x y) x y)"}
