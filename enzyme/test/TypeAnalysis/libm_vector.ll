; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=tester -S -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=tester -S -o /dev/null | FileCheck %s

declare <2 x float> @__mulsc3(float, float, float, float)
declare <2 x float> @__divsc3(float, float, float, float)
declare <2 x float> @Faddeeva_erf(<2 x float>, float)
declare float @cabsf(<2 x float>)
declare <2 x i32> @ilogbf(<2 x float>)
declare <2 x float> @scalbnf(<2 x float>, <2 x i32>)

define void @tester(float %a, float %b, float %c, float %d, <2 x float> %z, <2 x i32> %n) {
entry:
  %m = call <2 x float> @__mulsc3(float %a, float %b, float %c, float %d)
  %q = call <2 x float> @__divsc3(float %a, float %b, float %c, float %d)
  %e = call <2 x float> @Faddeeva_erf(<2 x float> %z, float %a)
  %r = call float @cabsf(<2 x float> %z)
  %il = call <2 x i32> @ilogbf(<2 x float> %z)
  %sc = call <2 x float> @scalbnf(<2 x float> %z, <2 x i32> %n)
  ret void
}

; CHECK: tester - {} |{[-1]:Float@float}:{} {[-1]:Float@float}:{} {[-1]:Float@float}:{} {[-1]:Float@float}:{} {[-1]:Float@float}:{} {[-1]:Integer}:{}
; CHECK-NEXT: float %a: {[-1]:Float@float}
; CHECK-NEXT: float %b: {[-1]:Float@float}
; CHECK-NEXT: float %c: {[-1]:Float@float}
; CHECK-NEXT: float %d: {[-1]:Float@float}
; CHECK-NEXT: <2 x float> %z: {[-1]:Float@float}
; CHECK-NEXT: <2 x i32> %n: {[-1]:Integer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %m = call <2 x float> @__mulsc3(float %a, float %b, float %c, float %d): {[-1]:Float@float}
; CHECK-NEXT:   %q = call <2 x float> @__divsc3(float %a, float %b, float %c, float %d): {[-1]:Float@float}
; CHECK-NEXT:   %e = call <2 x float> @Faddeeva_erf(<2 x float> %z, float %a): {[-1]:Float@float}
; CHECK-NEXT:   %r = call float @cabsf(<2 x float> %z): {[-1]:Float@float}
; CHECK-NEXT:   %il = call <2 x i32> @ilogbf(<2 x float> %z): {[-1]:Integer}
; CHECK-NEXT:   %sc = call <2 x float> @scalbnf(<2 x float> %z, <2 x i32> %n): {[-1]:Float@float}
; CHECK-NEXT:   ret void: {}
