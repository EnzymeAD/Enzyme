; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=tester -S -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=tester -S -o /dev/null | FileCheck %s

declare { double, double } @csin(double, double)
declare <2 x float> @csinf(<2 x float>)
declare float @cabsf(<2 x float>)
declare { double, double } @cpow([2 x double], [2 x double])

define void @tester(double %a, double %b, <2 x float> %c, [2 x double] %d, [2 x double] %e) {
entry:
  %r1 = call { double, double } @csin(double %a, double %b)
  %r2 = call <2 x float> @csinf(<2 x float> %c)
  %r3 = call float @cabsf(<2 x float> %r2)
  %r4 = call { double, double } @cpow([2 x double] %d, [2 x double] %e)
  ret void
}

; CHECK: tester
; CHECK-NEXT: double %a: {[-1]:Float@double}
; CHECK-NEXT: double %b: {[-1]:Float@double}
; CHECK-NEXT: <2 x float> %c: {[-1]:Float@float}
; CHECK-NEXT: [2 x double] %d: {[-1]:Float@double}
; CHECK-NEXT: [2 x double] %e: {[-1]:Float@double}
; CHECK-NEXT: entry
; CHECK-NEXT:   %r1 = call { double, double } @csin(double %a, double %b): {[-1]:Float@double}
; CHECK-NEXT:   %r2 = call <2 x float> @csinf(<2 x float> %c): {[-1]:Float@float}
; CHECK-NEXT:   %r3 = call float @cabsf(<2 x float> %r2): {[-1]:Float@float}
; CHECK-NEXT:   %r4 = call { double, double } @cpow([2 x double] %d, [2 x double] %e): {[-1]:Float@double}
