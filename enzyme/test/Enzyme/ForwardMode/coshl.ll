; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define x86_fp80 @tester(x86_fp80 %x) {
entry:
  %0 = tail call fast x86_fp80 @coshl(x86_fp80 %x)
  ret x86_fp80 %0
}

define x86_fp80 @test_derivative(x86_fp80 %x) {
entry:
  %0 = tail call x86_fp80 (x86_fp80 (x86_fp80)*, ...) @__enzyme_fwddiff(x86_fp80 (x86_fp80)* nonnull @tester, x86_fp80 %x, x86_fp80 0xK3FFF8000000000000000)
  ret x86_fp80 %0
}

; Function Attrs: nounwind readnone speculatable
declare x86_fp80 @coshl(x86_fp80)

; Function Attrs: nounwind
declare x86_fp80 @__enzyme_fwddiff(x86_fp80 (x86_fp80)*, ...)

; CHECK: define internal x86_fp80 @fwddiffetester(x86_fp80 %x, x86_fp80 %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call fast x86_fp80 @sinhl(x86_fp80 %x)
; CHECK-NEXT:   %1 = fmul fast x86_fp80 %"x'", %0
; CHECK-NEXT:   ret x86_fp80 %1
; CHECK-NEXT: }
