; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

; Same structure as the sinpi.ll test, only substituted double -> x86_fp80. The derivative multiplies by pi rounded to
; x86_fp80, 0xK4000C90FDAA22168C235.

; Function Attrs: nounwind readnone uwtable
define x86_fp80 @tester(x86_fp80 %x) {
entry:
  %0 = tail call x86_fp80 @sinpil(x86_fp80 %x)
  ret x86_fp80 %0
}

define x86_fp80 @test_derivative(x86_fp80 %x) {
entry:
  %0 = tail call x86_fp80 (...) @__enzyme_fwddiff(x86_fp80 (x86_fp80)* nonnull @tester, x86_fp80 %x, x86_fp80 0xK3FFF8000000000000000)
  ret x86_fp80 %0
}

declare x86_fp80 @sinpil(x86_fp80) readnone

; Function Attrs: nounwind
declare x86_fp80 @__enzyme_fwddiff(...)

; CHECK: define internal x86_fp80 @fwddiffetester(x86_fp80 %x, x86_fp80 %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = fadd fast x86_fp80 %x, 0xK3FFE8000000000000000
; CHECK-NEXT:   %1 = call fast x86_fp80 @sinpil(x86_fp80 %0)
; CHECK-NEXT:   %2 = fmul fast x86_fp80 %1, %"x'"
; CHECK-NEXT:   %3 = fmul fast x86_fp80 0xK4000C90FDAA22168C235, %2
; CHECK-NEXT:   ret x86_fp80 %3
; CHECK-NEXT: }
