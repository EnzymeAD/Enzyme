; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -instsimplify -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(instsimplify)" -enzyme-preopt=false -S | FileCheck %s

; Same structure as the div.ll test, only substituted double -> x86_fp80 and adapted the constants 1.0 and 0.0.

; Function Attrs: noinline nounwind readnone uwtable
define x86_fp80 @tester(x86_fp80 %x, x86_fp80 %y) {
entry:
  %0 = fdiv fast x86_fp80 %x, %y
  ret x86_fp80 %0
}

define x86_fp80 @test_derivative(x86_fp80 %x, x86_fp80 %y) {
entry:
  %0 = tail call x86_fp80 (x86_fp80 (x86_fp80, x86_fp80)*, ...) @__enzyme_fwddiff(x86_fp80 (x86_fp80, x86_fp80)* nonnull @tester, x86_fp80 %x, x86_fp80 0xK3FFF8000000000000000, x86_fp80 %y, x86_fp80 0xK00000000000000000000)
  ret x86_fp80 %0
}

; Function Attrs: nounwind
declare x86_fp80 @__enzyme_fwddiff(x86_fp80 (x86_fp80, x86_fp80)*, ...)

; CHECK: define internal {{(dso_local )?}}x86_fp80 @fwddiffetester(x86_fp80 %x, x86_fp80 %"x'", x86_fp80 %y, x86_fp80 %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = fmul fast x86_fp80 %"x'", %y
; CHECK-NEXT:   %1 = fmul fast x86_fp80 %"y'", %x
; CHECK-NEXT:   %2 = fsub fast x86_fp80 %0, %1
; CHECK-NEXT:   %3 = fmul fast x86_fp80 %y, %y
; CHECK-NEXT:   %4 = fdiv fast x86_fp80 %2, %3
; CHECK-NEXT:   ret x86_fp80 %4
; CHECK-NEXT: }
