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
; CHECK-NEXT:   %[[QUOT:.+]] = fdiv fast x86_fp80 %x, %y
; CHECK-NEXT:   %[[DYQ:.+]] = fmul fast x86_fp80 %"y'", %[[QUOT]]
; CHECK-NEXT:   %[[NUM:.+]] = fsub fast x86_fp80 %"x'", %[[DYQ]]
; CHECK-NEXT:   %[[RES:.+]] = fdiv fast x86_fp80 %[[NUM]], %y
; CHECK-NEXT:   ret x86_fp80 %[[RES]]
; CHECK-NEXT: }
