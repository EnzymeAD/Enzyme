; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -instsimplify -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(instsimplify)" -enzyme-preopt=false -S | FileCheck %s

; Same structure as the div.ll test, only substituted double -> float.

; Function Attrs: noinline nounwind readnone uwtable
define float @tester(float %x, float %y) {
entry:
  %0 = fdiv fast float %x, %y
  ret float %0
}

define float @test_derivative(float %x, float %y) {
entry:
  %0 = tail call float (float (float, float)*, ...) @__enzyme_fwddiff(float (float, float)* nonnull @tester, float %x, float 1.0, float %y, float 0.0)
  ret float %0
}

; Function Attrs: nounwind
declare float @__enzyme_fwddiff(float (float, float)*, ...)

; CHECK: define internal {{(dso_local )?}}float @fwddiffetester(float %x, float %"x'", float %y, float %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[QUOT:.+]] = fdiv fast float %x, %y
; CHECK-NEXT:   %[[DYQ:.+]] = fmul fast float %"y'", %[[QUOT]]
; CHECK-NEXT:   %[[NUM:.+]] = fsub fast float %"x'", %[[DYQ]]
; CHECK-NEXT:   %[[RES:.+]] = fdiv fast float %[[NUM]], %y
; CHECK-NEXT:   ret float %[[RES]]
; CHECK-NEXT: }
