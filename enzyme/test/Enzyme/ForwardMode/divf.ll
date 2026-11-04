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
; CHECK-NEXT:   %0 = fdiv nnan ninf nsz arcp contract afn float %x, %y
; CHECK-NEXT:   %1 = fmul nnan ninf nsz arcp contract afn float %"y'", %0
; CHECK-NEXT:   %2 = fsub nnan ninf nsz arcp contract afn float %"x'", %1
; CHECK-NEXT:   %3 = fdiv nnan ninf nsz arcp contract afn float %2, %y
; CHECK-NEXT:   ret float %3
; CHECK-NEXT: }
