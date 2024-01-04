; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false  -enzyme -mem2reg -early-cse -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,early-cse,instsimplify,%simplifycfg)" -S | FileCheck %s

; Same structure as the div.ll test, only substituted double -> float.

; Function Attrs: noinline nounwind readnone uwtable
define float @tester(float %x, float %y) {
entry:
  %0 = fdiv fast float %x, %y
  ret float %0
}

define float @test_derivative(float %x, float %y) {
entry:
  %0 = tail call float (float (float, float)*, ...) @__enzyme_autodiff(float (float, float)* nonnull @tester, float %x, float %y)
  ret float %0
}

; Function Attrs: nounwind
declare float @__enzyme_autodiff(float (float, float)*, ...)

; CHECK: define internal {{(dso_local )?}}{ float, float } @diffetester(float %x, float %y, float %[[differet:.+]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[diffex:.+]] = fdiv fast float %[[differet]], %y
; CHECK-NEXT:   %[[xdivy:.+]] = fdiv fast float %x, %y
; CHECK-NEXT:   %[[xdivydret:.+]] = fmul fast float %[[diffex]], %[[xdivy]]
; CHECK-NEXT:   %[[mxdivy2:.+]] = {{(fsub fast float 0.000000e\+00,|fneg fast float)}} %[[xdivydret]]
; CHECK-NEXT:   %[[res1:.+]] = insertvalue { float, float } undef, float %[[diffex]], 0
; CHECK-NEXT:   %[[res2:.+]] = insertvalue { float, float } %[[res1:.+]], float %[[mxdivy2]], 1
; CHECK-NEXT:   ret { float, float } %[[res2]]
; CHECK-NEXT: }
