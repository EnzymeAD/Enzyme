; RUN: if [ %llvmver -lt 16 ] && [ %llvmver -ge 15]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -early-cse -simplifycfg -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 15]; then %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,early-cse,%simplifycfg)" -S | FileCheck %s; fi



; Function Attrs: noinline nounwind readnone uwtable
define float @tester(float %x, float %y) {
entry:
  %0 = tail call float @llvm.maximum(float %x, float %y)
  ret float %0
}

define float @test_derivative(float %x, float %y) {
entry:
  %0 = tail call float (float (float, float)*, ...) @__enzyme_fwddiff(float (float, float)* nonnull @tester, float %x, float 1.0, float %y, float 1.0)
  ret float %0
}

; Function Attrs: nounwind readnone speculatable
declare float @llvm.maximum(float, float)

; Function Attrs: nounwind
declare float @__enzyme_fwddiff(float (float, float)*, ...)

; CHECK: define internal {{(dso_local )?}}float @fwddiffetester(float %x, float %"x'", float %y, float %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[lt:.+]] = fcmp fast olt float %x, %y
; CHECK-NEXT:   %[[eq:.+]] = fcmp fast oeq float %x, %y
; CHECK-NEXT:   %[[sx:.+]] = call fast float @llvm.copysign.f32(float 1.000000e+00, float %x)
; CHECK-NEXT:   %[[sy:.+]] = call fast float @llvm.copysign.f32(float 1.000000e+00, float %y)
; CHECK-NEXT:   %[[slt:.+]] = fcmp fast olt float %[[sx]], %[[sy]]
; CHECK-NEXT:   %[[tie:.+]] = and i1 %[[eq]], %[[slt]]
; CHECK-NEXT:   %[[cmp:.+]] = or i1 %[[lt]], %[[tie]]
; CHECK-NEXT:   %[[res:.+]] = select {{(fast )?}}i1 %[[cmp]], float %"y'", float %"x'"
; CHECK-NEXT:   ret float %[[res]]
; CHECK-NEXT: }

