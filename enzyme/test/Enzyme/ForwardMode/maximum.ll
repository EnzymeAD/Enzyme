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
; CHECK-NEXT:   %0 = fcmp fast ogt float %x, %y
; CHECK-NEXT:   %1 = select {{(fast )?}}i1 %0, float %"x'", float %"y'"
; CHECK-NEXT:   ret float %1
; CHECK-NEXT: }

