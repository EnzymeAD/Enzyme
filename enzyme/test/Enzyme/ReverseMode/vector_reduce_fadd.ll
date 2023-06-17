; RUN: if [ %llvmver -ge 12 ] && [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 12 ]; then %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi

define float @tester(float %start_value, <4 x float> %input) {
entry:
  %ord = call float @llvm.vector.reduce.fadd.v4f32(float %start_value, <4 x float> %input)
  ret float %ord
}

define float @test_derivative(float %start_value, <4 x float> %input) {
entry:
  %0 = tail call float (float (float, <4 x float>)*, ...) @__enzyme_autodiff(float (float, <4 x float>)* nonnull @tester, float %start_value, <4 x float> %input)
  ret float %0
}

declare float @llvm.vector.reduce.fadd.v4f32(float, <4 x float>)

; Function Attrs: nounwind
declare float @__enzyme_autodiff(float (float, <4 x float>)*, ...)


; CHECK: define internal {{(dso_local )?}}{ float, <4 x float> } @diffetester(float %start_value, <4 x float> %input, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[i0:.+]] = insertelement <4 x float> {{(undef|poison)}}, float %differeturn, {{(i64|i32)}} 0
; CHECK-NEXT:   %[[i1:.+]] = shufflevector <4 x float> %[[i0]], <4 x float> {{(undef|poison)}}, <4 x i32> zeroinitializer
; CHECK-NEXT:   %[[i2:.+]] = insertvalue { float, <4 x float> } undef, float %differeturn, 0
; CHECK-NEXT:   %[[i3:.+]] = insertvalue { float, <4 x float> } %[[i2]], <4 x float> %[[i1]], 1
; CHECK-NEXT:   ret { float, <4 x float> } %[[i3]]
; CHECK-NEXT: }
