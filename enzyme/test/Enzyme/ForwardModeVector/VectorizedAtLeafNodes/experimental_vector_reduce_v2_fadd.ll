; RUN: if [ %llvmver -ge 9 ] && [ %llvmver -le 11 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -enzyme-vectorize-at-leaf-nodes -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi

; Function Attrs: nounwind
declare <2 x float> @__enzyme_fwddiff(float (float, <4 x float>)*, ...)

define float @tester(float %start_value, <4 x float> %input) {
entry:
  %ord = call float @llvm.experimental.vector.reduce.v2.fadd.f32.v4f32(float %start_value, <4 x float> %input)
  ret float %ord
}

define <2 x float> @test_derivative(float %start_value, <4 x float> %input) {
entry:
  %0 = tail call <2 x float> (float (float, <4 x float>)*, ...) @__enzyme_fwddiff(float (float, <4 x float>)* nonnull @tester, metadata !"enzyme_width", i64 2, float %start_value, <2 x float> <float 1.0, float 2.0>, <4 x float> %input, <8 x float> <float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0>)
  ret <2 x float> %0
}

declare float @llvm.experimental.vector.reduce.v2.fadd.f32.v4f32(float, <4 x float>)


; CHECK: define internal <2 x float> @fwddiffe2tester(float %start_value, <2 x float> %"start_value'", <4 x float> %input, <8 x float> %"input'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractelement <2 x float> %"start_value'", i64 0
; CHECK-NEXT:   %"input'.subvector.0" = shufflevector <8 x float> %"input'", <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; CHECK-NEXT:   %1 = call fast float @llvm.vector.reduce.fadd.v4f32(float %0, <4 x float> %"input'.subvector.0")
; CHECK-NEXT:   %2 = insertelement <2 x float> undef, float %1, i32 0
; CHECK-NEXT:   %3 = extractelement <2 x float> %"start_value'", i64 1
; CHECK-NEXT:   %"input'.subvector.1" = shufflevector <8 x float> %"input'", <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
; CHECK-NEXT:   %4 = call fast float @llvm.vector.reduce.fadd.v4f32(float %3, <4 x float> %"input'.subvector.1")
; CHECK-NEXT:   %5 = insertelement <2 x float> %2, float %4, i32 1
; CHECK-NEXT:   ret <2 x float> %5
; CHECK-NEXT: }