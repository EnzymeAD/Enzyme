; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

; Same structure as the sin.ll test, only substituted double -> float and f64 -> f32.

; Function Attrs: nounwind readnone uwtable
define float @tester(float %x) {
entry:
  %0 = tail call fast float @llvm.sin.f32(float %x)
  ret float %0
}

define float @test_derivative(float %x) {
entry:
  %0 = tail call float (float (float)*, ...) @__enzyme_autodiff(float (float)* nonnull @tester, float %x)
  ret float %0
}

; Function Attrs: nounwind readnone speculatable
declare float @llvm.cos.f32(float)

; Function Attrs: nounwind readnone speculatable
declare float @llvm.sin.f32(float)

; Function Attrs: nounwind
declare float @__enzyme_autodiff(float (float)*, ...)

; CHECK: define internal { float } @diffetester(float %x, float %differeturn) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call fast float @llvm.cos.f32(float %x)
; CHECK-NEXT:   %1 = fmul fast float %differeturn, %0
; CHECK-NEXT:   %2 = insertvalue { float } undef, float %1, 0
; CHECK-NEXT:   ret { float } %2
