; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -preserve-nvvm -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="preserve-nvvm,enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

; Metal AIR math intrinsic (air.tan.f32): tan is not a real LLVM intrinsic,
; so this exercises the new air.tan.f32/air.tan.f64 CallPattern entry in
; InstructionDerivatives.td rather than the intrinsic-redirect path.

; Function Attrs: nounwind readnone uwtable
define float @tester(float %x) {
entry:
  %0 = tail call fast float @air.tan.f32(float %x)
  ret float %0
}

define float @test_derivative(float %x) {
entry:
  %0 = tail call float (float (float)*, ...) @__enzyme_autodiff(float (float)* nonnull @tester, float %x)
  ret float %0
}

; Function Attrs: nounwind readnone speculatable
declare float @air.tan.f32(float)

; Function Attrs: nounwind
declare float @__enzyme_autodiff(float (float)*, ...)

; CHECK: define internal { float } @diffetester(float %x, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call fast float @air.tan.f32(float %x)
; CHECK-NEXT:   %1 = fmul fast float %0, %0
; CHECK-NEXT:   %2 = fadd fast float 1.000000e+00, %1
; CHECK-NEXT:   %3 = fmul fast float %differeturn, %2
; CHECK-NEXT:   %4 = insertvalue { float } undef, float %3, 0
; CHECK-NEXT:   ret { float } %4
; CHECK-NEXT: }
