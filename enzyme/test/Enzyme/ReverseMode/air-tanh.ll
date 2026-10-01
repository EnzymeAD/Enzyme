; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -preserve-nvvm -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="preserve-nvvm,enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

; Metal AIR hyperbolic intrinsic (air.tanh.f32): exercises the new
; air.tanh.f32 CallPattern entry, whose companion call must be the AIR name
; air.cosh.f32 (not "coshf"), since Metal's AIR compiler cannot resolve
; libm symbols.

; Function Attrs: nounwind readnone uwtable
define float @tester(float %x) {
entry:
  %0 = tail call fast float @air.tanh.f32(float %x)
  ret float %0
}

define float @test_derivative(float %x) {
entry:
  %0 = tail call float (float (float)*, ...) @__enzyme_autodiff(float (float)* nonnull @tester, float %x)
  ret float %0
}

; Function Attrs: nounwind readnone speculatable
declare float @air.tanh.f32(float)

; Function Attrs: nounwind
declare float @__enzyme_autodiff(float (float)*, ...)

; CHECK: define internal { float } @diffetester(float %x, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call fast float @air.cosh.f32(float %x)
; CHECK-NEXT:   %1 = fmul fast float %0, %0
; CHECK-NEXT:   %2 = fdiv fast float %differeturn, %1
; CHECK-NEXT:   %3 = insertvalue { float } undef, float %2, 0
; CHECK-NEXT:   ret { float } %3
; CHECK-NEXT: }
