; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -preserve-nvvm -enzyme -mem2reg -sroa -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="preserve-nvvm,enzyme,function(mem2reg,sroa,instsimplify,%simplifycfg)" -S | FileCheck %s

; Metal AIR math intrinsic (air.cbrt.f32): not a real LLVM intrinsic, so
; this exercises the new air.cbrt.f32/air.cbrt.f64 CallPattern entry.

; Function Attrs: nounwind readnone uwtable
define float @tester(float %x) {
entry:
  %call = call float @air.cbrt.f32(float %x)
  ret float %call
}

define float @test_derivative(float %x) {
entry:
  %0 = tail call float (float (float)*, ...) @__enzyme_autodiff(float (float)* nonnull @tester, float %x)
  ret float %0
}

declare float @air.cbrt.f32(float)

; Function Attrs: nounwind
declare float @__enzyme_autodiff(float (float)*, ...)

; CHECK: define internal { float } @diffetester(float %x, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call float @air.cbrt.f32(float %x)
; CHECK-DAG:    [[REG1:%[0-9]+]] = fmul float 3.000000e+00, %x
; CHECK-DAG:    [[REG2:%[0-9]+]] = fmul float %differeturn, %0
; CHECK-NEXT:   %3 = fdiv float [[REG2]], [[REG1]]
; CHECK-NEXT:   %4 = fadd float 0.000000e+00, %3
; CHECK-NEXT:   %5 = insertvalue { float } undef, float %4, 0
; CHECK-NEXT:   ret { float } %5
; CHECK-NEXT: }
