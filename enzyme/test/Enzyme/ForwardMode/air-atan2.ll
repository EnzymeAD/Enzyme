; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -preserve-nvvm -enzyme -early-cse -instcombine -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="preserve-nvvm,enzyme,function(early-cse,instcombine)" -enzyme-preopt=false -S | FileCheck %s

; Metal AIR math intrinsic (air.atan2.f32): not a real LLVM intrinsic, so
; this exercises the new air.atan2.f32/air.atan2.f64 CallPattern entry in
; forward mode.

define float @tester(float %y, float %x) {
entry:
  %call = call float @air.atan2.f32(float %y, float %x)
  ret float %call
}

define float @test_derivative(float %y, float %x) {
entry:
  %0 = tail call float (...) @__enzyme_fwddiff(float (float, float)* nonnull @tester, float %y, float 1.000000e+00, float %x, float 1.000000e+00)
  ret float %0
}

declare float @air.atan2.f32(float, float)

; Function Attrs: nounwind
declare float @__enzyme_fwddiff(...)

; CHECK-LABEL: define internal float @fwddiffetester(
; CHECK-NEXT: entry:
; CHECK-DAG:   %[[a3:.+]] = fmul float %"y'", %x
; CHECK-DAG:    %[[a1:.+]] = fmul float %x, %x
; CHECK-DAG:    %[[a0:.+]] = fmul float %y, %y
; CHECK-DAG:   %[[a2:.+]] = fadd float %[[a1]], %[[a0]]
; CHECK-DAG:   %[[a7:.+]] = fdiv float %[[a3]], %[[a2]]
; CHECK-DAG:   %[[a4:.+]] = fmul float %"x'", %y
; CHECK-DAG:   %[[a8:.+]] = fdiv float %[[a4]], %[[a2]]
; CHECK-DAG:   %[[a5:.+]] = fsub float %[[a7]], %[[a8]]
; CHECK-NEXT:   ret float %[[a5]]
; CHECK-NEXT: }
