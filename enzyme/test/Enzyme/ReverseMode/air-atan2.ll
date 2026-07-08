; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -mem2reg -sroa -early-cse -instsimplify -simplifycfg -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="preserve-nvvm,enzyme,function(mem2reg,sroa,early-cse,instsimplify,%simplifycfg,adce)" -enzyme-preopt=false -S | FileCheck %s

; Metal AIR math intrinsic (air.atan2.f32): not a real LLVM intrinsic, so
; this exercises the new air.atan2.f32/air.atan2.f64 CallPattern entry.

define float @tester(float %y, float %x) {
entry:
  %call = call float @air.atan2.f32(float %y, float %x)
  ret float %call
}

define float @test_derivative(float %y, float %x) {
entry:
  %0 = tail call float (...) @__enzyme_autodiff(float (float, float)* nonnull @tester, float %y, float %x)
  ret float %0
}

declare float @air.atan2.f32(float, float)

; Function Attrs: nounwind
declare float @__enzyme_autodiff(...)

; CHECK: define internal { float, float } @diffetester(float %y, float %x, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-DAG:    %[[a0:.+]] = fmul fast float %y, %y
; CHECK-DAG:    %[[a1:.+]] = fmul fast float %x, %x
; CHECK-DAG:   %[[a2:.+]] = fadd fast float %[[a1]], %[[a0]]
; CHECK-DAG:   %[[a3:.+]] = fmul fast float %differeturn, %x
; CHECK-DAG:   %[[a4:.+]] = fdiv fast float %[[a3]], %[[a2]]
; CHECK-DAG:   %[[a5:.+]] = fmul fast float %differeturn, %y
; CHECK-DAG:   %[[a6:.+]] = fdiv fast float %[[a5]], %[[a2]]
; CHECK-DAG:   %[[a7:.+]] = {{(fneg fast float)|(fsub fast float (-)?0.000000e\+00,)}} %[[a6]]
; CHECK-DAG:   %[[a8:.+]] = insertvalue { float, float } undef, float %[[a4]], 0
; CHECK-DAG:   %[[a9:.+]] = insertvalue { float, float } %[[a8]], float %[[a7]], 1
; CHECK-DAG:   ret { float, float } %[[a9]]
; CHECK-NEXT: }
