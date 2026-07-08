; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -preserve-nvvm -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="preserve-nvvm,enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

; Metal AIR math intrinsic (air.pow.f32) resolved via isMemFreeLibMFunction's
; air. prefix stripping in TypeAnalysis.h to Intrinsic::pow. The self-
; referencing "d/dx x^y" term calls back the original air.pow.f32 (SameFunc
; preserves whatever name matched), while the "d/dy x^y" term's cross-
; function log companion becomes a genuine llvm.log.f32 intrinsic call.

; Function Attrs: noinline nounwind readnone uwtable
define float @tester(float %x, float %y) {
entry:
  %0 = tail call fast float @air.pow.f32(float %x, float %y)
  ret float %0
}

define float @test_derivative(float %x, float %y) {
entry:
  %0 = tail call float (float (float, float)*, ...) @__enzyme_autodiff(float (float, float)* nonnull @tester, float %x, float %y)
  ret float %0
}

; Function Attrs: nounwind readnone speculatable
declare float @air.pow.f32(float, float)

; Function Attrs: nounwind
declare float @__enzyme_autodiff(float (float, float)*, ...)

; CHECK: define internal {{(dso_local )?}}{ float, float } @diffetester(float %x, float %y, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[ym1:.+]] = fsub fast float %y, 1.000000e+00
; CHECK-NEXT:   %[[newpow:.+]] = call fast float @air.pow.f32(float %x, float %[[ym1]])
; CHECK-NEXT:   %[[newpowdret:.+]] = fmul fast float %y, %[[newpow]]
; CHECK-NEXT:   %[[dx:.+]] = fmul fast float %differeturn, %[[newpowdret]]
; CHECK-NEXT:   %[[isxzero:.+]] = fcmp fast oeq float %x, 0.000000e+00
; CHECK-NEXT:   %[[origpow:.+]] = call fast float @air.pow.f32(float %x, float %y)
; CHECK-NEXT:   %[[logy:.+]] = call fast float @llvm.log.f32(float %x)
; CHECK-NEXT:   %[[origpowdret:.+]] = fmul fast float %[[origpow]], %[[logy]]
; CHECK-NEXT:   %[[guardeddy:.+]] = select fast i1 %[[isxzero]], float 0.000000e+00, float %[[origpowdret]]
; CHECK-NEXT:   %[[dy:.+]] = fmul fast float %differeturn, %[[guardeddy]]
; CHECK-NEXT:   %[[interres:.+]] = insertvalue { float, float } undef, float %[[dx:.+]], 0
; CHECK-NEXT:   %[[finalres:.+]] = insertvalue { float, float } %[[interres]], float %[[dy:.+]], 1
; CHECK-NEXT:   ret { float, float } %[[finalres]]
; CHECK-NEXT: }
