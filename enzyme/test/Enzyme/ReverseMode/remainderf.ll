; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -simplifycfg -instsimplify -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,%simplifycfg,instsimplify,adce)" -S | FileCheck %s

; remainderf uses the same rule as remainder

declare float @remainderf(float, float)

define float @tester(float %x, float %y) {
entry:
  %call = call float @remainderf(float %x, float %y)
  ret float %call
}

define float @test_derivative(float %x, float %y) {
entry:
  %0 = tail call float (float (float, float)*, ...) @__enzyme_autodiff(float (float, float)* nonnull @tester, float %x, float %y)
  ret float %0
}

declare float @__enzyme_autodiff(float (float, float)*, ...)

; CHECK: define internal { float, float } @diffetester(float %x, float %y, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[V0:.+]] = {{(fneg fast float|fsub fast float \-0\.000000e\+00,)}} %differeturn
; CHECK-NEXT:   %[[V1:.+]] = fdiv fast float %x, %y
; CHECK-NEXT:   %[[V2:.+]] = call fast float @llvm.round.f32(float %[[V1]])
; CHECK-NEXT:   %[[V3:.+]] = fmul fast float %[[V0]], %[[V2]]
; CHECK-NEXT:   %[[V4:.+]] = insertvalue { float, float } undef, float %differeturn, 0
; CHECK-NEXT:   %[[V5:.+]] = insertvalue { float, float } %[[V4]], float %[[V3]], 1
; CHECK-NEXT:   ret { float, float } %[[V5]]
; CHECK-NEXT: }
