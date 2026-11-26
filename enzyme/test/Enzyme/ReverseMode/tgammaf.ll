; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -simplifycfg -instsimplify -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,%simplifycfg,instsimplify,adce)" -S | FileCheck %s

; d/dx tgammaf(x) = tgammaf(x) * digammaf(x)

declare float @tgammaf(float)

define float @tester(float %x) {
entry:
  %call = call float @tgammaf(float %x)
  ret float %call
}

define float @test_derivative(float %x) {
entry:
  %0 = tail call float (float (float)*, ...) @__enzyme_autodiff(float (float)* nonnull @tester, float %x)
  ret float %0
}

declare float @__enzyme_autodiff(float (float)*, ...)

; CHECK: define internal { float } @diffetester(float %x, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[V0:.+]] = call fast float @tgammaf(float %x)
; CHECK-NEXT:   %[[V1:.+]] = call fast float @digammaf(float %x)
; CHECK-NEXT:   %[[V2:.+]] = fmul fast float %[[V0]], %[[V1]]
; CHECK-NEXT:   %[[V3:.+]] = fmul fast float %[[V2]], %differeturn
; CHECK-NEXT:   %[[V4:.+]] = insertvalue { float } undef, float %[[V3]], 0
; CHECK-NEXT:   ret { float } %[[V4]]
; CHECK-NEXT: }
