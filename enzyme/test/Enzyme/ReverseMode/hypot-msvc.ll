; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -simplifycfg -instsimplify -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,%simplifycfg,instsimplify,adce)" -S | FileCheck %s

; The Windows CRT names hypotf _hypotf

declare float @_hypotf(float, float)

define float @tester(float %x, float %y) {
entry:
  %call = call float @_hypotf(float %x, float %y)
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
; CHECK-NEXT:   %[[V0:.+]] = fmul fast float %differeturn, %x
; CHECK-NEXT:   %[[V1:.+]] = call fast float @_hypotf(float %x, float %y)
; CHECK-NEXT:   %[[V2:.+]] = fdiv fast float %[[V0]], %[[V1]]
; CHECK-NEXT:   %[[V3:.+]] = fmul fast float %differeturn, %y
; CHECK-NEXT:   %[[V4:.+]] = call fast float @_hypotf(float %x, float %y)
; CHECK-NEXT:   %[[V5:.+]] = fdiv fast float %[[V3]], %[[V4]]
; CHECK-NEXT:   %[[V6:.+]] = insertvalue { float, float } undef, float %[[V2]], 0
; CHECK-NEXT:   %[[V7:.+]] = insertvalue { float, float } %[[V6]], float %[[V5]], 1
; CHECK-NEXT:   ret { float, float } %[[V7]]
; CHECK-NEXT: }
