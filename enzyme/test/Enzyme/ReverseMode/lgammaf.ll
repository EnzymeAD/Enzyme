; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -simplifycfg -instsimplify -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,%simplifycfg,instsimplify,adce)" -S | FileCheck %s

; C lgammaf returns a scalar: d/dx lgammaf(x) = digammaf(x)

declare float @lgammaf(float)

define float @tester(float %x) {
entry:
  %call = call float @lgammaf(float %x)
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
; CHECK-NEXT:   %0 = call fast float @digammaf(float %x)
; CHECK-NEXT:   %1 = fmul fast float %0, %differeturn
; CHECK-NEXT:   %2 = insertvalue { float } undef, float %1, 0
; CHECK-NEXT:   ret { float } %2
; CHECK-NEXT: }
