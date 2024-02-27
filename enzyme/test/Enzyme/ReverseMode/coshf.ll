; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -simplifycfg -instsimplify -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,%simplifycfg,instsimplify)" -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define float @tester(float %x) {
entry:
  %0 = tail call fast float @coshf(float %x)
  ret float %0
}

define float @test_derivative(float %x) {
entry:
  %0 = tail call float (float (float)*, ...) @__enzyme_autodiff(float (float)* nonnull @tester, float %x)
  ret float %0
}

; Function Attrs: nounwind readnone speculatable
declare float @coshf(float)

; Function Attrs: nounwind
declare float @__enzyme_autodiff(float (float)*, ...)

; CHECK: define internal { float } @diffetester(float %x, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call fast float @sinhf(float %x)
; CHECK-NEXT:   %1 = fmul fast float %differeturn, %0
; CHECK-NEXT:   %2 = insertvalue { float } undef, float %1, 0
; CHECK-NEXT:   ret { float } %2
; CHECK-NEXT: }
