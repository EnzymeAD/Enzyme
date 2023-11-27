; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -simplifycfg -instsimplify -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,%simplifycfg,instsimplify,adce)" -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define float @tester(float %x) {
entry:
  %0 = tail call fast float @expm1f(float %x)
  ret float %0
}

define float @test_derivative(float %x) {
entry:
  %0 = tail call float (float (float)*, ...) @__enzyme_autodiff(float (float)* nonnull @tester, float %x)
  ret float %0
}

; Function Attrs: nounwind readnone speculatable
declare float @expm1f(float)

; Function Attrs: nounwind
declare float @__enzyme_autodiff(float (float)*, ...)

; CHECK: define internal { float } @diffetester(float %x, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[i0:.+]] = call fast float @llvm.exp.f32(float %x)
; CHECK-NEXT:   %[[i1:.+]] = fmul fast float %[[i0]], %differeturn
; CHECK-NEXT:   %[[i2:.+]] = insertvalue { float } {{(undef|poison)}}, float %[[i1]], 0
; CHECK-NEXT:   ret { float } %[[i2]]
; CHECK-NEXT: }