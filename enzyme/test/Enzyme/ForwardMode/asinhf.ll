; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define float @tester(float %x) {
entry:
  %0 = tail call fast float @asinhf(float %x)
  ret float %0
}

define float @test_derivative(float %x) {
entry:
  %0 = tail call float (float (float)*, ...) @__enzyme_fwddiff(float (float)* nonnull @tester, float %x, float 1.0)
  ret float %0
}

; Function Attrs: nounwind readnone speculatable
declare float @asinhf(float)

; Function Attrs: nounwind
declare float @__enzyme_fwddiff(float (float)*, ...)

; CHECK: define internal float @fwddiffetester(float %x, float %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = fmul fast float %x, %x
; CHECK-NEXT:   %1 = fadd fast float %0, 1.000000e+00
; CHECK-NEXT:   %2 = call fast float @llvm.sqrt.f32(float %1)
; CHECK-NEXT:   %3 = fdiv fast float %"x'", %2
; CHECK-NEXT:   ret float %3
; CHECK-NEXT: }

