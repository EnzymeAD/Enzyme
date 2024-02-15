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
  %0 = tail call float (float (float)*, ...) @__enzyme_autodiff(float (float)* nonnull @tester, float %x)
  ret float %0
}

; Function Attrs: nounwind readnone speculatable
declare float @asinhf(float)

; Function Attrs: nounwind
declare float @__enzyme_autodiff(float (float)*, ...)

; CHECK: define internal { float } @diffetester(float %x, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"'de" = alloca float, align 4
; CHECK-NEXT:   store float 0.000000e+00, float* %"'de", align 4
; CHECK-NEXT:   %"x'de" = alloca float, align 4
; CHECK-NEXT:   store float 0.000000e+00, float* %"x'de", align 4
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:
; CHECK-NEXT:   store float %differeturn, float* %"'de", align 4
; CHECK-NEXT:   %0 = load float, float* %"'de", align 4
; CHECK-NEXT:   store float 0.000000e+00, float* %"'de", align 4
; CHECK-NEXT:   %1 = fmul fast float %x, %x
; CHECK-NEXT:   %2 = fadd fast float %1, 1.000000e+00
; CHECK-NEXT:   %3 = call fast float @llvm.sqrt.f32(float %2)
; CHECK-NEXT:   %4 = fdiv fast float %0, %3
; CHECK-NEXT:   %5 = load float, float* %"x'de", align 4
; CHECK-NEXT:   %6 = fadd fast float %5, %4
; CHECK-NEXT:   store float %6, float* %"x'de", align 4
; CHECK-NEXT:   %7 = load float, float* %"x'de", align 4
; CHECK-NEXT:   %8 = insertvalue { float } undef, float %7, 0
; CHECK-NEXT:   ret { float } %8
; CHECK-NEXT: }

