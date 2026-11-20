; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -preserve-nvvm -enzyme -mem2reg -simplifycfg -instsimplify -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="preserve-nvvm,enzyme,function(mem2reg,%simplifycfg,instsimplify,adce)" -S | FileCheck %s

target triple = "nvptx64-nvidia-cuda"

declare float @_ZL4fminff(float, float)
declare float @_ZL4fmaxff(float, float)
declare float @_ZL8copysignff(float, float)
declare float @_ZL3fmafff(float, float, float)

define float @tester(float %x, float %y) {
  %a = call float @_ZL4fminff(float %x, float %y)
  %b = call float @_ZL4fmaxff(float %a, float %y)
  %c = call float @_ZL8copysignff(float %b, float -1.000000e+00)
  %d = call float @_ZL3fmafff(float %c, float 2.000000e+00, float %x)
  ret float %d
}

declare { float, float } @__enzyme_autodiff(float (float, float)*, ...)

define { float, float } @d_tester(float %x, float %y) {
  %r = call { float, float } (float (float, float)*, ...) @__enzyme_autodiff(float (float, float)* nonnull @tester, float %x, float %y)
  ret { float, float } %r
}

; CHECK-NOT: @_ZL4fminff
; CHECK-NOT: @_ZL4fmaxff
; CHECK-NOT: @_ZL8copysignff
; CHECK-NOT: @_ZL3fmafff

; CHECK: define float @tester(float %x, float %y)
; CHECK-NEXT:   %a = call float @llvm.minnum.f32(float %x, float %y)
; CHECK-NEXT:   %b = call float @llvm.maxnum.f32(float %a, float %y)
; CHECK-NEXT:   %c = call float @llvm.copysign.f32(float %b, float -1.000000e+00)
; CHECK-NEXT:   %d = call float @llvm.fma.f32(float %c, float 2.000000e+00, float %x)
; CHECK-NEXT:   ret float %d

; CHECK: define internal { float, float } @diffetester(float %x, float %y, float %differeturn)
; CHECK:   ret { float, float } %{{.+}}
