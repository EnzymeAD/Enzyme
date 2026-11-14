; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="preserve-nvvm,enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s; fi

declare float @air.sinpi.f32(float)
declare float @air.cospi.f32(float)
declare float @air.asinh.f32(float)
declare float @air.acosh.f32(float)
declare float @air.atanh.f32(float)
declare float @air.fast_atanh.f32(float)
declare float @air.rsqrt.f32(float)

declare float @__enzyme_fwddiff(...)

define float @f_sinpi(float %x) {
entry:
  %r = call float @air.sinpi.f32(float %x)
  ret float %r
}

define float @df_sinpi(float %x, float %dx) {
entry:
  %r = call float (...) @__enzyme_fwddiff(ptr @f_sinpi, float %x, float %dx)
  ret float %r
}

define float @f_cospi(float %x) {
entry:
  %r = call float @air.cospi.f32(float %x)
  ret float %r
}

define float @df_cospi(float %x, float %dx) {
entry:
  %r = call float (...) @__enzyme_fwddiff(ptr @f_cospi, float %x, float %dx)
  ret float %r
}

define float @f_asinh(float %x) {
entry:
  %r = call float @air.asinh.f32(float %x)
  ret float %r
}

define float @df_asinh(float %x, float %dx) {
entry:
  %r = call float (...) @__enzyme_fwddiff(ptr @f_asinh, float %x, float %dx)
  ret float %r
}

define float @f_acosh(float %x) {
entry:
  %r = call float @air.acosh.f32(float %x)
  ret float %r
}

define float @df_acosh(float %x, float %dx) {
entry:
  %r = call float (...) @__enzyme_fwddiff(ptr @f_acosh, float %x, float %dx)
  ret float %r
}

define float @f_atanh(float %x) {
entry:
  %r = call float @air.atanh.f32(float %x)
  ret float %r
}

define float @df_atanh(float %x, float %dx) {
entry:
  %r = call float (...) @__enzyme_fwddiff(ptr @f_atanh, float %x, float %dx)
  ret float %r
}

define float @f_fast_atanh(float %x) {
entry:
  %r = call float @air.fast_atanh.f32(float %x)
  ret float %r
}

define float @df_fast_atanh(float %x, float %dx) {
entry:
  %r = call float (...) @__enzyme_fwddiff(ptr @f_fast_atanh, float %x, float %dx)
  ret float %r
}

define float @f_rsqrt(float %x) {
entry:
  %r = call float @air.rsqrt.f32(float %x)
  ret float %r
}

define float @df_rsqrt(float %x, float %dx) {
entry:
  %r = call float (...) @__enzyme_fwddiff(ptr @f_rsqrt, float %x, float %dx)
  ret float %r
}

; CHECK: define internal float @fwddiffef_sinpi(float %x, float %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[s0:.+]] = fadd fast float %x, 5.000000e-01
; CHECK-NEXT:   %[[s1:.+]] = call fast float @air.sinpi.f32(float %[[s0]])
; CHECK-NEXT:   %[[s2:.+]] = fmul fast float %[[s1]], %"x'"
; CHECK-NEXT:   %[[s3:.+]] = fmul fast float 0x400921FB60000000, %[[s2]]
; CHECK-NEXT:   ret float %[[s3]]

; CHECK: define internal float @fwddiffef_cospi(float %x, float %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[c0:.+]] = fadd fast float %x, 5.000000e-01
; CHECK-NEXT:   %[[c1:.+]] = call fast float @air.cospi.f32(float %[[c0]])
; CHECK-NEXT:   %[[c2:.+]] = fmul fast float %[[c1]], %"x'"
; CHECK-NEXT:   %[[c3:.+]] = fmul fast float 0x400921FB60000000, %[[c2]]
; CHECK-NEXT:   ret float %[[c3]]

; CHECK: define internal float @fwddiffef_asinh(float %x, float %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[as0:.+]] = fmul fast float %x, %x
; CHECK-NEXT:   %[[as1:.+]] = fadd fast float %[[as0]], 1.000000e+00
; CHECK-NEXT:   %[[as2:.+]] = call fast float @llvm.sqrt.f32(float %[[as1]])
; CHECK-NEXT:   %[[as3:.+]] = fdiv fast float %"x'", %[[as2]]
; CHECK-NEXT:   ret float %[[as3]]

; CHECK: define internal float @fwddiffef_acosh(float %x, float %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[ac0:.+]] = fmul fast float %x, %x
; CHECK-NEXT:   %[[ac1:.+]] = fsub fast float %[[ac0]], 1.000000e+00
; CHECK-NEXT:   %[[ac2:.+]] = call fast float @llvm.sqrt.f32(float %[[ac1]])
; CHECK-NEXT:   %[[ac3:.+]] = fdiv fast float %"x'", %[[ac2]]
; CHECK-NEXT:   ret float %[[ac3]]

; CHECK: define internal float @fwddiffef_atanh(float %x, float %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[at0:.+]] = fmul fast float %x, %x
; CHECK-NEXT:   %[[at1:.+]] = fsub fast float 1.000000e+00, %[[at0]]
; CHECK-NEXT:   %[[at2:.+]] = fdiv fast float %"x'", %[[at1]]
; CHECK-NEXT:   ret float %[[at2]]

; CHECK: define internal float @fwddiffef_fast_atanh(float %x, float %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[fat0:.+]] = fmul fast float %x, %x
; CHECK-NEXT:   %[[fat1:.+]] = fsub fast float 1.000000e+00, %[[fat0]]
; CHECK-NEXT:   %[[fat2:.+]] = fdiv fast float %"x'", %[[fat1]]
; CHECK-NEXT:   ret float %[[fat2]]

; CHECK: define internal float @fwddiffef_rsqrt(float %x, float %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[r0:.+]] = call fast float @air.rsqrt.f32(float %x)
; CHECK-DAG:    %[[r1:.+]] = fmul fast float %"x'", %[[r0]]
; CHECK-DAG:    %[[r2:.+]] = fmul fast float 2.000000e+00, %x
; CHECK-NEXT:   %[[r3:.+]] = fdiv fast float %[[r1]], %[[r2]]
; CHECK-NEXT:   %[[r4:.+]] = fneg fast float %[[r3]]
; CHECK-NEXT:   ret float %[[r4]]
