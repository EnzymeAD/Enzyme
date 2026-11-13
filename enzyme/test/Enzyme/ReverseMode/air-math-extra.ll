; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="preserve-nvvm,enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s; fi

declare float @air.sinpi.f32(float)
declare float @air.cospi.f32(float)
declare float @air.asinh.f32(float)
declare float @air.acosh.f32(float)
declare float @air.atanh.f32(float)
declare float @air.fast_atanh.f32(float)
declare float @air.fast_sinpi.f32(float)
declare float @air.fast_cospi.f32(float)
declare float @air.rsqrt.f32(float)
declare float @air.fast_rsqrt.f32(float)

declare double @air.sinpi.f64(double)
declare double @air.cospi.f64(double)
declare double @air.asinh.f64(double)
declare double @air.acosh.f64(double)
declare double @air.atanh.f64(double)

declare half @air.sinpi.f16(half)
declare half @air.asinh.f16(half)
declare half @air.atanh.f16(half)

declare float @__enzyme_autodiff(...)
declare double @__enzyme_autodiff.f64(...)
declare half @__enzyme_autodiff.f16(...)

define float @f_sinpi(float %x) {
entry:
  %r = call float @air.sinpi.f32(float %x)
  ret float %r
}

define float @df_sinpi(float %x) {
entry:
  %r = call float (...) @__enzyme_autodiff(ptr @f_sinpi, float %x)
  ret float %r
}

define float @f_cospi(float %x) {
entry:
  %r = call float @air.cospi.f32(float %x)
  ret float %r
}

define float @df_cospi(float %x) {
entry:
  %r = call float (...) @__enzyme_autodiff(ptr @f_cospi, float %x)
  ret float %r
}

define float @f_asinh(float %x) {
entry:
  %r = call float @air.asinh.f32(float %x)
  ret float %r
}

define float @df_asinh(float %x) {
entry:
  %r = call float (...) @__enzyme_autodiff(ptr @f_asinh, float %x)
  ret float %r
}

define float @f_acosh(float %x) {
entry:
  %r = call float @air.acosh.f32(float %x)
  ret float %r
}

define float @df_acosh(float %x) {
entry:
  %r = call float (...) @__enzyme_autodiff(ptr @f_acosh, float %x)
  ret float %r
}

define float @f_atanh(float %x) {
entry:
  %r = call float @air.atanh.f32(float %x)
  ret float %r
}

define float @df_atanh(float %x) {
entry:
  %r = call float (...) @__enzyme_autodiff(ptr @f_atanh, float %x)
  ret float %r
}

define float @f_fast_atanh(float %x) {
entry:
  %r = call float @air.fast_atanh.f32(float %x)
  ret float %r
}

define float @df_fast_atanh(float %x) {
entry:
  %r = call float (...) @__enzyme_autodiff(ptr @f_fast_atanh, float %x)
  ret float %r
}

define float @f_fast_sinpi(float %x) {
entry:
  %r = call float @air.fast_sinpi.f32(float %x)
  ret float %r
}

define float @df_fast_sinpi(float %x) {
entry:
  %r = call float (...) @__enzyme_autodiff(ptr @f_fast_sinpi, float %x)
  ret float %r
}

define float @f_fast_cospi(float %x) {
entry:
  %r = call float @air.fast_cospi.f32(float %x)
  ret float %r
}

define float @df_fast_cospi(float %x) {
entry:
  %r = call float (...) @__enzyme_autodiff(ptr @f_fast_cospi, float %x)
  ret float %r
}

define float @f_rsqrt(float %x) {
entry:
  %r = call float @air.rsqrt.f32(float %x)
  ret float %r
}

define float @df_rsqrt(float %x) {
entry:
  %r = call float (...) @__enzyme_autodiff(ptr @f_rsqrt, float %x)
  ret float %r
}

define float @f_fast_rsqrt(float %x) {
entry:
  %r = call float @air.fast_rsqrt.f32(float %x)
  ret float %r
}

define float @df_fast_rsqrt(float %x) {
entry:
  %r = call float (...) @__enzyme_autodiff(ptr @f_fast_rsqrt, float %x)
  ret float %r
}

define double @f_sinpi_f64(double %x) {
entry:
  %r = call double @air.sinpi.f64(double %x)
  ret double %r
}

define double @df_sinpi_f64(double %x) {
entry:
  %r = call double (...) @__enzyme_autodiff.f64(ptr @f_sinpi_f64, double %x)
  ret double %r
}

define double @f_cospi_f64(double %x) {
entry:
  %r = call double @air.cospi.f64(double %x)
  ret double %r
}

define double @df_cospi_f64(double %x) {
entry:
  %r = call double (...) @__enzyme_autodiff.f64(ptr @f_cospi_f64, double %x)
  ret double %r
}

define double @f_asinh_f64(double %x) {
entry:
  %r = call double @air.asinh.f64(double %x)
  ret double %r
}

define double @df_asinh_f64(double %x) {
entry:
  %r = call double (...) @__enzyme_autodiff.f64(ptr @f_asinh_f64, double %x)
  ret double %r
}

define double @f_acosh_f64(double %x) {
entry:
  %r = call double @air.acosh.f64(double %x)
  ret double %r
}

define double @df_acosh_f64(double %x) {
entry:
  %r = call double (...) @__enzyme_autodiff.f64(ptr @f_acosh_f64, double %x)
  ret double %r
}

define double @f_atanh_f64(double %x) {
entry:
  %r = call double @air.atanh.f64(double %x)
  ret double %r
}

define double @df_atanh_f64(double %x) {
entry:
  %r = call double (...) @__enzyme_autodiff.f64(ptr @f_atanh_f64, double %x)
  ret double %r
}

define half @f_sinpi_f16(half %x) {
entry:
  %r = call half @air.sinpi.f16(half %x)
  ret half %r
}

define half @df_sinpi_f16(half %x) {
entry:
  %r = call half (...) @__enzyme_autodiff.f16(ptr @f_sinpi_f16, half %x)
  ret half %r
}

define half @f_asinh_f16(half %x) {
entry:
  %r = call half @air.asinh.f16(half %x)
  ret half %r
}

define half @df_asinh_f16(half %x) {
entry:
  %r = call half (...) @__enzyme_autodiff.f16(ptr @f_asinh_f16, half %x)
  ret half %r
}

define half @f_atanh_f16(half %x) {
entry:
  %r = call half @air.atanh.f16(half %x)
  ret half %r
}

define half @df_atanh_f16(half %x) {
entry:
  %r = call half (...) @__enzyme_autodiff.f16(ptr @f_atanh_f16, half %x)
  ret half %r
}

; CHECK: define internal { float } @diffef_sinpi(float %x, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[s0:.+]] = fadd fast float %x, 5.000000e-01
; CHECK-NEXT:   %[[s1:.+]] = call fast float @air.sinpi.f32(float %[[s0]])
; CHECK-NEXT:   %[[s2:.+]] = fmul fast float %[[s1]], %differeturn
; CHECK-NEXT:   %[[s3:.+]] = fmul fast float 0x400921FB60000000, %[[s2]]

; CHECK: define internal { float } @diffef_cospi(float %x, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[c0:.+]] = fadd fast float %x, 5.000000e-01
; CHECK-NEXT:   %[[c1:.+]] = call fast float @air.cospi.f32(float %[[c0]])
; CHECK-NEXT:   %[[c2:.+]] = fmul fast float %[[c1]], %differeturn
; CHECK-NEXT:   %[[c3:.+]] = fmul fast float 0x400921FB60000000, %[[c2]]

; CHECK: define internal { float } @diffef_asinh(float %x, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[as0:.+]] = fmul fast float %x, %x
; CHECK-NEXT:   %[[as1:.+]] = fadd fast float %[[as0]], 1.000000e+00
; CHECK-NEXT:   %[[as2:.+]] = call fast float @llvm.sqrt.f32(float %[[as1]])
; CHECK-NEXT:   %[[as3:.+]] = fdiv fast float %differeturn, %[[as2]]

; CHECK: define internal { float } @diffef_acosh(float %x, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[ac0:.+]] = fmul fast float %x, %x
; CHECK-NEXT:   %[[ac1:.+]] = fsub fast float %[[ac0]], 1.000000e+00
; CHECK-NEXT:   %[[ac2:.+]] = call fast float @llvm.sqrt.f32(float %[[ac1]])
; CHECK-NEXT:   %[[ac3:.+]] = fdiv fast float %differeturn, %[[ac2]]

; CHECK: define internal { float } @diffef_atanh(float %x, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[at0:.+]] = fmul fast float %x, %x
; CHECK-NEXT:   %[[at1:.+]] = fsub fast float 1.000000e+00, %[[at0]]
; CHECK-NEXT:   %[[at2:.+]] = fdiv fast float %differeturn, %[[at1]]

; CHECK: define internal { float } @diffef_fast_atanh(float %x, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[fat0:.+]] = fmul fast float %x, %x
; CHECK-NEXT:   %[[fat1:.+]] = fsub fast float 1.000000e+00, %[[fat0]]
; CHECK-NEXT:   %[[fat2:.+]] = fdiv fast float %differeturn, %[[fat1]]

; CHECK: define internal { float } @diffef_fast_sinpi(float %x, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[fs0:.+]] = fadd fast float %x, 5.000000e-01
; CHECK-NEXT:   %[[fs1:.+]] = call fast float @air.fast_sinpi.f32(float %[[fs0]])

; CHECK: define internal { float } @diffef_fast_cospi(float %x, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[fc0:.+]] = fadd fast float %x, 5.000000e-01
; CHECK-NEXT:   %[[fc1:.+]] = call fast float @air.fast_cospi.f32(float %[[fc0]])

; CHECK: define internal { float } @diffef_rsqrt(float %x, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[r0:.+]] = call fast float @air.rsqrt.f32(float %x)
; CHECK-DAG:    %[[r1:.+]] = fmul fast float %differeturn, %[[r0]]
; CHECK-DAG:    %[[r2:.+]] = fmul fast float 2.000000e+00, %x
; CHECK-NEXT:   %[[r3:.+]] = fdiv fast float %[[r1]], %[[r2]]
; CHECK-NEXT:   %[[r4:.+]] = fneg fast float %[[r3]]

; CHECK: define internal { float } @diffef_fast_rsqrt(float %x, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[fr0:.+]] = call fast float @air.fast_rsqrt.f32(float %x)

; CHECK: define internal { double } @diffef_sinpi_f64(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[ds0:.+]] = fadd fast double %x, 5.000000e-01
; CHECK-NEXT:   %[[ds1:.+]] = call fast double @air.sinpi.f64(double %[[ds0]])

; CHECK: define internal { double } @diffef_cospi_f64(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[dc0:.+]] = fadd fast double %x, 5.000000e-01
; CHECK-NEXT:   %[[dc1:.+]] = call fast double @air.cospi.f64(double %[[dc0]])

; CHECK: define internal { double } @diffef_asinh_f64(double %x, double %differeturn)
; CHECK:        call fast double @llvm.sqrt.f64(

; CHECK: define internal { double } @diffef_acosh_f64(double %x, double %differeturn)
; CHECK:        call fast double @llvm.sqrt.f64(

; CHECK: define internal { double } @diffef_atanh_f64(double %x, double %differeturn)
; CHECK:        fdiv fast double %differeturn,

; CHECK: define internal { half } @diffef_sinpi_f16(half %x, half %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[hs0:.+]] = fadd fast half %x, 0xH3800
; CHECK-NEXT:   %[[hs1:.+]] = call fast half @air.sinpi.f16(half %[[hs0]])

; CHECK: define internal { half } @diffef_asinh_f16(half %x, half %differeturn)
; CHECK:        call fast half @llvm.sqrt.f16(

; CHECK: define internal { half } @diffef_atanh_f16(half %x, half %differeturn)
; CHECK:        fdiv fast half %differeturn,
