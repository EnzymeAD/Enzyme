; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | %lli - | FileCheck %s --check-prefix=EVAL
; CHECK-NOT: @diffe__mulsc3(
; CHECK-NOT: @fwddiffe__mulsc3(
; CHECK-NOT: @diffe2__mulsc3(
; CHECK-NOT: @fwddiffe2__mulsc3(
; CHECK-NOT: @diffe__divsc3(
; CHECK-NOT: @fwddiffe__divsc3(
; CHECK-NOT: @diffe2__divsc3(
; CHECK-NOT: @fwddiffe2__divsc3(
; CHECK-NOT: @diffecabsf(
; CHECK-NOT: @fwddiffecabsf(
; CHECK-NOT: @diffe2cabsf(
; CHECK-NOT: @fwddiffe2cabsf(
; CHECK-NOT: @diffecabs(
; CHECK-NOT: @fwddiffecabs(
; CHECK-NOT: @diffe2cabs(
; CHECK-NOT: @fwddiffe2cabs(
; EVAL: mul: -6.00 -17.00 -20.00 -35.00 -65.00 -65.00 -84.00
; EVAL-NEXT: mul_vec_rev: 7.00 -1.00 15.00 -5.00
; EVAL-NEXT: div: 0.72 -0.04 -7.84 1.88 -0.08 -0.08 9.42
; EVAL-NEXT: cabsf_arg: 0.60 0.80 2.20 2.20 -0.40 0.60 0.80
; EVAL-NEXT: cabsf_ins: 0.60 0.80 2.20
; EVAL-NEXT: cabs_arg: 0.60 0.80 2.20 2.20 -0.40

; Verify forward/reverse (scalar and vector) derivatives of <2 x float> complex
; libm builtins (__mulsc3, __divsc3, cabsf) and direct-argument [2 x double] cabs
; numerically against exact analytic values.

@.fmt7 = private unnamed_addr constant [40 x i8] c"%s: %.2f %.2f %.2f %.2f %.2f %.2f %.2f\0A\00", align 1
@.fmt4 = private unnamed_addr constant [25 x i8] c"%s: %.2f %.2f %.2f %.2f\0A\00", align 1
@.fmt5 = private unnamed_addr constant [30 x i8] c"%s: %.2f %.2f %.2f %.2f %.2f\0A\00", align 1
@.fmt3 = private unnamed_addr constant [20 x i8] c"%s: %.2f %.2f %.2f\0A\00", align 1

@.s_mul = private unnamed_addr constant [4 x i8] c"mul\00", align 1
@.s_mul_vr = private unnamed_addr constant [12 x i8] c"mul_vec_rev\00", align 1
@.s_div = private unnamed_addr constant [4 x i8] c"div\00", align 1
@.s_cabsf_arg = private unnamed_addr constant [10 x i8] c"cabsf_arg\00", align 1
@.s_cabsf_ins = private unnamed_addr constant [10 x i8] c"cabsf_ins\00", align 1
@.s_cabs_arg = private unnamed_addr constant [9 x i8] c"cabs_arg\00", align 1

declare float @llvm.sqrt.f32(float)
declare double @llvm.sqrt.f64(double)

; Provide self-contained definitions so lli does not depend on compiler-rt/libm.
define <2 x float> @__mulsc3(float %a, float %b, float %c, float %d) noinline {
entry:
  %ac = fmul float %a, %c
  %bd = fmul float %b, %d
  %ad = fmul float %a, %d
  %bc = fmul float %b, %c
  %re = fsub float %ac, %bd
  %im = fadd float %ad, %bc
  %v0 = insertelement <2 x float> undef, float %re, i64 0
  %v1 = insertelement <2 x float> %v0, float %im, i64 1
  ret <2 x float> %v1
}

define <2 x float> @__divsc3(float %a, float %b, float %c, float %d) noinline {
entry:
  %cc = fmul float %c, %c
  %dd = fmul float %d, %d
  %denom = fadd float %cc, %dd
  %ac = fmul float %a, %c
  %bd = fmul float %b, %d
  %bc = fmul float %b, %c
  %ad = fmul float %a, %d
  %num_re = fadd float %ac, %bd
  %num_im = fsub float %bc, %ad
  %re = fdiv float %num_re, %denom
  %im = fdiv float %num_im, %denom
  %v0 = insertelement <2 x float> undef, float %re, i64 0
  %v1 = insertelement <2 x float> %v0, float %im, i64 1
  ret <2 x float> %v1
}

define float @cabsf(<2 x float> %z) noinline {
entry:
  %re = extractelement <2 x float> %z, i64 0
  %im = extractelement <2 x float> %z, i64 1
  %r2 = fmul float %re, %re
  %i2 = fmul float %im, %im
  %sum = fadd float %r2, %i2
  %res = call float @llvm.sqrt.f32(float %sum)
  ret float %res
}

define double @cabs([2 x double] %z) noinline {
entry:
  %re = extractvalue [2 x double] %z, 0
  %im = extractvalue [2 x double] %z, 1
  %r2 = fmul double %re, %re
  %i2 = fmul double %im, %im
  %sum = fadd double %r2, %i2
  %res = call double @llvm.sqrt.f64(double %sum)
  ret double %res
}

declare { float, float, float, float } @__enzyme_autodiff_f4(...)
declare { [2 x float], [2 x float], [2 x float], [2 x float] } @__enzyme_autodiff_f4_v2(...)
declare float @__enzyme_fwddiff_f(...)
declare [2 x float] @__enzyme_fwddiff_f_v2(...)
declare { <2 x float> } @__enzyme_autodiff_v2f(...)
declare { [2 x <2 x float>] } @__enzyme_autodiff_v2f_v2(...)
declare { float, float } @__enzyme_autodiff_f2(...)
declare { [2 x double] } @__enzyme_autodiff_a2d(...)
declare double @__enzyme_fwddiff_d(...)
declare [2 x double] @__enzyme_fwddiff_d_v2(...)
declare i32 @printf(ptr, ...)

define <2 x float> @mul_raw(float %a, float %b, float %c, float %d) {
entry:
  %prod = call <2 x float> @__mulsc3(float %a, float %b, float %c, float %d)
  ret <2 x float> %prod
}

define float @mul_loss(float %xre, float %xim, float %yre, float %yim) {
entry:
  %prod = call <2 x float> @__mulsc3(float %xre, float %xim, float %yre, float %yim)
  %re = extractelement <2 x float> %prod, i64 0
  %im = extractelement <2 x float> %prod, i64 1
  %w_re = fmul float 2.000000e+00, %re
  %w_im = fmul float 3.000000e+00, %im
  %loss = fsub float %w_re, %w_im
  ret float %loss
}

define float @div_loss(float %xre, float %xim, float %yre, float %yim) {
entry:
  %quot = call <2 x float> @__divsc3(float %xre, float %xim, float %yre, float %yim)
  %re = extractelement <2 x float> %quot, i64 0
  %im = extractelement <2 x float> %quot, i64 1
  %w_re = fmul float 2.000000e+00, %re
  %w_im = fmul float 3.000000e+00, %im
  %loss = fsub float %w_re, %w_im
  ret float %loss
}

define float @cabsf_arg(<2 x float> %z) {
entry:
  %res = call float @cabsf(<2 x float> %z)
  ret float %res
}

define float @cabsf_ins(float %a, float %b) {
entry:
  %v0 = insertelement <2 x float> poison, float %a, i64 0
  %v1 = insertelement <2 x float> %v0, float %b, i64 1
  %res = call float @cabsf(<2 x float> %v1)
  ret float %res
}

define double @cabs_arg([2 x double] %z) {
entry:
  %res = call double @cabs([2 x double] %z)
  ret double %res
}

; CHECK-LABEL: define i32 @main(
define i32 @main() {
entry:
  ; __mulsc3 reverse, forward, forward-vector
  ; CHECK: call { float, float, float, float } @diffemul_loss(
  %g_mul = call { float, float, float, float } (...) @__enzyme_autodiff_f4(ptr @mul_loss, float 5.000000e+00, float 1.000000e+01, float 3.000000e+00, float 4.000000e+00)
  %gm0_f = extractvalue { float, float, float, float } %g_mul, 0
  %gm1_f = extractvalue { float, float, float, float } %g_mul, 1
  %gm2_f = extractvalue { float, float, float, float } %g_mul, 2
  %gm3_f = extractvalue { float, float, float, float } %g_mul, 3
  %gm0 = fpext float %gm0_f to double
  %gm1 = fpext float %gm1_f to double
  %gm2 = fpext float %gm2_f to double
  %gm3 = fpext float %gm3_f to double
  ; CHECK: call fast float @fwddiffemul_loss(
  %fm_f = call float (...) @__enzyme_fwddiff_f(ptr @mul_loss, float 5.000000e+00, float 1.000000e+00, float 1.000000e+01, float 2.000000e+00, float 3.000000e+00, float 3.000000e+00, float 4.000000e+00, float -1.000000e+00)
  %fm = fpext float %fm_f to double
  ; CHECK: call fast [2 x float] @fwddiffe2mul_loss(
  %fmv = call [2 x float] (...) @__enzyme_fwddiff_f_v2(ptr @mul_loss, metadata !"enzyme_width", i64 2, float 5.000000e+00, float 1.000000e+00, float -1.000000e+00, float 1.000000e+01, float 2.000000e+00, float 0.000000e+00, float 3.000000e+00, float 3.000000e+00, float 1.000000e+00, float 4.000000e+00, float -1.000000e+00, float 2.000000e+00)
  %fmv0_f = extractvalue [2 x float] %fmv, 0
  %fmv1_f = extractvalue [2 x float] %fmv, 1
  %fmv0 = fpext float %fmv0_f to double
  %fmv1 = fpext float %fmv1_f to double
  call i32 (ptr, ...) @printf(ptr @.fmt7, ptr @.s_mul, double %gm0, double %gm1, double %gm2, double %gm3, double %fm, double %fmv0, double %fmv1)

  ; __mulsc3 reverse-vector directly returning <2 x float>
  ; CHECK: call { [2 x float], [2 x float], [2 x float], [2 x float] } @diffe2mul_raw(
  %g_mul_v = call { [2 x float], [2 x float], [2 x float], [2 x float] } (...) @__enzyme_autodiff_f4_v2(ptr @mul_raw, metadata !"enzyme_width", i64 2, float 5.000000e+00, float 1.000000e+01, float 3.000000e+00, float 4.000000e+00)
  %gmv0_f = extractvalue { [2 x float], [2 x float], [2 x float], [2 x float] } %g_mul_v, 0, 1
  %gmv1_f = extractvalue { [2 x float], [2 x float], [2 x float], [2 x float] } %g_mul_v, 1, 1
  %gmv2_f = extractvalue { [2 x float], [2 x float], [2 x float], [2 x float] } %g_mul_v, 2, 1
  %gmv3_f = extractvalue { [2 x float], [2 x float], [2 x float], [2 x float] } %g_mul_v, 3, 1
  %gmv0 = fpext float %gmv0_f to double
  %gmv1 = fpext float %gmv1_f to double
  %gmv2 = fpext float %gmv2_f to double
  %gmv3 = fpext float %gmv3_f to double
  call i32 (ptr, ...) @printf(ptr @.fmt4, ptr @.s_mul_vr, double %gmv0, double %gmv1, double %gmv2, double %gmv3)

  ; __divsc3 reverse, forward, forward-vector
  ; CHECK: call { float, float, float, float } @diffediv_loss(
  %g_div = call { float, float, float, float } (...) @__enzyme_autodiff_f4(ptr @div_loss, float 2.500000e+01, float 5.000000e+01, float 3.000000e+00, float 4.000000e+00)
  %gd0_f = extractvalue { float, float, float, float } %g_div, 0
  %gd1_f = extractvalue { float, float, float, float } %g_div, 1
  %gd2_f = extractvalue { float, float, float, float } %g_div, 2
  %gd3_f = extractvalue { float, float, float, float } %g_div, 3
  %gd0 = fpext float %gd0_f to double
  %gd1 = fpext float %gd1_f to double
  %gd2 = fpext float %gd2_f to double
  %gd3 = fpext float %gd3_f to double
  ; CHECK: call fast float @fwddiffediv_loss(
  %fd_f = call float (...) @__enzyme_fwddiff_f(ptr @div_loss, float 2.500000e+01, float 5.000000e+00, float 5.000000e+01, float -1.000000e+01, float 3.000000e+00, float 1.000000e+00, float 4.000000e+00, float 2.000000e+00)
  %fd = fpext float %fd_f to double
  ; CHECK: call fast [2 x float] @fwddiffe2div_loss(
  %fdv = call [2 x float] (...) @__enzyme_fwddiff_f_v2(ptr @div_loss, metadata !"enzyme_width", i64 2, float 2.500000e+01, float 5.000000e+00, float 1.000000e+00, float 5.000000e+01, float -1.000000e+01, float 2.000000e+00, float 3.000000e+00, float 1.000000e+00, float -1.000000e+00, float 4.000000e+00, float 2.000000e+00, float 5.000000e-01)
  %fdv0_f = extractvalue [2 x float] %fdv, 0
  %fdv1_f = extractvalue [2 x float] %fdv, 1
  %fdv0 = fpext float %fdv0_f to double
  %fdv1 = fpext float %fdv1_f to double
  call i32 (ptr, ...) @printf(ptr @.fmt7, ptr @.s_div, double %gd0, double %gd1, double %gd2, double %gd3, double %fd, double %fdv0, double %fdv1)

  ; cabsf with direct <2 x float> parameter (reverse, forward, forward-vector, reverse-vector)
  ; CHECK: call { <2 x float> } @diffecabsf_arg(
  %g_cf = call { <2 x float> } (...) @__enzyme_autodiff_v2f(ptr @cabsf_arg, <2 x float> <float 3.000000e+00, float 4.000000e+00>)
  %gcf_v = extractvalue { <2 x float> } %g_cf, 0
  %gcf0_f = extractelement <2 x float> %gcf_v, i64 0
  %gcf1_f = extractelement <2 x float> %gcf_v, i64 1
  %gcf0 = fpext float %gcf0_f to double
  %gcf1 = fpext float %gcf1_f to double
  ; CHECK: call fast float @fwddiffecabsf_arg(
  %fcf_f = call float (...) @__enzyme_fwddiff_f(ptr @cabsf_arg, <2 x float> <float 3.000000e+00, float 4.000000e+00>, <2 x float> <float 1.000000e+00, float 2.000000e+00>)
  %fcf = fpext float %fcf_f to double
  ; CHECK: call fast [2 x float] @fwddiffe2cabsf_arg(
  %fcfv = call [2 x float] (...) @__enzyme_fwddiff_f_v2(ptr @cabsf_arg, metadata !"enzyme_width", i64 2, <2 x float> <float 3.000000e+00, float 4.000000e+00>, <2 x float> <float 1.000000e+00, float 2.000000e+00>, <2 x float> <float -2.000000e+00, float 1.000000e+00>)
  %fcfv0_f = extractvalue [2 x float] %fcfv, 0
  %fcfv1_f = extractvalue [2 x float] %fcfv, 1
  %fcfv0 = fpext float %fcfv0_f to double
  %fcfv1 = fpext float %fcfv1_f to double
  ; CHECK: call { [2 x <2 x float>] } @diffe2cabsf_arg(
  %g_cfv = call { [2 x <2 x float>] } (...) @__enzyme_autodiff_v2f_v2(ptr @cabsf_arg, metadata !"enzyme_width", i64 2, <2 x float> <float 3.000000e+00, float 4.000000e+00>)
  %gcfv1 = extractvalue { [2 x <2 x float>] } %g_cfv, 0, 1
  %gcfv10_f = extractelement <2 x float> %gcfv1, i64 0
  %gcfv11_f = extractelement <2 x float> %gcfv1, i64 1
  %gcfv10 = fpext float %gcfv10_f to double
  %gcfv11 = fpext float %gcfv11_f to double
  call i32 (ptr, ...) @printf(ptr @.fmt7, ptr @.s_cabsf_arg, double %gcf0, double %gcf1, double %fcf, double %fcfv0, double %fcfv1, double %gcfv10, double %gcfv11)

  ; cabsf with insertelement-constructed <2 x float>
  ; CHECK: call { float, float } @diffecabsf_ins(
  %g_cfi = call { float, float } (...) @__enzyme_autodiff_f2(ptr @cabsf_ins, float 3.000000e+00, float 4.000000e+00)
  %gcfi0_f = extractvalue { float, float } %g_cfi, 0
  %gcfi1_f = extractvalue { float, float } %g_cfi, 1
  %gcfi0 = fpext float %gcfi0_f to double
  %gcfi1 = fpext float %gcfi1_f to double
  ; CHECK: call fast float @fwddiffecabsf_ins(
  %fcfi_f = call float (...) @__enzyme_fwddiff_f(ptr @cabsf_ins, float 3.000000e+00, float 1.000000e+00, float 4.000000e+00, float 2.000000e+00)
  %fcfi = fpext float %fcfi_f to double
  call i32 (ptr, ...) @printf(ptr @.fmt3, ptr @.s_cabsf_ins, double %gcfi0, double %gcfi1, double %fcfi)

  ; cabs with direct [2 x double] parameter
  ; CHECK: call { [2 x double] } @diffecabs_arg(
  %g_cd = call { [2 x double] } (...) @__enzyme_autodiff_a2d(ptr @cabs_arg, [2 x double] [double 3.000000e+00, double 4.000000e+00])
  %gcd0 = extractvalue { [2 x double] } %g_cd, 0, 0
  %gcd1 = extractvalue { [2 x double] } %g_cd, 0, 1
  ; CHECK: call fast double @fwddiffecabs_arg(
  %fcd = call double (...) @__enzyme_fwddiff_d(ptr @cabs_arg, [2 x double] [double 3.000000e+00, double 4.000000e+00], [2 x double] [double 1.000000e+00, double 2.000000e+00])
  ; CHECK: call fast [2 x double] @fwddiffe2cabs_arg(
  %fcdv = call [2 x double] (...) @__enzyme_fwddiff_d_v2(ptr @cabs_arg, metadata !"enzyme_width", i64 2, [2 x double] [double 3.000000e+00, double 4.000000e+00], [2 x double] [double 1.000000e+00, double 2.000000e+00], [2 x double] [double -2.000000e+00, double 1.000000e+00])
  %fcdv0 = extractvalue [2 x double] %fcdv, 0
  %fcdv1 = extractvalue [2 x double] %fcdv, 1
  call i32 (ptr, ...) @printf(ptr @.fmt5, ptr @.s_cabs_arg, double %gcd0, double %gcd1, double %fcd, double %fcdv0, double %fcdv1)
  ret i32 0
}
