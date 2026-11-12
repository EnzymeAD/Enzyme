; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -simplifycfg -instsimplify -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,%simplifycfg,instsimplify,adce)" -S | FileCheck %s

@enzyme_width = external global i32
@enzyme_out = external global i32
@enzyme_dup = external global i32

declare { double, double } @csin(double, double)
declare <2 x float> @csinf(<2 x float>)
declare float @cabsf(<2 x float>)

define double @test_csin_re(double %x, double %y) {
entry:
  %r = call { double, double } @csin(double %x, double %y)
  %re = extractvalue { double, double } %r, 0
  ret double %re
}

define { double, double } @test_csin(double %x, double %y) {
entry:
  %r = call { double, double } @csin(double %x, double %y)
  ret { double, double } %r
}

define float @test_vec(<2 x float> %z) {
entry:
  %s = call <2 x float> @csinf(<2 x float> %z)
  %m = call float @cabsf(<2 x float> %s)
  ret float %m
}

declare { [2 x double], [2 x double] } @__enzyme_autodiff_s(...)
declare [2 x <2 x float>] @__enzyme_autodiff_v(...)
declare [2 x { double, double }] @__enzyme_fwddiff_s(...)
declare [2 x float] @__enzyme_fwddiff_v(...)

define { [2 x double], [2 x double] } @rev2_csin(double %x, double %y) {
  %w = load i32, i32* @enzyme_width
  %o = load i32, i32* @enzyme_out
  %r = call { [2 x double], [2 x double] } (...) @__enzyme_autodiff_s(double (double, double)* nonnull @test_csin_re, i32 %w, i32 2, i32 %o, double %x, i32 %o, double %y)
  ret { [2 x double], [2 x double] } %r
}

define [2 x <2 x float>] @rev2_vec(<2 x float> %z) {
  %w = load i32, i32* @enzyme_width
  %o = load i32, i32* @enzyme_out
  %r = call [2 x <2 x float>] (...) @__enzyme_autodiff_v(float (<2 x float>)* nonnull @test_vec, i32 %w, i32 2, i32 %o, <2 x float> %z)
  ret [2 x <2 x float>] %r
}

define [2 x { double, double }] @fwd2_csin(double %x, double %dx0, double %dx1, double %y, double %dy0, double %dy1) {
  %w = load i32, i32* @enzyme_width
  %d = load i32, i32* @enzyme_dup
  %r = call [2 x { double, double }] (...) @__enzyme_fwddiff_s({ double, double } (double, double)* nonnull @test_csin, i32 %w, i32 2, i32 %d, double %x, double %dx0, double %dx1, i32 %d, double %y, double %dy0, double %dy1)
  ret [2 x { double, double }] %r
}

define [2 x float] @fwd2_vec(<2 x float> %z, <2 x float> %dz0, <2 x float> %dz1) {
  %w = load i32, i32* @enzyme_width
  %d = load i32, i32* @enzyme_dup
  %r = call [2 x float] (...) @__enzyme_fwddiff_v(float (<2 x float>)* nonnull @test_vec, i32 %w, i32 2, i32 %d, <2 x float> %z, <2 x float> %dz0, <2 x float> %dz1)
  ret [2 x float] %r
}

; CHECK: define internal { [2 x double], [2 x double] } @diffe2test_csin_re(double %x, double %y, [2 x double] %differeturn)
; CHECK: call { double, double } @ccos(double %x, double %y)
; CHECK: ret { [2 x double], [2 x double] }

; CHECK: define internal { [2 x <2 x float>] } @diffe2test_vec(<2 x float> %z, [2 x float] %differeturn)
; CHECK: call fast float @cabsf(<2 x float>
; CHECK: call fast <2 x float> @ccosf(<2 x float>
; CHECK: ret { [2 x <2 x float>] }

; CHECK: define internal [2 x { double, double }] @fwddiffe2test_csin(double %x, [2 x double] %"x'", double %y, [2 x double] %"y'")
; CHECK: call { double, double } @ccos(double %x, double %y)
; CHECK: ret [2 x { double, double }]

; CHECK: define internal [2 x float] @fwddiffe2test_vec(<2 x float> %z, [2 x <2 x float>] %"z'")
; CHECK: call fast <2 x float> @ccosf(<2 x float>
; CHECK: call fast float @cabsf(<2 x float>
; CHECK: ret [2 x float]
