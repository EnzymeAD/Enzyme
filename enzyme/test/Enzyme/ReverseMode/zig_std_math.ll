; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,early-cse,simplifycfg)" -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="preserve-nvvm,inline,enzyme,preserve-nvvm-end,function(mem2reg,instsimplify,early-cse,simplifycfg)" -enzyme-preopt=false -S | FileCheck %s --check-prefix=PRESERVE; fi

declare float @__enzyme_autodiff(ptr, float)
declare double @__enzyme_autodiff_d(ptr, double)
declare { double, double } @__enzyme_autodiff_dd(ptr, double, double)
declare float @__enzyme_autodiff_fi(ptr, float, i32)

; Simulate Zig's std.math.pow(f32, x, y) body containing bit-twiddling frexp
; operations that cannot be differentiated directly if stepped into or inlined.
define internal fastcc float @math.pow.pow__anon_19626(float %x, float %y) {
entry:
  %bits = bitcast float %x to i32
  %cleared = and i32 %bits, -2139095041
  %mant = or i32 %cleared, 1056964608
  %f = bitcast i32 %mant to float
  %res = fmul float %f, %y
  ret float %res
}

define internal double @math.acos.acos__anon_42(double %x) {
entry:
  %bits = bitcast double %x to i64
  %cleared = and i64 %bits, -9218868437227405313
  %f = bitcast i64 %cleared to double
  ret double %f
}

define internal fastcc double @math.atan2.atan2_64(double %y, double %x) {
entry:
  %bits = bitcast double %y to i64
  %cleared = and i64 %bits, -9218868437227405313
  %f = bitcast i64 %cleared to double
  %r = fadd double %f, %x
  ret double %r
}

define internal float @math.ldexp.ldexp__anon_7(float %x, i32 %n) {
entry:
  %bits = bitcast float %x to i32
  %cleared = and i32 %bits, -2139095041
  %f = bitcast i32 %cleared to float
  ret float %f
}

; Integer instantiation must NOT be treated as floating-point pow.
define internal i32 @math.pow.pow__anon_999(i32 %x, i32 %y) {
entry:
  %r = mul i32 %x, %y
  ret i32 %r
}

define float @zig_pow2(float %x) {
entry:
  %r = call fastcc float @math.pow.pow__anon_19626(float %x, float 2.000000e+00)
  ret float %r
}

define double @zig_acos(double %x) {
entry:
  %r = call double @math.acos.acos__anon_42(double %x)
  ret double %r
}

define double @zig_atan2(double %y, double %x) {
entry:
  %r = call fastcc double @math.atan2.atan2_64(double %y, double %x)
  ret double %r
}

define float @zig_ldexp(float %x, i32 %n) {
entry:
  %r = call float @math.ldexp.ldexp__anon_7(float %x, i32 %n)
  ret float %r
}

define float @d_pow2(float %x) {
entry:
  %d = call float @__enzyme_autodiff(ptr @zig_pow2, float %x)
  ret float %d
}

define double @d_acos(double %x) {
entry:
  %d = call double @__enzyme_autodiff_d(ptr @zig_acos, double %x)
  ret double %d
}

define { double, double } @d_atan2(double %y, double %x) {
entry:
  %d = call { double, double } @__enzyme_autodiff_dd(ptr @zig_atan2, double %y, double %x)
  ret { double, double } %d
}

define float @d_ldexp(float %x, i32 %n) {
entry:
  %d = call float @__enzyme_autodiff_fi(ptr @zig_ldexp, float %x, i32 %n)
  ret float %d
}

; CHECK-LABEL: define internal fastcc float @math.pow.pow__anon_19626(
; CHECK: define internal i32 @math.pow.pow__anon_999(i32 %x, i32 %y) #[[INT_ATTR:[0-9]+]]

; CHECK-LABEL: define internal { float } @diffezig_pow2(float %x, float %differeturn)
; CHECK:   %[[POW:.+]] = call fast fastcc float @math.pow.pow__anon_19626(float %x, float 1.000000e+00)
; CHECK:   %[[SCALED:.+]] = fmul fast float 2.000000e+00, %[[POW]]
; CHECK:   %[[DR:.+]] = fmul fast float %differeturn, %[[SCALED]]
; CHECK:   ret { float }

; CHECK-LABEL: define internal { double } @diffezig_acos(double %x, double %differeturn)
; CHECK:   %[[SQ:.+]] = fmul fast double %x, %x
; CHECK:   %[[SUB:.+]] = fsub fast double 1.000000e+00, %[[SQ]]
; CHECK:   %[[SQRT:.+]] = call fast double @llvm.sqrt.f64(double %[[SUB]])
; CHECK:   %[[DIV:.+]] = fdiv fast double %differeturn, %[[SQRT]]
; CHECK:   %[[NEG:.+]] = fneg fast double %[[DIV]]
; CHECK:   ret { double }

; CHECK-LABEL: define internal { double, double } @diffezig_atan2(double %y, double %x, double %differeturn)
; CHECK:   %[[X2:.+]] = fmul fast double %x, %x
; CHECK:   %[[Y2:.+]] = fmul fast double %y, %y
; CHECK:   %[[DEN:.+]] = fadd fast double %[[X2]], %[[Y2]]
; CHECK:   ret { double, double }

; CHECK-LABEL: define internal { float } @diffezig_ldexp(float %x, i32 %n, float %differeturn)
; CHECK:   call fast float @math.ldexp.ldexp__anon_7(float %differeturn, i32 %n)
; CHECK:   ret { float }

; CHECK: attributes #{{[0-9]+}} = { {{.*}}"enzyme_math"="powf"
; CHECK: attributes #[[INT_ATTR]] = { nounwind "enzyme_ReadOnlyOrThrow" }

; PRESERVE-LABEL: define internal fastcc float @math.pow.pow__anon_19626(
; PRESERVE-LABEL: define internal { float } @diffezig_pow2(float %x, float %differeturn)
; PRESERVE:   %[[PPOW:.+]] = call fast fastcc float @math.pow.pow__anon_19626(float %x, float 1.000000e+00)
; PRESERVE:   %[[PSCALED:.+]] = fmul fast float 2.000000e+00, %[[PPOW]]
; PRESERVE:   %[[PDR:.+]] = fmul fast float %differeturn, %[[PSCALED]]
; PRESERVE:   ret { float }
