; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(instsimplify)" -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -S | %lli -; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="default<O2>,enzyme,default<O2>" -S | %lli -; fi

declare float @atan2f(float, float) nounwind readnone
declare float @__enzyme_fwddiff(...)
declare <2 x float> @__enzyme_fwddiff_vec(...)
declare void @__enzyme_autodiff(...)

; CHECK-LABEL: define internal float @fwddifferatio(
; CHECK-SAME: float %x, float %"x'", float %y, float %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[QUOT:.+]] = fdiv fast float %x, %y
; CHECK-NEXT:   %[[DYQ:.+]] = fmul fast float %"y'", %[[QUOT]]
; CHECK-NEXT:   %[[NUM:.+]] = fsub fast float %"x'", %[[DYQ]]
; CHECK-NEXT:   %[[RES:.+]] = fdiv fast float %[[NUM]], %y
; CHECK-NEXT:   ret float %[[RES]]
define float @ratio(float %x, float %y) {
entry:
  %quotient = fdiv float %x, %y
  ret float %quotient
}

; CHECK-LABEL: define internal <2 x float> @fwddiffevector_ratio(
; CHECK-SAME: <2 x float> %x, <2 x float> %"x'", <2 x float> %y, <2 x float> %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[VQUOT:.+]] = fdiv fast <2 x float> %x, %y
; CHECK-NEXT:   %[[VDYQ:.+]] = fmul fast <2 x float> %"y'", %[[VQUOT]]
; CHECK-NEXT:   %[[VNUM:.+]] = fsub fast <2 x float> %"x'", %[[VDYQ]]
; CHECK-NEXT:   %[[VRES:.+]] = fdiv fast <2 x float> %[[VNUM]], %y
; CHECK-NEXT:   ret <2 x float> %[[VRES]]
define <2 x float> @vector_ratio(<2 x float> %x, <2 x float> %y) {
entry:
  %quotient = fdiv fast <2 x float> %x, %y
  ret <2 x float> %quotient
}

define float @direct_ratio_derivative(float %x, float %dx, float %y, float %dy) noinline {
entry:
  %result = call float (...) @__enzyme_fwddiff(ptr @ratio, metadata !"enzyme_dup", float %x, float %dx, metadata !"enzyme_dup", float %y, float %dy)
  ret float %result
}

define <2 x float> @direct_vector_ratio_derivative(<2 x float> %x, <2 x float> %dx, <2 x float> %y, <2 x float> %dy) noinline {
entry:
  %result = call <2 x float> (...) @__enzyme_fwddiff_vec(ptr @vector_ratio, metadata !"enzyme_dup", <2 x float> %x, <2 x float> %dx, metadata !"enzyme_dup", <2 x float> %y, <2 x float> %dy)
  ret <2 x float> %result
}

define float @polar_objective(ptr nocapture readonly %values, float %scale) {
entry:
  %v0 = load float, ptr %values, align 4
  %v1_ptr = getelementptr inbounds float, ptr %values, i64 1
  %v1 = load float, ptr %v1_ptr, align 4
  %y0 = fmul float %scale, %v0
  %y1 = fmul float %scale, %v1
  %angle0 = call float @atan2f(float %y0, float 1.000000e+00)
  %angle1 = call float @atan2f(float %y1, float 1.000000e+00)
  %square0 = fmul float %v0, %v0
  %square1 = fmul float %v1, %v1
  %term0 = fadd float %square0, %angle0
  %term1 = fadd float %square1, %angle1
  %total = fadd float %term0, %term1
  ret float %total
}

define void @gradient_polar(ptr %values, ptr %gradients, float %scale) {
entry:
  call void (...) @__enzyme_autodiff(ptr @polar_objective, metadata !"enzyme_dup", ptr %values, ptr %gradients, metadata !"enzyme_const", float %scale)
  ret void
}

define void @hessian_vector_polar(ptr %values, ptr %tangents, ptr %gradients, ptr %hessian, float %scale) noinline {
entry:
  call void (...) @__enzyme_fwddiff(ptr @gradient_polar, metadata !"enzyme_dup", ptr %values, ptr %tangents, metadata !"enzyme_dup", ptr %gradients, ptr %hessian, metadata !"enzyme_const", float %scale)
  ret void
}

define i32 @main() {
entry:
  ; Direct forward-mode overflow check:
  ; x = 3 * 2^70, dx = 1.0, y = 2^70, dy = 1.0 => (1 - 3) / 2^70 = -2^-69
  %deriv_large = call float @direct_ratio_derivative(float 0x4468000000000000, float 1.000000e+00, float 0x4450000000000000, float 1.000000e+00)
  %ok_large = fcmp oeq float %deriv_large, 0xBBA0000000000000
  br i1 %ok_large, label %check_small, label %fail

check_small:
  ; Direct forward-mode underflow check:
  ; x = 3 * 2^-70, dx = 2^-70, y = 2^-70, dy = 2^-70 => (2^-70 - 3 * 2^-70) / 2^-70 = -2.0
  %deriv_small = call float @direct_ratio_derivative(float 0x3BA8000000000000, float 0x3B90000000000000, float 0x3B90000000000000, float 0x3B90000000000000)
  %ok_small = fcmp oeq float %deriv_small, -2.000000e+00
  br i1 %ok_small, label %check_vector, label %fail

check_vector:
  ; Vector fast fdiv lowers via rcpps Newton-Raphson on x86, turning +inf into -NaN
  ; if the denominator is squared. Scaling the derivative by 2^70 yields -2.0 on both lanes.
  %deriv_vec = call <2 x float> @direct_vector_ratio_derivative(<2 x float> <float 0x4468000000000000, float 0x4460000000000000>, <2 x float> <float 1.000000e+00, float 0.000000e+00>, <2 x float> <float 0x4450000000000000, float 0x4450000000000000>, <2 x float> <float 1.000000e+00, float 1.000000e+00>)
  %scaled_vec = fmul <2 x float> %deriv_vec, <float 0x4450000000000000, float 0x4450000000000000>
  %vec_lane0 = extractelement <2 x float> %scaled_vec, i64 0
  %vec_lane1 = extractelement <2 x float> %scaled_vec, i64 1
  %ok_vec0_lo = fcmp ogt float %vec_lane0, -2.00390625e+00
  %ok_vec0_hi = fcmp olt float %vec_lane0, -1.99609375e+00
  %ok_vec1_lo = fcmp ogt float %vec_lane1, -2.00390625e+00
  %ok_vec1_hi = fcmp olt float %vec_lane1, -1.99609375e+00
  %ok_vec0 = and i1 %ok_vec0_lo, %ok_vec0_hi
  %ok_vec1 = and i1 %ok_vec1_lo, %ok_vec1_hi
  %ok_vec = and i1 %ok_vec0, %ok_vec1
  br i1 %ok_vec, label %check_hessian, label %fail

check_hessian:
  ; Forward-over-reverse Hessian check where atan2f's reverse pass divides by
  ; 1 + y^2 = 2^72 (finite in binary32, whereas (2^72)^2 overflows to +inf).
  %values = alloca [2 x float], align 8
  %tangents = alloca [2 x float], align 8
  %gradients = alloca [2 x float], align 8
  %hessian = alloca [2 x float], align 8

  %values1 = getelementptr inbounds float, ptr %values, i64 1
  store float 1.000000e+00, ptr %values, align 4
  store float 2.000000e+00, ptr %values1, align 4

  %tangents1 = getelementptr inbounds float, ptr %tangents, i64 1
  store float 1.000000e+00, ptr %tangents, align 4
  store float 1.000000e+00, ptr %tangents1, align 4

  %gradients1 = getelementptr inbounds float, ptr %gradients, i64 1
  store float 0.000000e+00, ptr %gradients, align 4
  store float 0.000000e+00, ptr %gradients1, align 4

  %hessian1 = getelementptr inbounds float, ptr %hessian, i64 1
  store float 0.000000e+00, ptr %hessian, align 4
  store float 0.000000e+00, ptr %hessian1, align 4

  ; scale = 2^36 = 68719476736.0 => y0^2 = 2^72, y1^2 = 2^74
  call void @hessian_vector_polar(ptr %values, ptr %tangents, ptr %gradients, ptr %hessian, float 0x4230000000000000)

  %h0 = load float, ptr %hessian, align 4
  %h1 = load float, ptr %hessian1, align 4
  %ok_h0 = fcmp oeq float %h0, 2.000000e+00
  %ok_h1 = fcmp oeq float %h1, 2.000000e+00
  %ok_hessian = and i1 %ok_h0, %ok_h1
  br i1 %ok_hessian, label %pass, label %fail

pass:
  ret i32 0

fail:
  ret i32 1
}
