; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | %lli - | FileCheck %s --check-prefix=EVAL
; EVAL: 7.0 40.0 8.0 20.0 33.0 1064.0

%struct.float2 = type { float, float }
%struct.float3 = type { float, float, float }
%struct.padded = type { i64, i32 }

@enzyme_const = global i32 0, align 4
@enzyme_out = global i32 0, align 4
@enzyme_dup = global i32 0, align 4
@enzyme_interleave = global i32 0, align 4
@.str = private unnamed_addr constant [31 x i8] c"%.1f %.1f %.1f %.1f %.1f %.1f\0A\00", align 1

declare i32 @printf(ptr, ...)
declare double @__enzyme_autodiff(...)
declare { <2 x float>, float } @__enzyme_autodiff_f3(...)
declare float @__enzyme_fwddiff(...)

define double @loss_callee_byval(double %x, ptr nocapture readonly byval(%struct.float2) align 8 %scale) {
entry:
  %sx = load float, ptr %scale, align 8
  %sy_ptr = getelementptr inbounds %struct.float2, ptr %scale, i64 0, i32 1
  %sy = load float, ptr %sy_ptr, align 4
  %sum_f = fadd float %sx, %sy
  %sum_d = fpext float %sum_f to double
  %prod = fmul double %x, %sum_d
  ret double %prod
}

; CHECK-LABEL: define double @grad_callee_byval(
define double @grad_callee_byval(double %x, i64 %coerced_scale) {
entry:
  %out = load i32, ptr @enzyme_out, align 4
  %cst = load i32, ptr @enzyme_const, align 4
; CHECK: [[SLOT:%.+]] = alloca %struct.float2, align 8
; CHECK: store i64 %coerced_scale, ptr [[SLOT]], align 8
; CHECK: call {{(fast )?}}{ double } @diffeloss_callee_byval(double %x, ptr byval(%struct.float2) [[SLOT]], double 1.000000e+00)
  %dx = call double (...) @__enzyme_autodiff(ptr @loss_callee_byval, i32 %out, double %x, i32 %cst, i64 %coerced_scale)
  ret double %dx
}

define float @fwd_callee_byval(ptr nocapture readonly byval(%struct.float3) align 4 %vec) {
entry:
  %vx = load float, ptr %vec, align 4
  %vy_ptr = getelementptr inbounds %struct.float3, ptr %vec, i64 0, i32 1
  %vy = load float, ptr %vy_ptr, align 4
  %vz_ptr = getelementptr inbounds %struct.float3, ptr %vec, i64 0, i32 2
  %vz = load float, ptr %vz_ptr, align 4
  %xy = fmul float %vx, %vy
  %res = fadd float %xy, %vz
  ret float %res
}

; CHECK-LABEL: define float @jvp_callee_byval(
define float @jvp_callee_byval(<2 x float> %v01, float %v2, <2 x float> %dv01, float %dv2) {
entry:
  %dup = load i32, ptr @enzyme_dup, align 4
; CHECK: [[SSLOT:%.+]] = alloca %struct.float3, align 8
; CHECK: [[PSLOT:%.+]] = alloca %struct.float3, align 8
; CHECK: store <2 x float> %v01, ptr [[PSLOT]], align 8
; CHECK: store <2 x float> %dv01, ptr [[SSLOT]], align 8
; CHECK: call {{(fast )?}}float @fwddiffefwd_callee_byval(ptr byval(%struct.float3) [[PSLOT]], ptr [[SSLOT]])
  %dv = call float (...) @__enzyme_fwddiff(ptr @fwd_callee_byval, i32 %dup, <2 x float> %v01, float %v2, <2 x float> %dv01, float %dv2)
  ret float %dv
}

define float @loss_caller_byval(<2 x float> %v01, float %v2, double %weight) {
entry:
  %v0 = extractelement <2 x float> %v01, i64 0
  %v1 = extractelement <2 x float> %v01, i64 1
  %t0 = fmul float %v0, 2.000000e+00
  %t1 = fmul float %v1, 3.000000e+00
  %t2 = fmul float %v2, 5.000000e+00
  %s01 = fadd float %t0, %t1
  %s012 = fadd float %s01, %t2
  %wf = fptrunc double %weight to float
  %scaled = fmul float %s012, %wf
  ret float %scaled
}

; CHECK-LABEL: define { <2 x float>, float } @grad_caller_byval(
define { <2 x float>, float } @grad_caller_byval(ptr %vec_ptr, double %weight) {
entry:
  %out = load i32, ptr @enzyme_out, align 4
  %cst = load i32, ptr @enzyme_const, align 4
; CHECK: [[V01:%.+]] = load <2 x float>, ptr %vec_ptr, align 4
; CHECK: [[GEP8:%.+]] = getelementptr inbounds i8, ptr %vec_ptr, i64 8
; CHECK: [[V2:%.+]] = load float, ptr [[GEP8]], align 4
; CHECK: call {{(fast )?}}{ <2 x float>, float } @diffeloss_caller_byval(<2 x float> [[V01]], float [[V2]], double %weight, float 1.000000e+00)
  %grad = call { <2 x float>, float } (...) @__enzyme_autodiff_f3(ptr @loss_caller_byval, i32 %out, ptr byval(%struct.float3) align 4 %vec_ptr, i32 %cst, double %weight)
  ret { <2 x float>, float } %grad
}

; CHECK-LABEL: define float @jvp_interleaved_shadow_byval(
define float @jvp_interleaved_shadow_byval(<2 x float> %v01, float %v2, double %weight, ptr %dvec_ptr) {
entry:
  %dup = load i32, ptr @enzyme_dup, align 4
  %cst = load i32, ptr @enzyme_const, align 4
  %interleave = load i32, ptr @enzyme_interleave, align 4
; CHECK: [[DV01:%.+]] = load <2 x float>, ptr %dvec_ptr, align 4
; CHECK: [[DGEP8:%.+]] = getelementptr inbounds i8, ptr %dvec_ptr, i64 8
; CHECK: [[DV2:%.+]] = load float, ptr [[DGEP8]], align 4
; CHECK: call {{(fast )?}}float @fwddiffeloss_caller_byval(<2 x float> %v01, <2 x float> [[DV01]], float %v2, float [[DV2]], double %weight)
  %dv = call float (...) @__enzyme_fwddiff(ptr @loss_caller_byval, i32 %dup, <2 x float> %v01, float %v2, i32 %cst, double %weight, i32 %interleave, ptr byval(%struct.float3) align 4 %dvec_ptr)
  ret float %dv
}

define double @padded_caller_byval(i64 %first, i32 %second, i32 %trailing, double %x) {
entry:
  %second_ext = sext i32 %second to i64
  %trailing_ext = sext i32 %trailing to i64
  %sum0 = add i64 %first, %second_ext
  %sum1 = add i64 %sum0, %trailing_ext
  %coeff = sitofp i64 %sum1 to double
  %prod = fmul double %coeff, %x
  ret double %prod
}

; CHECK-LABEL: define double @grad_padded_caller_byval(
define double @grad_padded_caller_byval(ptr %pad_ptr, i32 %trailing, double %x) {
entry:
  %cst = load i32, ptr @enzyme_const, align 4
  %out = load i32, ptr @enzyme_out, align 4
; CHECK: [[P0:%.+]] = load i64, ptr %pad_ptr, align 8
; CHECK: [[PGEP:%.+]] = getelementptr inbounds i8, ptr %pad_ptr, i64 8
; CHECK: [[P1:%.+]] = load i32, ptr [[PGEP]], align 8
; CHECK: call {{(fast )?}}{ double } @diffepadded_caller_byval(i64 [[P0]], i32 [[P1]], i32 %trailing, double %x, double 1.000000e+00)
  %dx = call double (...) @__enzyme_autodiff(ptr @padded_caller_byval, i32 %cst, ptr byval(%struct.padded) align 8 %pad_ptr, i32 %cst, i32 %trailing, i32 %out, double %x)
  ret double %dx
}

define i32 @main() {
entry:
  %scale_slot = alloca %struct.float2, align 8
  store float 2.000000e+00, ptr %scale_slot, align 8
  %scale_y = getelementptr inbounds %struct.float2, ptr %scale_slot, i64 0, i32 1
  store float 5.000000e+00, ptr %scale_y, align 4
  %packed_scale = load i64, ptr %scale_slot, align 8
  %g1 = call double @grad_callee_byval(double 3.000000e+00, i64 %packed_scale)

  %jvp = call float @jvp_callee_byval(<2 x float> <float 2.000000e+00, float 3.000000e+00>, float 4.000000e+00, <2 x float> <float 5.000000e+00, float 7.000000e+00>, float 1.100000e+01)
  %jvp_d = fpext float %jvp to double

  %vec_slot = alloca %struct.float3, align 4
  store float 1.000000e+00, ptr %vec_slot, align 4
  %vec_y = getelementptr inbounds %struct.float3, ptr %vec_slot, i64 0, i32 1
  store float 2.000000e+00, ptr %vec_y, align 4
  %vec_z = getelementptr inbounds %struct.float3, ptr %vec_slot, i64 0, i32 2
  store float 3.000000e+00, ptr %vec_z, align 4
  %gf3 = call { <2 x float>, float } @grad_caller_byval(ptr %vec_slot, double 4.000000e+00)
  %gf3_01 = extractvalue { <2 x float>, float } %gf3, 0
  %gf3_x = extractelement <2 x float> %gf3_01, i64 0
  %gf3_z = extractvalue { <2 x float>, float } %gf3, 1
  %gf3_xd = fpext float %gf3_x to double
  %gf3_zd = fpext float %gf3_z to double

  %dvec_slot = alloca %struct.float3, align 4
  store float 1.000000e+00, ptr %dvec_slot, align 4
  %dvec_y = getelementptr inbounds %struct.float3, ptr %dvec_slot, i64 0, i32 1
  store float 1.000000e+01, ptr %dvec_y, align 4
  %dvec_z = getelementptr inbounds %struct.float3, ptr %dvec_slot, i64 0, i32 2
  store float 1.000000e+02, ptr %dvec_z, align 4
  %jvp_int = call float @jvp_interleaved_shadow_byval(<2 x float> <float 1.000000e+00, float 2.000000e+00>, float 3.000000e+00, double 2.000000e+00, ptr %dvec_slot)
  %jvp_int_d = fpext float %jvp_int to double

  %pad_slot = alloca %struct.padded, align 8
  store i64 10, ptr %pad_slot, align 8
  %pad_b = getelementptr inbounds %struct.padded, ptr %pad_slot, i64 0, i32 1
  store i32 20, ptr %pad_b, align 8
  %pad_tail = getelementptr inbounds i8, ptr %pad_slot, i64 12
  store i32 999, ptr %pad_tail, align 4
  %gpad = call double @grad_padded_caller_byval(ptr %pad_slot, i32 3, double 2.000000e+00)

  call i32 (ptr, ...) @printf(ptr @.str, double %g1, double %jvp_d, double %gf3_xd, double %gf3_zd, double %gpad, double %jvp_int_d)
  ret i32 0
}
