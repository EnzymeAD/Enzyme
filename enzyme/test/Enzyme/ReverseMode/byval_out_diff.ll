; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | %lli - | FileCheck %s --check-prefix=EVAL
; EVAL: 3.0 70.0 2.0 5.0 11.0 3.0 40.0 9.0 6.0

%struct.double2 = type { double, double }
%struct.float2 = type { float, float }
%struct.float3 = type { float, float, float }

@enzyme_out = global i32 0, align 4
@enzyme_width = global i32 0, align 4
@.str = private unnamed_addr constant [46 x i8] c"%.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f\0A\00", align 1

declare i32 @printf(ptr, ...)
declare { double, double } @__enzyme_autodiff_d2(...)
declare { <2 x float>, float } @__enzyme_autodiff_f3(...)
declare { ptr, float } @__enzyme_augmentfwd(...)
declare <2 x float> @__enzyme_reverse_f2(...)
declare [2 x %struct.float2] @__enzyme_reverse_w2(...)

define double @loss_d2(ptr nocapture readonly byval(%struct.double2) align 8 %pair) {
entry:
  %a = load double, ptr %pair, align 8
  %b_ptr = getelementptr inbounds %struct.double2, ptr %pair, i64 0, i32 1
  %b = load double, ptr %b_ptr, align 8
  %ta = fmul double %a, 3.000000e+00
  %b2 = fmul double %b, %b
  %tb = fmul double %b2, 7.000000e+00
  %sum = fadd double %ta, %tb
  ret double %sum
}

; CHECK-LABEL: define { double, double } @grad_d2(
define { double, double } @grad_d2(ptr %pair_ptr) {
entry:
  %out = load i32, ptr @enzyme_out, align 4
; CHECK: [[D2CALL:%.+]] = call {{(fast )?}}{ %struct.double2 } @diffeloss_d2(ptr byval(%struct.double2) %pair_ptr, double 1.000000e+00)
; CHECK: extractvalue { %struct.double2 } [[D2CALL]], 0
  %grad = call { double, double } (...) @__enzyme_autodiff_d2(ptr @loss_d2, i32 %out, ptr byval(%struct.double2) align 8 %pair_ptr)
  ret { double, double } %grad
}

define float @loss_f3(ptr nocapture readonly byval(%struct.float3) align 4 %vec) {
entry:
  %x = load float, ptr %vec, align 4
  %y_ptr = getelementptr inbounds %struct.float3, ptr %vec, i64 0, i32 1
  %y = load float, ptr %y_ptr, align 4
  %z_ptr = getelementptr inbounds %struct.float3, ptr %vec, i64 0, i32 2
  %z = load float, ptr %z_ptr, align 4
  %tx = fmul float %x, 2.000000e+00
  %ty = fmul float %y, 5.000000e+00
  %tz = fmul float %z, 1.100000e+01
  %sxy = fadd float %tx, %ty
  %sxyz = fadd float %sxy, %tz
  ret float %sxyz
}

; CHECK-LABEL: define { <2 x float>, float } @grad_f3(
define { <2 x float>, float } @grad_f3(ptr %vec_ptr) {
entry:
  %out = load i32, ptr @enzyme_out, align 4
; CHECK: [[F3SLOT:%.+]] = alloca [16 x i8], align 8
; CHECK: [[F3CALL:%.+]] = call {{(fast )?}}{ %struct.float3 } @diffeloss_f3(ptr byval(%struct.float3) %vec_ptr, float 1.000000e+00)
; CHECK: store { %struct.float3 } [[F3CALL]], ptr [[F3SLOT]], align 8
; CHECK: load { <2 x float>, float }, ptr [[F3SLOT]], align 8
  %grad = call { <2 x float>, float } (...) @__enzyme_autodiff_f3(ptr @loss_f3, i32 %out, ptr byval(%struct.float3) align 4 %vec_ptr)
  ret { <2 x float>, float } %grad
}

define float @loss_lin2(ptr nocapture readonly byval(%struct.float2) align 4 %vec) {
entry:
  %x = load float, ptr %vec, align 4
  %y_ptr = getelementptr inbounds %struct.float2, ptr %vec, i64 0, i32 1
  %y = load float, ptr %y_ptr, align 4
  %tx = fmul float %x, 3.000000e+00
  %ty = fmul float %y, 4.000000e+00
  %sum = fadd float %tx, %ty
  ret float %sum
}

; CHECK-LABEL: define [2 x %struct.float2] @grad_width2(
define [2 x %struct.float2] @grad_width2(ptr %vec_ptr) {
entry:
  %w = load i32, ptr @enzyme_width, align 4
  %out = load i32, ptr @enzyme_out, align 4
  %aug = call { ptr, float } (...) @__enzyme_augmentfwd(ptr @loss_lin2, i32 %w, i32 2, i32 %out, ptr byval(%struct.float2) align 4 %vec_ptr)
  %tape = extractvalue { ptr, float } %aug, 0
; CHECK: [[W2CALL:%.+]] = call {{(fast )?}}{ [2 x %struct.float2] } @diffe2loss_lin2(ptr byval(%struct.float2) %vec_ptr, [2 x float] [float 1.000000e+00, float 1.000000e+01], ptr {{.+}})
; CHECK: extractvalue { [2 x %struct.float2] } [[W2CALL]], 0
  %grad = call [2 x %struct.float2] (...) @__enzyme_reverse_w2(ptr @loss_lin2, i32 %w, i32 2, i32 %out, ptr byval(%struct.float2) align 4 %vec_ptr, [2 x float] [float 1.000000e+00, float 1.000000e+01], ptr %tape)
  ret [2 x %struct.float2] %grad
}

define float @loss_prod2(ptr nocapture readonly byval(%struct.float2) align 4 %vec) {
entry:
  %x = load float, ptr %vec, align 4
  %y_ptr = getelementptr inbounds %struct.float2, ptr %vec, i64 0, i32 1
  %y = load float, ptr %y_ptr, align 4
  %prod = fmul float %x, %y
  ret float %prod
}

; CHECK-LABEL: define <2 x float> @grad_split(
define <2 x float> @grad_split(ptr %vec_ptr) {
entry:
  %out = load i32, ptr @enzyme_out, align 4
  %aug = call { ptr, float } (...) @__enzyme_augmentfwd(ptr @loss_prod2, i32 %out, ptr byval(%struct.float2) align 4 %vec_ptr)
  %tape = extractvalue { ptr, float } %aug, 0
; CHECK: [[F2SLOT:%.+]] = alloca [8 x i8], align 8
; CHECK: [[F2CALL:%.+]] = call {{(fast )?}}{ %struct.float2 } @diffeloss_prod2(ptr byval(%struct.float2) %vec_ptr, float 1.000000e+00, ptr {{.+}})
; CHECK: store { %struct.float2 } [[F2CALL]], ptr [[F2SLOT]], align 8
; CHECK: load <2 x float>, ptr [[F2SLOT]], align 8
  %grad = call <2 x float> (...) @__enzyme_reverse_f2(ptr @loss_prod2, i32 %out, ptr byval(%struct.float2) align 4 %vec_ptr, float 1.000000e+00, ptr %tape)
  ret <2 x float> %grad
}

define i32 @main() {
entry:
  %d2_slot = alloca %struct.double2, align 8
  store double 2.000000e+00, ptr %d2_slot, align 8
  %d2_y = getelementptr inbounds %struct.double2, ptr %d2_slot, i64 0, i32 1
  store double 5.000000e+00, ptr %d2_y, align 8
  %gd2 = call { double, double } @grad_d2(ptr %d2_slot)
  %gd2_0 = extractvalue { double, double } %gd2, 0
  %gd2_1 = extractvalue { double, double } %gd2, 1

  %f3_slot = alloca %struct.float3, align 4
  store float 1.000000e+00, ptr %f3_slot, align 4
  %f3_y = getelementptr inbounds %struct.float3, ptr %f3_slot, i64 0, i32 1
  store float 2.000000e+00, ptr %f3_y, align 4
  %f3_z = getelementptr inbounds %struct.float3, ptr %f3_slot, i64 0, i32 2
  store float 3.000000e+00, ptr %f3_z, align 4
  %gf3 = call { <2 x float>, float } @grad_f3(ptr %f3_slot)
  %gf3_01 = extractvalue { <2 x float>, float } %gf3, 0
  %gf3_x = extractelement <2 x float> %gf3_01, i64 0
  %gf3_y = extractelement <2 x float> %gf3_01, i64 1
  %gf3_z = extractvalue { <2 x float>, float } %gf3, 1
  %gf3_xd = fpext float %gf3_x to double
  %gf3_yd = fpext float %gf3_y to double
  %gf3_zd = fpext float %gf3_z to double

  %w2_slot = alloca %struct.float2, align 4
  store float 7.000000e+00, ptr %w2_slot, align 4
  %w2_y = getelementptr inbounds %struct.float2, ptr %w2_slot, i64 0, i32 1
  store float 8.000000e+00, ptr %w2_y, align 4
  %gw2 = call [2 x %struct.float2] @grad_width2(ptr %w2_slot)
  %gw2_0x = extractvalue [2 x %struct.float2] %gw2, 0, 0
  %gw2_1y = extractvalue [2 x %struct.float2] %gw2, 1, 1
  %gw2_0xd = fpext float %gw2_0x to double
  %gw2_1yd = fpext float %gw2_1y to double

  %sp_slot = alloca %struct.float2, align 4
  store float 6.000000e+00, ptr %sp_slot, align 4
  %sp_y = getelementptr inbounds %struct.float2, ptr %sp_slot, i64 0, i32 1
  store float 9.000000e+00, ptr %sp_y, align 4
  %gsp = call <2 x float> @grad_split(ptr %sp_slot)
  %gsp_x = extractelement <2 x float> %gsp, i64 0
  %gsp_y = extractelement <2 x float> %gsp, i64 1
  %gsp_xd = fpext float %gsp_x to double
  %gsp_yd = fpext float %gsp_y to double

  call i32 (ptr, ...) @printf(ptr @.str, double %gd2_0, double %gd2_1, double %gf3_xd, double %gf3_yd, double %gf3_zd, double %gw2_0xd, double %gw2_1yd, double %gsp_xd, double %gsp_yd)
  ret i32 0
}
