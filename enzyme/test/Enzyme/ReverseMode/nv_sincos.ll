; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -sroa -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,sroa,instsimplify,%simplifycfg)" -S | FileCheck %s

; Function Attrs: nounwind
declare void @__nv_sincos(double, double*, double*)
declare void @__nv_sincosf(float, float*, float*)

define void @tester(double %x, double* %s_out, double* %c_out) {
entry:
  call void @__nv_sincos(double %x, double* %s_out, double* %c_out)
  ret void
}

define double @test_derivative(double %x) {
entry:
  %s = alloca double
  %c = alloca double
  %ds = alloca double
  %dc = alloca double
  
  store double 1.0, double* %ds
  store double 2.0, double* %dc
  
  %0 = call double (...) @__enzyme_autodiff(void (double, double*, double*)* @tester, double %x, double* %s, double* %ds, double* %c, double* %dc)
  
  ret double %0
}

define float @tester_f(float %x, float* %s_out, float* %c_out) {
entry:
  call void @__nv_sincosf(float %x, float* %s_out, float* %c_out)
  ret float %x
}

define float @test_derivative_f(float %x) {
entry:
  %s = alloca float
  %c = alloca float
  %ds = alloca float
  %dc = alloca float
  
  store float 1.0, float* %ds
  store float 2.0, float* %dc
  
  %0 = call float (...) @__enzyme_autodiff(float (float, float*, float*)* @tester_f, float %x, float* %s, float* %ds, float* %c, float* %dc)
  
  ret float %0
}

declare double @__enzyme_autodiff(...)

; CHECK-LABEL: define internal { float } @diffetester_f(float %x, float* %s_out, float* %"s_out'", float* %c_out, float* %"c_out'")
; CHECK: call void @__nv_sincosf(float %x, float* %s_out, float* %c_out)
; CHECK: %[[dsf:.+]] = load float, float* %"s_out'"
; CHECK: %[[dcf:.+]] = load float, float* %"c_out'"
; CHECK: store float 0.000000e+00, float* %"s_out'"
; CHECK: store float 0.000000e+00, float* %"c_out'"
; CHECK: %[[sinf:.+]] = call fast float @llvm.sin.f32(float %x)
; CHECK: %[[cosf:.+]] = call fast float @llvm.cos.f32(float %x)
; CHECK: %[[term1f:.+]] = fmul fast float %[[dsf]], %[[cosf]]
; CHECK: %[[term2f:.+]] = fmul fast float %[[dcf]], %[[sinf]]
; CHECK: %[[difff:.+]] = fsub fast float %[[term1f]], %[[term2f]]
; CHECK: ret { float }

; CHECK-LABEL: define internal { double } @diffetester(double %x, double* %s_out, double* %"s_out'", double* %c_out, double* %"c_out'")
; CHECK: call void @__nv_sincos(double %x, double* %s_out, double* %c_out)
; CHECK: %[[ds:.+]] = load double, double* %"s_out'"
; CHECK: %[[dc:.+]] = load double, double* %"c_out'"
; CHECK: store double 0.000000e+00, double* %"s_out'"
; CHECK: store double 0.000000e+00, double* %"c_out'"
; CHECK: %[[sin:.+]] = call fast double @llvm.sin.f64(double %x)
; CHECK: %[[cos:.+]] = call fast double @llvm.cos.f64(double %x)
; CHECK: %[[term1:.+]] = fmul fast double %[[ds]], %[[cos]]
; CHECK: %[[term2:.+]] = fmul fast double %[[dc]], %[[sin]]
; CHECK: %[[diff:.+]] = fsub fast double %[[term1]], %[[term2]]
; CHECK: ret { double } 
