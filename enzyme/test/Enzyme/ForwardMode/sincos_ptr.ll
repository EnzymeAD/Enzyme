; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -sroa -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,sroa,instsimplify,%simplifycfg)" -S | FileCheck %s

declare float @air.sincos.f32(float, float*)
declare void @__nv_sincosf(float, float*, float*)
declare void @__nv_sincospif(float, float*, float*)
declare void @sincos(double, double*, double*)

declare float @__enzyme_fwddifff(i8*, ...)
declare double @__enzyme_fwddiff(i8*, ...)
declare [2 x double] @__enzyme_fwddiff2(i8*, ...)

define float @test_air(float %x) {
entry:
  %cosp = alloca float, align 4
  %s = call fast float @air.sincos.f32(float %x, float* %cosp)
  %c = load float, float* %cosp, align 4
  %res = fadd fast float %s, %c
  ret float %res
}

define float @dtest_air(float %x, float %dx) {
entry:
  %res = call float (i8*, ...) @__enzyme_fwddifff(i8* bitcast (float (float)* @test_air to i8*), float %x, float %dx)
  ret float %res
}

define float @test_nv(float %x) {
entry:
  %sinp = alloca float, align 4
  %cosp = alloca float, align 4
  call void @__nv_sincosf(float %x, float* %sinp, float* %cosp)
  %s = load float, float* %sinp, align 4
  %c = load float, float* %cosp, align 4
  %res = fadd fast float %s, %c
  ret float %res
}

define float @dtest_nv(float %x, float %dx) {
entry:
  %res = call float (i8*, ...) @__enzyme_fwddifff(i8* bitcast (float (float)* @test_nv to i8*), float %x, float %dx)
  ret float %res
}

define double @test_c_sincos(double %x) {
entry:
  %sinp = alloca double, align 8
  %cosp = alloca double, align 8
  call void @sincos(double %x, double* %sinp, double* %cosp)
  %s = load double, double* %sinp, align 8
  %c = load double, double* %cosp, align 8
  %res = fadd fast double %s, %c
  ret double %res
}

define [2 x double] @dtest_c_sincos_vec(double %x, double %dx1, double %dx2) {
entry:
  %res = call [2 x double] (i8*, ...) @__enzyme_fwddiff2(i8* bitcast (double (double)* @test_c_sincos to i8*), metadata !"enzyme_width", i64 2, double %x, double %dx1, double %dx2)
  ret [2 x double] %res
}

; CHECK: define internal float @fwddiffetest_air(float %x, float %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[SIN1:.+]] = call fast float @llvm.sin.f32(float %x)
; CHECK-NEXT:   %[[COS1:.+]] = call fast float @llvm.cos.f32(float %x)
; CHECK-NEXT:   %[[DS1:.+]] = fmul fast float %[[COS1]], %"x'"
; CHECK-NEXT:   %[[DC10:.+]] = fmul fast float %[[SIN1]], %"x'"
; CHECK-NEXT:   %[[DC1:.+]] = {{(fsub fast float -0.000000e\+00,|fneg fast float)}} %[[DC10]]
; CHECK-NEXT:   %[[SUM1:.+]] = fadd fast float %[[DS1]], %[[DC1]]
; CHECK-NEXT:   ret float %[[SUM1]]
; CHECK-NEXT: }

; CHECK: define internal float @fwddiffetest_nv(float %x, float %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[SIN2:.+]] = call fast float @llvm.sin.f32(float %x)
; CHECK-NEXT:   %[[COS2:.+]] = call fast float @llvm.cos.f32(float %x)
; CHECK-NEXT:   %[[DS2:.+]] = fmul fast float %[[COS2]], %"x'"
; CHECK-NEXT:   %[[DC20:.+]] = fmul fast float %[[SIN2]], %"x'"
; CHECK-NEXT:   %[[DC2:.+]] = {{(fsub fast float -0.000000e\+00,|fneg fast float)}} %[[DC20]]
; CHECK-NEXT:   %[[SUM2:.+]] = fadd fast float %[[DS2]], %[[DC2]]
; CHECK-NEXT:   ret float %[[SUM2]]
; CHECK-NEXT: }

; CHECK: define internal [2 x double] @fwddiffe2test_c_sincos(double %x, [2 x double] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[SIN3:.+]] = call fast double @llvm.sin.f64(double %x)
; CHECK-NEXT:   %[[COS3:.+]] = call fast double @llvm.cos.f64(double %x)
; CHECK-NEXT:   %[[DX0:.+]] = extractvalue [2 x double] %"x'", 0
; CHECK-NEXT:   %[[DS30:.+]] = fmul fast double %[[COS3]], %[[DX0]]
; CHECK-NEXT:   %[[DC300:.+]] = fmul fast double %[[SIN3]], %[[DX0]]
; CHECK-NEXT:   %[[DC30:.+]] = {{(fsub fast double -0.000000e\+00,|fneg fast double)}} %[[DC300]]
; CHECK-NEXT:   %[[DX1:.+]] = extractvalue [2 x double] %"x'", 1
; CHECK-NEXT:   %[[DS31:.+]] = fmul fast double %[[COS3]], %[[DX1]]
; CHECK-NEXT:   %[[DC310:.+]] = fmul fast double %[[SIN3]], %[[DX1]]
; CHECK-NEXT:   %[[DC31:.+]] = {{(fsub fast double -0.000000e\+00,|fneg fast double)}} %[[DC310]]
; CHECK-NEXT:   %[[SUM30:.+]] = fadd fast double %[[DS30]], %[[DC30]]
; CHECK-NEXT:   %[[INS0:.+]] = insertvalue [2 x double] undef, double %[[SUM30]], 0
; CHECK-NEXT:   %[[SUM31:.+]] = fadd fast double %[[DS31]], %[[DC31]]
; CHECK-NEXT:   %[[INS1:.+]] = insertvalue [2 x double] %[[INS0]], double %[[SUM31]], 1
; CHECK-NEXT:   ret [2 x double] %[[INS1]]
; CHECK-NEXT: }
