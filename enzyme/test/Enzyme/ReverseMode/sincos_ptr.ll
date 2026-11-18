; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -sroa -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,sroa,instsimplify,%simplifycfg)" -S | FileCheck %s

declare float @air.sincos.f32(float, float*)
declare float @__ocml_sincos_f32(float, float*)
declare void @__nv_sincosf(float, float*, float*)
declare void @__nv_sincospif(float, float*, float*)
declare void @sincos(double, double*, double*)

declare float @__enzyme_autodifff(i8*, ...)
declare double @__enzyme_autodiff(i8*, ...)
declare { i8*, float } @__enzyme_augmentfwd(i8*, ...)
declare { float } @__enzyme_reverse(i8*, ...)

define float @test_air(float %x) {
entry:
  %cosp = alloca float, align 4
  %s = call fast float @air.sincos.f32(float %x, float* %cosp)
  %c = load float, float* %cosp, align 4
  %res = fadd fast float %s, %c
  ret float %res
}

define float @dtest_air(float %x) {
entry:
  %res = call float (i8*, ...) @__enzyme_autodifff(i8* bitcast (float (float)* @test_air to i8*), float %x)
  ret float %res
}

define float @dtest_air_split(float %x) {
entry:
  %aug = call { i8*, float } (i8*, ...) @__enzyme_augmentfwd(i8* bitcast (float (float)* @test_air to i8*), metadata !"enzyme_allocated", i64 0, float %x)
  %tape = extractvalue { i8*, float } %aug, 0
  %rev = call { float } (i8*, ...) @__enzyme_reverse(i8* bitcast (float (float)* @test_air to i8*), metadata !"enzyme_allocated", i64 0, float %x, float 1.000000e+00, i8* %tape)
  %dx = extractvalue { float } %rev, 0
  ret float %dx
}

define float @test_ocml(float %x) {
entry:
  %cosp = alloca float, align 4
  %s = call fast float @__ocml_sincos_f32(float %x, float* %cosp)
  %c = load float, float* %cosp, align 4
  %res = fsub fast float %s, %c
  ret float %res
}

define float @dtest_ocml(float %x) {
entry:
  %res = call float (i8*, ...) @__enzyme_autodifff(i8* bitcast (float (float)* @test_ocml to i8*), float %x)
  ret float %res
}

define float @test_nv(float %x) {
entry:
  %sinp = alloca float, align 4
  %cosp = alloca float, align 4
  call void @__nv_sincosf(float %x, float* %sinp, float* %cosp)
  %s = load float, float* %sinp, align 4
  %c = load float, float* %cosp, align 4
  %s2 = fmul fast float %s, 2.000000e+00
  %c3 = fmul fast float %c, 3.000000e+00
  %res = fadd fast float %s2, %c3
  ret float %res
}

define float @dtest_nv(float %x) {
entry:
  %res = call float (i8*, ...) @__enzyme_autodifff(i8* bitcast (float (float)* @test_nv to i8*), float %x)
  ret float %res
}

define float @dtest_nv_split(float %x) {
entry:
  %aug = call { i8*, float } (i8*, ...) @__enzyme_augmentfwd(i8* bitcast (float (float)* @test_nv to i8*), metadata !"enzyme_allocated", i64 0, float %x)
  %tape = extractvalue { i8*, float } %aug, 0
  %rev = call { float } (i8*, ...) @__enzyme_reverse(i8* bitcast (float (float)* @test_nv to i8*), metadata !"enzyme_allocated", i64 0, float %x, float 1.000000e+00, i8* %tape)
  %dx = extractvalue { float } %rev, 0
  ret float %dx
}

define float @test_nv_pi(float %x) {
entry:
  %sinp = alloca float, align 4
  %cosp = alloca float, align 4
  call void @__nv_sincospif(float %x, float* %sinp, float* %cosp)
  %s = load float, float* %sinp, align 4
  %c = load float, float* %cosp, align 4
  %res = fadd fast float %s, %c
  ret float %res
}

define float @dtest_nv_pi(float %x) {
entry:
  %res = call float (i8*, ...) @__enzyme_autodifff(i8* bitcast (float (float)* @test_nv_pi to i8*), float %x)
  ret float %res
}

define double @test_c_sincos(double %x) {
entry:
  %sinp = alloca double, align 8
  %cosp = alloca double, align 8
  call void @sincos(double %x, double* %sinp, double* %cosp)
  %s = load double, double* %sinp, align 8
  %c = load double, double* %cosp, align 8
  %res = fmul fast double %s, %c
  ret double %res
}

define double @dtest_c_sincos(double %x) {
entry:
  %res = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (double)* @test_c_sincos to i8*), double %x)
  ret double %res
}

; CHECK: define internal { float } @diffetest_air(float %x, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[SIN1:.+]] = call fast float @llvm.sin.f32(float %x)
; CHECK-NEXT:   %[[COS1:.+]] = call fast float @llvm.cos.f32(float %x)
; CHECK-NEXT:   %[[DS:.+]] = fmul fast float %differeturn, %[[COS1]]
; CHECK-NEXT:   %[[DC0:.+]] = fmul fast float %differeturn, %[[SIN1]]
; CHECK-NEXT:   %[[DC:.+]] = {{(fsub fast float -0.000000e\+00,|fneg fast float)}} %[[DC0]]
; CHECK-NEXT:   %[[SUM:.+]] = fadd fast float %[[DS]], %[[DC]]
; CHECK-NEXT:   %[[RET:.+]] = insertvalue { float } undef, float %[[SUM]], 0
; CHECK-NEXT:   ret { float } %[[RET]]
; CHECK-NEXT: }

; CHECK: define internal float @augmented_test_air(float %x)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %cosp = alloca float, i64 1, align 4
; CHECK-NEXT:   %s = call fast float @air.sincos.f32(float %x, {{(float\*|ptr)}} %cosp)
; CHECK-NEXT:   %c = load float, {{(float\*, )?}}{{(float\*|ptr)}} %cosp, align 4
; CHECK-NEXT:   %res = fadd fast float %s, %c
; CHECK-NEXT:   ret float %res
; CHECK-NEXT: }

; CHECK: define internal { float } @diffetest_air.{{[0-9]+}}(float %x, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[SINS:.+]] = call fast float @llvm.sin.f32(float %x)
; CHECK-NEXT:   %[[COSS:.+]] = call fast float @llvm.cos.f32(float %x)
; CHECK-NEXT:   %[[DSS:.+]] = fmul fast float %differeturn, %[[COSS]]
; CHECK-NEXT:   %[[DCS0:.+]] = fmul fast float %differeturn, %[[SINS]]
; CHECK-NEXT:   %[[DCS:.+]] = {{(fsub fast float -0.000000e\+00,|fneg fast float)}} %[[DCS0]]
; CHECK-NEXT:   %[[SUMS:.+]] = fadd fast float %[[DSS]], %[[DCS]]
; CHECK-NEXT:   %[[RETS:.+]] = insertvalue { float } undef, float %[[SUMS]], 0
; CHECK-NEXT:   ret { float } %[[RETS]]
; CHECK-NEXT: }

; CHECK: define internal { float } @diffetest_ocml(float %x, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[NEG:.+]] = {{(fsub fast float -0.000000e\+00,|fneg fast float)}} %differeturn
; CHECK-NEXT:   %[[SIN2:.+]] = call fast float @llvm.sin.f32(float %x)
; CHECK-NEXT:   %[[COS2:.+]] = call fast float @llvm.cos.f32(float %x)
; CHECK-NEXT:   %[[DS2:.+]] = fmul fast float %differeturn, %[[COS2]]
; CHECK-NEXT:   %[[DC20:.+]] = fmul fast float %[[NEG]], %[[SIN2]]
; CHECK-NEXT:   %[[DC2:.+]] = {{(fsub fast float -0.000000e\+00,|fneg fast float)}} %[[DC20]]
; CHECK-NEXT:   %[[SUM2:.+]] = fadd fast float %[[DS2]], %[[DC2]]
; CHECK-NEXT:   %[[RET2:.+]] = insertvalue { float } undef, float %[[SUM2]], 0
; CHECK-NEXT:   ret { float } %[[RET2]]
; CHECK-NEXT: }

; CHECK: define internal { float } @diffetest_nv(float %x, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[M3:.+]] = fmul fast float %differeturn, 3.000000e+00
; CHECK-NEXT:   %[[M2:.+]] = fmul fast float %differeturn, 2.000000e+00
; CHECK-NEXT:   %[[SIN3:.+]] = call fast float @llvm.sin.f32(float %x)
; CHECK-NEXT:   %[[COS3:.+]] = call fast float @llvm.cos.f32(float %x)
; CHECK-NEXT:   %[[DS3:.+]] = fmul fast float %[[M2]], %[[COS3]]
; CHECK-NEXT:   %[[DC30:.+]] = fmul fast float %[[M3]], %[[SIN3]]
; CHECK-NEXT:   %[[DC3:.+]] = {{(fsub fast float -0.000000e\+00,|fneg fast float)}} %[[DC30]]
; CHECK-NEXT:   %[[SUM3:.+]] = fadd fast float %[[DS3]], %[[DC3]]
; CHECK-NEXT:   %[[RET3:.+]] = insertvalue { float } undef, float %[[SUM3]], 0
; CHECK-NEXT:   ret { float } %[[RET3]]
; CHECK-NEXT: }

; CHECK: define internal { float } @diffetest_nv_pi(float %x, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[PIX:.+]] = fmul fast float %x, 0x400921FB60000000
; CHECK-NEXT:   %[[SINP:.+]] = call fast float @llvm.sin.f32(float %[[PIX]])
; CHECK-NEXT:   %[[COSP:.+]] = call fast float @llvm.cos.f32(float %[[PIX]])
; CHECK-NEXT:   %[[SINPI:.+]] = fmul fast float %[[SINP]], 0x400921FB60000000
; CHECK-NEXT:   %[[COSPI:.+]] = fmul fast float %[[COSP]], 0x400921FB60000000
; CHECK-NEXT:   %[[DSP:.+]] = fmul fast float %differeturn, %[[COSPI]]
; CHECK-NEXT:   %[[DCP0:.+]] = fmul fast float %differeturn, %[[SINPI]]
; CHECK-NEXT:   %[[DCP:.+]] = {{(fsub fast float -0.000000e\+00,|fneg fast float)}} %[[DCP0]]
; CHECK-NEXT:   %[[SUMP:.+]] = fadd fast float %[[DSP]], %[[DCP]]
; CHECK-NEXT:   %[[RETP:.+]] = insertvalue { float } undef, float %[[SUMP]], 0
; CHECK-NEXT:   ret { float } %[[RETP]]
; CHECK-NEXT: }

; CHECK: define internal { double } @diffetest_c_sincos(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %sinp = alloca double, align 8
; CHECK-NEXT:   %cosp = alloca double, align 8
; CHECK-NEXT:   call void @sincos(double %x, {{(double\*|ptr)}} %sinp, {{(double\*|ptr)}} %cosp)
; CHECK-NEXT:   %s = load double, {{(double\*|ptr)}} %sinp, align 8
; CHECK-NEXT:   %c = load double, {{(double\*|ptr)}} %cosp, align 8
; CHECK-NEXT:   %[[DSIN:.+]] = fmul fast double %differeturn, %c
; CHECK-NEXT:   %[[DCOS:.+]] = fmul fast double %differeturn, %s
; CHECK-NEXT:   %[[SINC:.+]] = call fast double @llvm.sin.f64(double %x)
; CHECK-NEXT:   %[[COSC:.+]] = call fast double @llvm.cos.f64(double %x)
; CHECK-NEXT:   %[[DSC:.+]] = fmul fast double %[[DSIN]], %[[COSC]]
; CHECK-NEXT:   %[[DCC0:.+]] = fmul fast double %[[DCOS]], %[[SINC]]
; CHECK-NEXT:   %[[DCC:.+]] = {{(fsub fast double -0.000000e\+00,|fneg fast double)}} %[[DCC0]]
; CHECK-NEXT:   %[[SUMC:.+]] = fadd fast double %[[DSC]], %[[DCC]]
; CHECK-NEXT:   %[[RETC:.+]] = insertvalue { double } undef, double %[[SUMC]], 0
; CHECK-NEXT:   ret { double } %[[RETC]]
; CHECK-NEXT: }
