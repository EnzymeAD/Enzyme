; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -simplifycfg -instsimplify -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,%simplifycfg,instsimplify,adce)" -S | FileCheck %s

declare double @erfcx(double)
declare float @erfcxf(float)
declare double @erfcinv(double)
declare float @erfcinvf(float)
declare double @rcbrt(double)
declare float @rcbrtf(float)
declare double @rsqrt(double)
declare float @rsqrtf(float)
declare double @normcdf(double)
declare float @normcdff(float)
declare double @normcdfinv(double)
declare float @normcdfinvf(float)
declare double @llvm.nvvm.rsqrt.approx.d(double)
declare float @llvm.nvvm.rsqrt.approx.f(float)
declare float @llvm.nvvm.rsqrt.approx.ftz.f(float)

define double @test_erfcx(double %x) {
entry:
  %r = call double @erfcx(double %x)
  ret double %r
}

define float @test_erfcxf(float %x) {
entry:
  %r = call float @erfcxf(float %x)
  ret float %r
}

define double @test_erfcinv(double %x) {
entry:
  %r = call double @erfcinv(double %x)
  ret double %r
}

define float @test_erfcinvf(float %x) {
entry:
  %r = call float @erfcinvf(float %x)
  ret float %r
}

define double @test_rcbrt(double %x) {
entry:
  %r = call double @rcbrt(double %x)
  ret double %r
}

define float @test_rcbrtf(float %x) {
entry:
  %r = call float @rcbrtf(float %x)
  ret float %r
}

define double @test_rsqrt(double %x) {
entry:
  %r = call double @rsqrt(double %x)
  ret double %r
}

define float @test_rsqrtf(float %x) {
entry:
  %r = call float @rsqrtf(float %x)
  ret float %r
}

define double @test_normcdf(double %x) {
entry:
  %r = call double @normcdf(double %x)
  ret double %r
}

define float @test_normcdff(float %x) {
entry:
  %r = call float @normcdff(float %x)
  ret float %r
}

define double @test_normcdfinv(double %x) {
entry:
  %r = call double @normcdfinv(double %x)
  ret double %r
}

define float @test_normcdfinvf(float %x) {
entry:
  %r = call float @normcdfinvf(float %x)
  ret float %r
}

define double @test_nvvm_rsqrt_d(double %x) {
entry:
  %r = call double @llvm.nvvm.rsqrt.approx.d(double %x)
  ret double %r
}

define float @test_nvvm_rsqrt_f(float %x) {
entry:
  %r = call float @llvm.nvvm.rsqrt.approx.f(float %x)
  ret float %r
}

define float @test_nvvm_rsqrt_ftz_f(float %x) {
entry:
  %r = call float @llvm.nvvm.rsqrt.approx.ftz.f(float %x)
  ret float %r
}

declare double @__enzyme_autodiff(double (double)*, ...)
declare float @__enzyme_autodiff_f(float (float)*, ...)

define double @d_erfcx(double %x) {
  %r = call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @test_erfcx, double %x)
  ret double %r
}
define float @d_erfcxf(float %x) {
  %r = call float (float (float)*, ...) @__enzyme_autodiff_f(float (float)* nonnull @test_erfcxf, float %x)
  ret float %r
}
define double @d_erfcinv(double %x) {
  %r = call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @test_erfcinv, double %x)
  ret double %r
}
define float @d_erfcinvf(float %x) {
  %r = call float (float (float)*, ...) @__enzyme_autodiff_f(float (float)* nonnull @test_erfcinvf, float %x)
  ret float %r
}
define double @d_rcbrt(double %x) {
  %r = call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @test_rcbrt, double %x)
  ret double %r
}
define float @d_rcbrtf(float %x) {
  %r = call float (float (float)*, ...) @__enzyme_autodiff_f(float (float)* nonnull @test_rcbrtf, float %x)
  ret float %r
}
define double @d_rsqrt(double %x) {
  %r = call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @test_rsqrt, double %x)
  ret double %r
}
define float @d_rsqrtf(float %x) {
  %r = call float (float (float)*, ...) @__enzyme_autodiff_f(float (float)* nonnull @test_rsqrtf, float %x)
  ret float %r
}
define double @d_normcdf(double %x) {
  %r = call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @test_normcdf, double %x)
  ret double %r
}
define float @d_normcdff(float %x) {
  %r = call float (float (float)*, ...) @__enzyme_autodiff_f(float (float)* nonnull @test_normcdff, float %x)
  ret float %r
}
define double @d_normcdfinv(double %x) {
  %r = call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @test_normcdfinv, double %x)
  ret double %r
}
define float @d_normcdfinvf(float %x) {
  %r = call float (float (float)*, ...) @__enzyme_autodiff_f(float (float)* nonnull @test_normcdfinvf, float %x)
  ret float %r
}
define double @d_nvvm_rsqrt_d(double %x) {
  %r = call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @test_nvvm_rsqrt_d, double %x)
  ret double %r
}
define float @d_nvvm_rsqrt_f(float %x) {
  %r = call float (float (float)*, ...) @__enzyme_autodiff_f(float (float)* nonnull @test_nvvm_rsqrt_f, float %x)
  ret float %r
}
define float @d_nvvm_rsqrt_ftz_f(float %x) {
  %r = call float (float (float)*, ...) @__enzyme_autodiff_f(float (float)* nonnull @test_nvvm_rsqrt_ftz_f, float %x)
  ret float %r
}

; CHECK: define internal { double } @diffetest_erfcx(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[V0:.+]] = call fast double @erfcx(double %x)
; CHECK-NEXT:   %[[V1:.+]] = fmul fast double %x, %[[V0]]
; CHECK-NEXT:   %[[V2:.+]] = fsub fast double %[[V1]], 0x3FE20DD750429B6D
; CHECK-NEXT:   %[[V3:.+]] = fmul fast double 2.000000e+00, %[[V2]]
; CHECK-NEXT:   %[[V4:.+]] = fmul fast double %differeturn, %[[V3]]
; CHECK-NEXT:   %[[V5:.+]] = insertvalue { double } undef, double %[[V4]], 0
; CHECK-NEXT:   ret { double } %[[V5]]

; CHECK: define internal { float } @diffetest_erfcxf(float %x, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[F0:.+]] = call fast float @erfcxf(float %x)
; CHECK-NEXT:   %[[F1:.+]] = fmul fast float %x, %[[F0]]
; CHECK-NEXT:   %[[F2:.+]] = fsub fast float %[[F1]], 0x3FE20DD760000000
; CHECK-NEXT:   %[[F3:.+]] = fmul fast float 2.000000e+00, %[[F2]]
; CHECK-NEXT:   %[[F4:.+]] = fmul fast float %differeturn, %[[F3]]
; CHECK-NEXT:   %[[F5:.+]] = insertvalue { float } undef, float %[[F4]], 0
; CHECK-NEXT:   ret { float } %[[F5]]

; CHECK: define internal { double } @diffetest_erfcinv(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[EI0:.+]] = call fast double @erfcinv(double %x)
; CHECK-NEXT:   %[[EI1:.+]] = fmul fast double %[[EI0]], %[[EI0]]
; CHECK-NEXT:   %[[EI2:.+]] = call fast double @llvm.exp.f64(double %[[EI1]])
; CHECK-NEXT:   %[[EI3:.+]] = fmul fast double 0xBFEC5BF891B4EF6B, %[[EI2]]
; CHECK-NEXT:   %[[EI4:.+]] = fmul fast double %[[EI3]], %differeturn
; CHECK-NEXT:   %[[EI5:.+]] = insertvalue { double } undef, double %[[EI4]], 0
; CHECK-NEXT:   ret { double } %[[EI5]]

; CHECK: define internal { float } @diffetest_erfcinvf(float %x, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[EIF0:.+]] = call fast float @erfcinvf(float %x)
; CHECK-NEXT:   %[[EIF1:.+]] = fmul fast float %[[EIF0]], %[[EIF0]]
; CHECK-NEXT:   %[[EIF2:.+]] = call fast float @llvm.exp.f32(float %[[EIF1]])
; CHECK-NEXT:   %[[EIF3:.+]] = fmul fast float 0xBFEC5BF8A0000000, %[[EIF2]]
; CHECK-NEXT:   %[[EIF4:.+]] = fmul fast float %[[EIF3]], %differeturn
; CHECK-NEXT:   %[[EIF5:.+]] = insertvalue { float } undef, float %[[EIF4]], 0
; CHECK-NEXT:   ret { float } %[[EIF5]]

; CHECK: define internal { double } @diffetest_rcbrt(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[RC0:.+]] = call fast double @rcbrt(double %x)
; CHECK-NEXT:   %[[RC1:.+]] = fmul fast double %differeturn, %[[RC0]]
; CHECK-NEXT:   %[[RC2:.+]] = fmul fast double 3.000000e+00, %x
; CHECK-NEXT:   %[[RC3:.+]] = fdiv fast double %[[RC1]], %[[RC2]]
; CHECK-NEXT:   %[[RC4:.+]] = fneg fast double %[[RC3]]
; CHECK-NEXT:   %[[RC5:.+]] = insertvalue { double } undef, double %[[RC4]], 0
; CHECK-NEXT:   ret { double } %[[RC5]]

; CHECK: define internal { float } @diffetest_rcbrtf(float %x, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[RCF0:.+]] = call fast float @rcbrtf(float %x)
; CHECK-NEXT:   %[[RCF1:.+]] = fmul fast float %differeturn, %[[RCF0]]
; CHECK-NEXT:   %[[RCF2:.+]] = fmul fast float 3.000000e+00, %x
; CHECK-NEXT:   %[[RCF3:.+]] = fdiv fast float %[[RCF1]], %[[RCF2]]
; CHECK-NEXT:   %[[RCF4:.+]] = fneg fast float %[[RCF3]]
; CHECK-NEXT:   %[[RCF5:.+]] = insertvalue { float } undef, float %[[RCF4]], 0
; CHECK-NEXT:   ret { float } %[[RCF5]]

; CHECK: define internal { double } @diffetest_rsqrt(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[RS0:.+]] = call fast double @rsqrt(double %x)
; CHECK-NEXT:   %[[RS1:.+]] = fmul fast double %differeturn, %[[RS0]]
; CHECK-NEXT:   %[[RS2:.+]] = fmul fast double 2.000000e+00, %x
; CHECK-NEXT:   %[[RS3:.+]] = fdiv fast double %[[RS1]], %[[RS2]]
; CHECK-NEXT:   %[[RS4:.+]] = fneg fast double %[[RS3]]
; CHECK-NEXT:   %[[RS5:.+]] = insertvalue { double } undef, double %[[RS4]], 0
; CHECK-NEXT:   ret { double } %[[RS5]]

; CHECK: define internal { float } @diffetest_rsqrtf(float %x, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[RSF0:.+]] = call fast float @rsqrtf(float %x)
; CHECK-NEXT:   %[[RSF1:.+]] = fmul fast float %differeturn, %[[RSF0]]
; CHECK-NEXT:   %[[RSF2:.+]] = fmul fast float 2.000000e+00, %x
; CHECK-NEXT:   %[[RSF3:.+]] = fdiv fast float %[[RSF1]], %[[RSF2]]
; CHECK-NEXT:   %[[RSF4:.+]] = fneg fast float %[[RSF3]]
; CHECK-NEXT:   %[[RSF5:.+]] = insertvalue { float } undef, float %[[RSF4]], 0
; CHECK-NEXT:   ret { float } %[[RSF5]]

; CHECK: define internal { double } @diffetest_normcdf(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[NC0:.+]] = fmul fast double %x, %x
; CHECK-NEXT:   %[[NC1:.+]] = fmul fast double -5.000000e-01, %[[NC0]]
; CHECK-NEXT:   %[[NC2:.+]] = call fast double @llvm.exp.f64(double %[[NC1]])
; CHECK-NEXT:   %[[NC3:.+]] = fmul fast double 0x3FD9884533D43651, %[[NC2]]
; CHECK-NEXT:   %[[NC4:.+]] = fmul fast double %differeturn, %[[NC3]]
; CHECK-NEXT:   %[[NC5:.+]] = insertvalue { double } undef, double %[[NC4]], 0
; CHECK-NEXT:   ret { double } %[[NC5]]

; CHECK: define internal { float } @diffetest_normcdff(float %x, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[NCF0:.+]] = fmul fast float %x, %x
; CHECK-NEXT:   %[[NCF1:.+]] = fmul fast float -5.000000e-01, %[[NCF0]]
; CHECK-NEXT:   %[[NCF2:.+]] = call fast float @llvm.exp.f32(float %[[NCF1]])
; CHECK-NEXT:   %[[NCF3:.+]] = fmul fast float 0x3FD9884540000000, %[[NCF2]]
; CHECK-NEXT:   %[[NCF4:.+]] = fmul fast float %differeturn, %[[NCF3]]
; CHECK-NEXT:   %[[NCF5:.+]] = insertvalue { float } undef, float %[[NCF4]], 0
; CHECK-NEXT:   ret { float } %[[NCF5]]

; CHECK: define internal { double } @diffetest_normcdfinv(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[NCI0:.+]] = call fast double @normcdfinv(double %x)
; CHECK-NEXT:   %[[NCI1:.+]] = fmul fast double %[[NCI0]], %[[NCI0]]
; CHECK-NEXT:   %[[NCI2:.+]] = fmul fast double 5.000000e-01, %[[NCI1]]
; CHECK-NEXT:   %[[NCI3:.+]] = call fast double @llvm.exp.f64(double %[[NCI2]])
; CHECK-NEXT:   %[[NCI4:.+]] = fmul fast double 0x40040D931FF62706, %[[NCI3]]
; CHECK-NEXT:   %[[NCI5:.+]] = fmul fast double %differeturn, %[[NCI4]]
; CHECK-NEXT:   %[[NCI6:.+]] = insertvalue { double } undef, double %[[NCI5]], 0
; CHECK-NEXT:   ret { double } %[[NCI6]]

; CHECK: define internal { float } @diffetest_normcdfinvf(float %x, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[NCIF0:.+]] = call fast float @normcdfinvf(float %x)
; CHECK-NEXT:   %[[NCIF1:.+]] = fmul fast float %[[NCIF0]], %[[NCIF0]]
; CHECK-NEXT:   %[[NCIF2:.+]] = fmul fast float 5.000000e-01, %[[NCIF1]]
; CHECK-NEXT:   %[[NCIF3:.+]] = call fast float @llvm.exp.f32(float %[[NCIF2]])
; CHECK-NEXT:   %[[NCIF4:.+]] = fmul fast float 0x40040D9320000000, %[[NCIF3]]
; CHECK-NEXT:   %[[NCIF5:.+]] = fmul fast float %differeturn, %[[NCIF4]]
; CHECK-NEXT:   %[[NCIF6:.+]] = insertvalue { float } undef, float %[[NCIF5]], 0
; CHECK-NEXT:   ret { float } %[[NCIF6]]

; CHECK: define internal { double } @diffetest_nvvm_rsqrt_d(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[NRD0:.+]] = call fast double @llvm.nvvm.rsqrt.approx.d(double %x)
; CHECK-NEXT:   %[[NRD1:.+]] = fmul fast double %differeturn, %[[NRD0]]
; CHECK-NEXT:   %[[NRD2:.+]] = fmul fast double 2.000000e+00, %x
; CHECK-NEXT:   %[[NRD3:.+]] = fdiv fast double %[[NRD1]], %[[NRD2]]
; CHECK-NEXT:   %[[NRD4:.+]] = fneg fast double %[[NRD3]]
; CHECK-NEXT:   %[[NRD5:.+]] = insertvalue { double } undef, double %[[NRD4]], 0
; CHECK-NEXT:   ret { double } %[[NRD5]]

; CHECK: define internal { float } @diffetest_nvvm_rsqrt_f(float %x, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[NRF0:.+]] = call fast float @llvm.nvvm.rsqrt.approx.f(float %x)
; CHECK-NEXT:   %[[NRF1:.+]] = fmul fast float %differeturn, %[[NRF0]]
; CHECK-NEXT:   %[[NRF2:.+]] = fmul fast float 2.000000e+00, %x
; CHECK-NEXT:   %[[NRF3:.+]] = fdiv fast float %[[NRF1]], %[[NRF2]]
; CHECK-NEXT:   %[[NRF4:.+]] = fneg fast float %[[NRF3]]
; CHECK-NEXT:   %[[NRF5:.+]] = insertvalue { float } undef, float %[[NRF4]], 0
; CHECK-NEXT:   ret { float } %[[NRF5]]

; CHECK: define internal { float } @diffetest_nvvm_rsqrt_ftz_f(float %x, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[NRFTZ0:.+]] = call fast float @llvm.nvvm.rsqrt.approx.ftz.f(float %x)
; CHECK-NEXT:   %[[NRFTZ1:.+]] = fmul fast float %differeturn, %[[NRFTZ0]]
; CHECK-NEXT:   %[[NRFTZ2:.+]] = fmul fast float 2.000000e+00, %x
; CHECK-NEXT:   %[[NRFTZ3:.+]] = fdiv fast float %[[NRFTZ1]], %[[NRFTZ2]]
; CHECK-NEXT:   %[[NRFTZ4:.+]] = fneg fast float %[[NRFTZ3]]
; CHECK-NEXT:   %[[NRFTZ5:.+]] = insertvalue { float } undef, float %[[NRFTZ4]], 0
; CHECK-NEXT:   ret { float } %[[NRFTZ5]]
