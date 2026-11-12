; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -simplifycfg -instsimplify -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,%simplifycfg,instsimplify,adce)" -S | FileCheck %s

declare double @erfcx(double)
declare double @erfcinv(double)
declare double @rcbrt(double)
declare double @rsqrt(double)
declare double @normcdf(double)
declare double @normcdfinv(double)
declare double @llvm.nvvm.rsqrt.approx.d(double)

define double @test_erfcx(double %x) {
entry:
  %r = call double @erfcx(double %x)
  ret double %r
}

define double @test_erfcinv(double %x) {
entry:
  %r = call double @erfcinv(double %x)
  ret double %r
}

define double @test_rcbrt(double %x) {
entry:
  %r = call double @rcbrt(double %x)
  ret double %r
}

define double @test_rsqrt(double %x) {
entry:
  %r = call double @rsqrt(double %x)
  ret double %r
}

define double @test_normcdf(double %x) {
entry:
  %r = call double @normcdf(double %x)
  ret double %r
}

define double @test_normcdfinv(double %x) {
entry:
  %r = call double @normcdfinv(double %x)
  ret double %r
}

define double @test_nvvm_rsqrt_d(double %x) {
entry:
  %r = call double @llvm.nvvm.rsqrt.approx.d(double %x)
  ret double %r
}

declare double @__enzyme_fwddiff(double (double)*, ...)

define double @d_erfcx(double %x) {
  %r = call double (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @test_erfcx, double %x, double 1.0)
  ret double %r
}
define double @d_erfcinv(double %x) {
  %r = call double (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @test_erfcinv, double %x, double 1.0)
  ret double %r
}
define double @d_rcbrt(double %x) {
  %r = call double (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @test_rcbrt, double %x, double 1.0)
  ret double %r
}
define double @d_rsqrt(double %x) {
  %r = call double (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @test_rsqrt, double %x, double 1.0)
  ret double %r
}
define double @d_normcdf(double %x) {
  %r = call double (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @test_normcdf, double %x, double 1.0)
  ret double %r
}
define double @d_normcdfinv(double %x) {
  %r = call double (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @test_normcdfinv, double %x, double 1.0)
  ret double %r
}
define double @d_nvvm_rsqrt_d(double %x) {
  %r = call double (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @test_nvvm_rsqrt_d, double %x, double 1.0)
  ret double %r
}

; CHECK: define internal double @fwddiffetest_erfcx(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[V0:.+]] = call fast double @erfcx(double %x)
; CHECK-NEXT:   %[[V1:.+]] = fmul fast double %x, %[[V0]]
; CHECK-NEXT:   %[[V2:.+]] = fsub fast double %[[V1]], 0x3FE20DD750429B6D
; CHECK-NEXT:   %[[V3:.+]] = fmul fast double 2.000000e+00, %[[V2]]
; CHECK-NEXT:   %[[V4:.+]] = fmul fast double %"x'", %[[V3]]
; CHECK-NEXT:   ret double %[[V4]]

; CHECK: define internal double @fwddiffetest_erfcinv(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[EI0:.+]] = call fast double @erfcinv(double %x)
; CHECK-NEXT:   %[[EI1:.+]] = fmul fast double %[[EI0]], %[[EI0]]
; CHECK-NEXT:   %[[EI2:.+]] = call fast double @llvm.exp.f64(double %[[EI1]])
; CHECK-NEXT:   %[[EI3:.+]] = fmul fast double 0xBFEC5BF891B4EF6B, %[[EI2]]
; CHECK-NEXT:   %[[EI4:.+]] = fmul fast double %[[EI3]], %"x'"
; CHECK-NEXT:   ret double %[[EI4]]

; CHECK: define internal double @fwddiffetest_rcbrt(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[RC0:.+]] = call fast double @rcbrt(double %x)
; CHECK-NEXT:   %[[RC1:.+]] = fmul fast double %"x'", %[[RC0]]
; CHECK-NEXT:   %[[RC2:.+]] = fmul fast double 3.000000e+00, %x
; CHECK-NEXT:   %[[RC3:.+]] = fdiv fast double %[[RC1]], %[[RC2]]
; CHECK-NEXT:   %[[RC4:.+]] = fneg fast double %[[RC3]]
; CHECK-NEXT:   ret double %[[RC4]]

; CHECK: define internal double @fwddiffetest_rsqrt(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[RS0:.+]] = call fast double @rsqrt(double %x)
; CHECK-NEXT:   %[[RS1:.+]] = fmul fast double %"x'", %[[RS0]]
; CHECK-NEXT:   %[[RS2:.+]] = fmul fast double 2.000000e+00, %x
; CHECK-NEXT:   %[[RS3:.+]] = fdiv fast double %[[RS1]], %[[RS2]]
; CHECK-NEXT:   %[[RS4:.+]] = fneg fast double %[[RS3]]
; CHECK-NEXT:   ret double %[[RS4]]

; CHECK: define internal double @fwddiffetest_normcdf(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[NC0:.+]] = fmul fast double %x, %x
; CHECK-NEXT:   %[[NC1:.+]] = fmul fast double -5.000000e-01, %[[NC0]]
; CHECK-NEXT:   %[[NC2:.+]] = call fast double @llvm.exp.f64(double %[[NC1]])
; CHECK-NEXT:   %[[NC3:.+]] = fmul fast double 0x3FD9884533D43651, %[[NC2]]
; CHECK-NEXT:   %[[NC4:.+]] = fmul fast double %"x'", %[[NC3]]
; CHECK-NEXT:   ret double %[[NC4]]

; CHECK: define internal double @fwddiffetest_normcdfinv(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[NCI0:.+]] = call fast double @normcdfinv(double %x)
; CHECK-NEXT:   %[[NCI1:.+]] = fmul fast double %[[NCI0]], %[[NCI0]]
; CHECK-NEXT:   %[[NCI2:.+]] = fmul fast double 5.000000e-01, %[[NCI1]]
; CHECK-NEXT:   %[[NCI3:.+]] = call fast double @llvm.exp.f64(double %[[NCI2]])
; CHECK-NEXT:   %[[NCI4:.+]] = fmul fast double 0x40040D931FF62706, %[[NCI3]]
; CHECK-NEXT:   %[[NCI5:.+]] = fmul fast double %"x'", %[[NCI4]]
; CHECK-NEXT:   ret double %[[NCI5]]

; CHECK: define internal double @fwddiffetest_nvvm_rsqrt_d(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[NRD0:.+]] = call fast double @llvm.nvvm.rsqrt.approx.d(double %x)
; CHECK-NEXT:   %[[NRD1:.+]] = fmul fast double %"x'", %[[NRD0]]
; CHECK-NEXT:   %[[NRD2:.+]] = fmul fast double 2.000000e+00, %x
; CHECK-NEXT:   %[[NRD3:.+]] = fdiv fast double %[[NRD1]], %[[NRD2]]
; CHECK-NEXT:   %[[NRD4:.+]] = fneg fast double %[[NRD3]]
; CHECK-NEXT:   ret double %[[NRD4]]
