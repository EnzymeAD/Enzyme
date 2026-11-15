; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -simplifycfg -instsimplify -adce -S | FileCheck %s; fi
; RUN: if [ %llvmver -lt 21 ]; then %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,%simplifycfg,instsimplify,adce)" -S | FileCheck %s; fi

declare double @llvm.nvvm.fabs.d(double)
declare float @llvm.nvvm.fabs.f(float)
declare float @llvm.nvvm.fabs.ftz.f(float)

define double @test_fabs_d(double %x) {
entry:
  %r = call double @llvm.nvvm.fabs.d(double %x)
  ret double %r
}

define float @test_fabs_f(float %x) {
entry:
  %r = call float @llvm.nvvm.fabs.f(float %x)
  ret float %r
}

define float @test_fabs_ftz_f(float %x) {
entry:
  %r = call float @llvm.nvvm.fabs.ftz.f(float %x)
  ret float %r
}

declare double @__enzyme_autodiff(double (double)*, ...)
declare float @__enzyme_autodiff_f(float (float)*, ...)
declare double @__enzyme_fwddiff(double (double)*, ...)

define double @d_fabs_d(double %x) {
  %r = call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @test_fabs_d, double %x)
  ret double %r
}

define float @d_fabs_f(float %x) {
  %r = call float (float (float)*, ...) @__enzyme_autodiff_f(float (float)* nonnull @test_fabs_f, float %x)
  ret float %r
}

define float @d_fabs_ftz_f(float %x) {
  %r = call float (float (float)*, ...) @__enzyme_autodiff_f(float (float)* nonnull @test_fabs_ftz_f, float %x)
  ret float %r
}

define double @fd_fabs_d(double %x) {
  %r = call double (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @test_fabs_d, double %x, double 1.0)
  ret double %r
}

; CHECK: define internal { double } @diffetest_fabs_d(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[C0:.+]] = fcmp fast olt double %x, 0.000000e+00
; CHECK-NEXT:   %[[S0:.+]] = select fast i1 %[[C0]], double -1.000000e+00, double 1.000000e+00
; CHECK-NEXT:   %[[M0:.+]] = fmul fast double %differeturn, %[[S0]]
; CHECK-NEXT:   %[[R0:.+]] = insertvalue { double } undef, double %[[M0]], 0
; CHECK-NEXT:   ret { double } %[[R0]]

; CHECK: define internal { float } @diffetest_fabs_f(float %x, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[C1:.+]] = fcmp fast olt float %x, 0.000000e+00
; CHECK-NEXT:   %[[S1:.+]] = select fast i1 %[[C1]], float -1.000000e+00, float 1.000000e+00
; CHECK-NEXT:   %[[M1:.+]] = fmul fast float %differeturn, %[[S1]]
; CHECK-NEXT:   %[[R1:.+]] = insertvalue { float } undef, float %[[M1]], 0
; CHECK-NEXT:   ret { float } %[[R1]]

; CHECK: define internal { float } @diffetest_fabs_ftz_f(float %x, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[C2:.+]] = fcmp fast olt float %x, 0.000000e+00
; CHECK-NEXT:   %[[S2:.+]] = select fast i1 %[[C2]], float -1.000000e+00, float 1.000000e+00
; CHECK-NEXT:   %[[M2:.+]] = fmul fast float %differeturn, %[[S2]]
; CHECK-NEXT:   %[[R2:.+]] = insertvalue { float } undef, float %[[M2]], 0
; CHECK-NEXT:   ret { float } %[[R2]]

; CHECK: define internal double @fwddiffetest_fabs_d(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[C3:.+]] = fcmp fast olt double %x, 0.000000e+00
; CHECK-NEXT:   %[[S3:.+]] = select fast i1 %[[C3]], double -1.000000e+00, double 1.000000e+00
; CHECK-NEXT:   %[[M3:.+]] = fmul fast double %"x'", %[[S3]]
; CHECK-NEXT:   ret double %[[M3]]
