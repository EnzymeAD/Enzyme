; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -simplifycfg -instsimplify -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,%simplifycfg,instsimplify,adce)" -S | FileCheck %s

declare { double, double } @cexp(double, double)
declare { double, double } @csin(double, double)
declare { double, double } @ccos(double, double)
declare { double, double } @ctan(double, double)
declare { double, double } @casin(double, double)
declare { double, double } @cacos(double, double)
declare { double, double } @catan(double, double)
declare { double, double } @csinh(double, double)
declare { double, double } @ccosh(double, double)
declare { double, double } @ctanh(double, double)
declare { double, double } @casinh(double, double)
declare { double, double } @cacosh(double, double)
declare { double, double } @catanh(double, double)
declare { double, double } @clog(double, double)
declare { double, double } @clog10(double, double)
declare { double, double } @csqrt(double, double)
declare { double, double } @conj(double, double)
declare { double, double } @cproj(double, double)
declare { double, double } @cpow(double, double, double, double)
declare double @carg(double, double)
declare double @creal(double, double)
declare double @cimag(double, double)

declare <2 x float> @csinf(<2 x float>)
declare <2 x float> @cpowf(<2 x float>, <2 x float>)
declare float @cabsf(<2 x float>)

define { double, double } @test_csin(double %x, double %y) {
entry:
  %r = call { double, double } @csin(double %x, double %y)
  ret { double, double } %r
}

define { double, double } @test_clog(double %x, double %y) {
entry:
  %r = call { double, double } @clog(double %x, double %y)
  ret { double, double } %r
}

define { double, double } @test_cacosh(double %x, double %y) {
entry:
  %r = call { double, double } @cacosh(double %x, double %y)
  ret { double, double } %r
}

define { double, double } @test_cpow(double %ax, double %ay, double %bx, double %by) {
entry:
  %r = call { double, double } @cpow(double %ax, double %ay, double %bx, double %by)
  ret { double, double } %r
}

define double @test_carg(double %x, double %y) {
entry:
  %r = call double @carg(double %x, double %y)
  ret double %r
}

define double @test_all_scalar(double %x, double %y) {
entry:
  %z0 = call { double, double } @cexp(double %x, double %y)
  %z1 = call { double, double } @ccos(double %x, double %y)
  %z2 = call { double, double } @ctan(double %x, double %y)
  %z3 = call { double, double } @casin(double %x, double %y)
  %z4 = call { double, double } @cacos(double %x, double %y)
  %z5 = call { double, double } @catan(double %x, double %y)
  %z6 = call { double, double } @csinh(double %x, double %y)
  %z7 = call { double, double } @ccosh(double %x, double %y)
  %z8 = call { double, double } @ctanh(double %x, double %y)
  %z9 = call { double, double } @casinh(double %x, double %y)
  %z10 = call { double, double } @catanh(double %x, double %y)
  %z11 = call { double, double } @clog10(double %x, double %y)
  %z12 = call { double, double } @csqrt(double %x, double %y)
  %z13 = call { double, double } @conj(double %x, double %y)
  %z14 = call { double, double } @cproj(double %x, double %y)
  %r0 = call double @creal(double %x, double %y)
  %r1 = call double @cimag(double %x, double %y)
  %e0 = extractvalue { double, double } %z0, 0
  %e1 = extractvalue { double, double } %z1, 0
  %e2 = extractvalue { double, double } %z2, 0
  %e3 = extractvalue { double, double } %z3, 0
  %e4 = extractvalue { double, double } %z4, 0
  %e5 = extractvalue { double, double } %z5, 0
  %e6 = extractvalue { double, double } %z6, 0
  %e7 = extractvalue { double, double } %z7, 0
  %e8 = extractvalue { double, double } %z8, 0
  %e9 = extractvalue { double, double } %z9, 0
  %e10 = extractvalue { double, double } %z10, 0
  %e11 = extractvalue { double, double } %z11, 0
  %e12 = extractvalue { double, double } %z12, 0
  %e13 = extractvalue { double, double } %z13, 0
  %e14 = extractvalue { double, double } %z14, 0
  %s0 = fadd double %e0, %e1
  %s1 = fadd double %s0, %e2
  %s2 = fadd double %s1, %e3
  %s3 = fadd double %s2, %e4
  %s4 = fadd double %s3, %e5
  %s5 = fadd double %s4, %e6
  %s6 = fadd double %s5, %e7
  %s7 = fadd double %s6, %e8
  %s8 = fadd double %s7, %e9
  %s9 = fadd double %s8, %e10
  %s10 = fadd double %s9, %e11
  %s11 = fadd double %s10, %e12
  %s12 = fadd double %s11, %e13
  %s13 = fadd double %s12, %e14
  %s14 = fadd double %s13, %r0
  %s15 = fadd double %s14, %r1
  ret double %s15
}

define float @test_vec(<2 x float> %a, <2 x float> %b) {
entry:
  %s = call <2 x float> @csinf(<2 x float> %a)
  %p = call <2 x float> @cpowf(<2 x float> %s, <2 x float> %b)
  %m = call float @cabsf(<2 x float> %p)
  ret float %m
}

declare { double, double } @__enzyme_autodiff2(...)
declare { double, double, double, double } @__enzyme_autodiff4(...)
declare { <2 x float>, <2 x float> } @__enzyme_autodiff_v2(...)

define { double, double } @d_csin(double %x, double %y) {
  %r = call { double, double } (...) @__enzyme_autodiff2({ double, double } (double, double)* nonnull @test_csin, double %x, double %y)
  ret { double, double } %r
}

define { double, double } @d_clog(double %x, double %y) {
  %r = call { double, double } (...) @__enzyme_autodiff2({ double, double } (double, double)* nonnull @test_clog, double %x, double %y)
  ret { double, double } %r
}

define { double, double } @d_cacosh(double %x, double %y) {
  %r = call { double, double } (...) @__enzyme_autodiff2({ double, double } (double, double)* nonnull @test_cacosh, double %x, double %y)
  ret { double, double } %r
}

define { double, double, double, double } @d_cpow(double %ax, double %ay, double %bx, double %by) {
  %r = call { double, double, double, double } (...) @__enzyme_autodiff4({ double, double } (double, double, double, double)* nonnull @test_cpow, double %ax, double %ay, double %bx, double %by)
  ret { double, double, double, double } %r
}

define { double, double } @d_carg(double %x, double %y) {
  %r = call { double, double } (...) @__enzyme_autodiff2(double (double, double)* nonnull @test_carg, double %x, double %y)
  ret { double, double } %r
}

define { double, double } @d_all_scalar(double %x, double %y) {
  %r = call { double, double } (...) @__enzyme_autodiff2(double (double, double)* nonnull @test_all_scalar, double %x, double %y)
  ret { double, double } %r
}

define { <2 x float>, <2 x float> } @d_vec(<2 x float> %a, <2 x float> %b) {
  %r = call { <2 x float>, <2 x float> } (...) @__enzyme_autodiff_v2(float (<2 x float>, <2 x float>)* nonnull @test_vec, <2 x float> %a, <2 x float> %b)
  ret { <2 x float>, <2 x float> } %r
}

; CHECK: define internal { double, double } @diffetest_csin(double %x, double %y, { double, double } %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[COS:.+]] = call { double, double } @ccos(double %x, double %y)
; CHECK-NEXT:   %[[CRE:.+]] = extractvalue { double, double } %[[COS]], 0
; CHECK-NEXT:   %[[CIM:.+]] = extractvalue { double, double } %[[COS]], 1
; CHECK-NEXT:   %[[DR0:.+]] = extractvalue { double, double } %differeturn, 0
; CHECK-NEXT:   %[[DR1:.+]] = extractvalue { double, double } %differeturn, 1
; CHECK-NEXT:   %[[T0:.+]] = fmul fast double %[[CRE]], %[[DR0]]
; CHECK-NEXT:   %[[T1:.+]] = fmul fast double %[[CIM]], %[[DR1]]
; CHECK-NEXT:   %[[DX:.+]] = fadd fast double %[[T0]], %[[T1]]
; CHECK-NEXT:   %[[T2:.+]] = fmul fast double %[[CRE]], %[[DR1]]
; CHECK-NEXT:   %[[T3:.+]] = fmul fast double %[[CIM]], %[[DR0]]
; CHECK-NEXT:   %[[DY:.+]] = fsub fast double %[[T2]], %[[T3]]
; CHECK-NEXT:   %[[I0:.+]] = insertvalue { double, double } undef, double %[[DX]], 0
; CHECK-NEXT:   %[[I1:.+]] = insertvalue { double, double } %[[I0]], double %[[DY]], 1
; CHECK-NEXT:   ret { double, double } %[[I1]]

; CHECK: define internal { double, double } @diffetest_clog(double %x, double %y, { double, double } %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[XX:.+]] = fmul fast double %x, %x
; CHECK-NEXT:   %[[YY:.+]] = fmul fast double %y, %y
; CHECK-NEXT:   %[[N2:.+]] = fadd fast double %[[XX]], %[[YY]]
; CHECK-NEXT:   %[[RRE:.+]] = fdiv fast double %x, %[[N2]]
; CHECK-NEXT:   %[[NY:.+]] = fneg fast double %y
; CHECK-NEXT:   %[[RIM:.+]] = fdiv fast double %[[NY]], %[[N2]]

; CHECK: define internal { double, double } @diffetest_cacosh(double %x, double %y, { double, double } %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[XM1:.+]] = fsub fast double %x, 1.000000e+00
; CHECK-NEXT:   %[[SM1:.+]] = call { double, double } @csqrt(double %[[XM1]], double %y)
; CHECK:        %[[XP1:.+]] = fadd fast double %x, 1.000000e+00
; CHECK-NEXT:   %[[SP1:.+]] = call { double, double } @csqrt(double %[[XP1]], double %y)

; CHECK: define internal { double, double, double, double } @diffetest_cpow(double %ax, double %ay, double %bx, double %by, { double, double } %differeturn)
; CHECK: call { double, double } @cpow(double %ax, double %ay,
; CHECK: call { double, double } @cpow(double %ax, double %ay, double %bx, double %by)
; CHECK: call { double, double } @clog(double %ax, double %ay)

; CHECK: define internal { double, double } @diffetest_carg(double %x, double %y, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[AXX:.+]] = fmul fast double %x, %x
; CHECK-NEXT:   %[[AYY:.+]] = fmul fast double %y, %y
; CHECK-NEXT:   %[[AN2:.+]] = fadd fast double %[[AXX]], %[[AYY]]
; CHECK-NEXT:   %[[ANY:.+]] = fneg fast double %y
; CHECK-NEXT:   %[[ADX:.+]] = fdiv fast double %[[ANY]], %[[AN2]]
; CHECK-NEXT:   %[[ADY:.+]] = fdiv fast double %x, %[[AN2]]
; CHECK-NEXT:   %[[M0:.+]] = fmul fast double %differeturn, %[[ADX]]
; CHECK-NEXT:   %[[M1:.+]] = fmul fast double %differeturn, %[[ADY]]
; CHECK-NEXT:   %[[AI0:.+]] = insertvalue { double, double } undef, double %[[M0]], 0
; CHECK-NEXT:   %[[AI1:.+]] = insertvalue { double, double } %[[AI0]], double %[[M1]], 1
; CHECK-NEXT:   ret { double, double } %[[AI1]]

; CHECK: define internal { double, double } @diffetest_all_scalar(double %x, double %y, double %differeturn)
; CHECK: ret { double, double }

; CHECK: define internal { <2 x float>, <2 x float> } @diffetest_vec(<2 x float> %a, <2 x float> %b, float %differeturn)
; CHECK: call fast float @cabsf(<2 x float>
; CHECK: call fast <2 x float> @cpowf(<2 x float>
; CHECK: call fast <2 x float> @clogf(<2 x float>
; CHECK: call fast <2 x float> @ccosf(<2 x float>
; CHECK: ret { <2 x float>, <2 x float> }
