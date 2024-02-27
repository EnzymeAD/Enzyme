; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

declare double @__enzyme_autodiff(i8*, ...)
declare float @__enzyme_autodifff(i8*, ...)
declare x86_fp80 @__enzyme_autodiffl(i8*, ...)

; double
declare double @modf(double, double*)
define double @testint(double %x) {
entry:
  %integral_part = alloca double, align 8
  %fractional_part = call double @modf(double %x, double* %integral_part)
  %ret = load double, double* %integral_part, align 8
  ret double %ret
}
define double @testfrac(double %x) {
entry:
  %integral_part = alloca double, align 8
  %fractional_part = call double @modf(double %x, double* %integral_part)
  ret double %fractional_part
}

define double @dtestint(double %x, double %dx) {
entry:
  %call = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (double)* @testint to i8*), double %x)
  ret double %call
}
define double @dtestfrac(double %x, double %dx) {
entry:
  %call = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (double)* @testfrac to i8*), double %x)
  ret double %call
}

; float
declare float @modff(float, float*)
define float @testintf(float %x) {
entry:
  %integral_part = alloca float, align 4
  %fractional_part = call float @modff(float %x, float* %integral_part)
  %ret = load float, float* %integral_part, align 4
  ret float %ret
}
define float @testfracf(float %x) {
entry:
  %integral_part = alloca float, align 4
  %fractional_part = call float @modff(float %x, float* %integral_part)
  ret float %fractional_part
}

define float @dtestintf(float %x, float %dx) {
entry:
  %call = call float (i8*, ...) @__enzyme_autodifff(i8* bitcast (float (float)* @testintf to i8*), float %x)
  ret float %call
}
define float @dtestfracf(float %x, float %dx) {
entry:
  %call = call float (i8*, ...) @__enzyme_autodifff(i8* bitcast (float (float)* @testfracf to i8*), float %x)
  ret float %call
}

; x86_fp80
declare x86_fp80 @modfl(x86_fp80, x86_fp80*)
define x86_fp80 @testintl(x86_fp80 %x) {
entry:
  %integral_part = alloca x86_fp80, align 8
  %fractional_part = call x86_fp80 @modfl(x86_fp80 %x, x86_fp80* %integral_part)
  %ret = load x86_fp80, x86_fp80* %integral_part, align 8
  ret x86_fp80 %ret
}
define x86_fp80 @testfracl(x86_fp80 %x) {
entry:
  %integral_part = alloca x86_fp80, align 8
  %fractional_part = call x86_fp80 @modfl(x86_fp80 %x, x86_fp80* %integral_part)
  ret x86_fp80 %fractional_part
}

define x86_fp80 @dtestintl(x86_fp80 %x, x86_fp80 %dx) {
entry:
  %call = call x86_fp80 (i8*, ...) @__enzyme_autodiffl(i8* bitcast (x86_fp80 (x86_fp80)* @testintl to i8*), x86_fp80 %x)
  ret x86_fp80 %call
}
define x86_fp80 @dtestfracl(x86_fp80 %x, x86_fp80 %dx) {
entry:
  %call = call x86_fp80 (i8*, ...) @__enzyme_autodiffl(i8* bitcast (x86_fp80 (x86_fp80)* @testfracl to i8*), x86_fp80 %x)
  ret x86_fp80 %call
}

; double tests

; CHECK: define internal { double } @diffetestint(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"x'de" = alloca double, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"x'de", align 8
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:
; CHECK-NEXT:   %0 = load double, double* %"x'de", align 8
; CHECK-NEXT:   %1 = insertvalue { double } undef, double %0, 0
; CHECK-NEXT:   ret { double } %1
; CHECK-NEXT: }

; CHECK: define internal { double } @diffetestfrac(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"fractional_part'de" = alloca double, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"fractional_part'de", align 8
; CHECK-NEXT:   %"x'de" = alloca double, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"x'de", align 8
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:
; CHECK-NEXT:   store double %differeturn, double* %"fractional_part'de", align 8
; CHECK-NEXT:   %0 = load double, double* %"fractional_part'de", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"fractional_part'de", align 8
; CHECK-NEXT:   %1 = load double, double* %"x'de", align 8
; CHECK-NEXT:   %2 = fadd fast double %1, %0
; CHECK-NEXT:   store double %2, double* %"x'de", align 8
; CHECK-NEXT:   %3 = load double, double* %"x'de", align 8
; CHECK-NEXT:   %4 = insertvalue { double } undef, double %3, 0
; CHECK-NEXT:   ret { double } %4
; CHECK-NEXT: }

; float tests

; CHECK: define internal { float } @diffetestintf(float %x, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"x'de" = alloca float, align 4
; CHECK-NEXT:   store float 0.000000e+00, float* %"x'de", align 4
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:
; CHECK-NEXT:   %0 = load float, float* %"x'de", align 4
; CHECK-NEXT:   %1 = insertvalue { float } undef, float %0, 0
; CHECK-NEXT:   ret { float } %1
; CHECK-NEXT: }

; CHECK: define internal { float } @diffetestfracf(float %x, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"fractional_part'de" = alloca float, align 4
; CHECK-NEXT:   store float 0.000000e+00, float* %"fractional_part'de", align 4
; CHECK-NEXT:   %"x'de" = alloca float, align 4
; CHECK-NEXT:   store float 0.000000e+00, float* %"x'de", align 4
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:
; CHECK-NEXT:   store float %differeturn, float* %"fractional_part'de", align 4
; CHECK-NEXT:   %0 = load float, float* %"fractional_part'de", align 4
; CHECK-NEXT:   store float 0.000000e+00, float* %"fractional_part'de", align 4
; CHECK-NEXT:   %1 = load float, float* %"x'de", align 4
; CHECK-NEXT:   %2 = fadd fast float %1, %0
; CHECK-NEXT:   store float %2, float* %"x'de", align 4
; CHECK-NEXT:   %3 = load float, float* %"x'de", align 4
; CHECK-NEXT:   %4 = insertvalue { float } undef, float %3, 0
; CHECK-NEXT:   ret { float } %4
; CHECK-NEXT: }

; x86_fp80 tests

; CHECK: define internal { x86_fp80 } @diffetestintl(x86_fp80 %x, x86_fp80 %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"x'de" = alloca x86_fp80, align 16
; CHECK-NEXT:   store x86_fp80 0xK00000000000000000000, x86_fp80* %"x'de", align 16
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:
; CHECK-NEXT:   %0 = load x86_fp80, x86_fp80* %"x'de", align 16
; CHECK-NEXT:   %1 = insertvalue { x86_fp80 } undef, x86_fp80 %0, 0
; CHECK-NEXT:   ret { x86_fp80 } %1
; CHECK-NEXT: }

; CHECK: define internal { x86_fp80 } @diffetestfracl(x86_fp80 %x, x86_fp80 %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"fractional_part'de" = alloca x86_fp80, align 16
; CHECK-NEXT:   store x86_fp80 0xK00000000000000000000, x86_fp80* %"fractional_part'de", align 16
; CHECK-NEXT:   %"x'de" = alloca x86_fp80, align 16
; CHECK-NEXT:   store x86_fp80 0xK00000000000000000000, x86_fp80* %"x'de", align 16
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:
; CHECK-NEXT:   store x86_fp80 %differeturn, x86_fp80* %"fractional_part'de", align 16
; CHECK-NEXT:   %0 = load x86_fp80, x86_fp80* %"fractional_part'de", align 16
; CHECK-NEXT:   store x86_fp80 0xK00000000000000000000, x86_fp80* %"fractional_part'de", align 16
; CHECK-NEXT:   %1 = load x86_fp80, x86_fp80* %"x'de", align 16
; CHECK-NEXT:   %2 = fadd fast x86_fp80 %1, %0
; CHECK-NEXT:   store x86_fp80 %2, x86_fp80* %"x'de", align 16
; CHECK-NEXT:   %3 = load x86_fp80, x86_fp80* %"x'de", align 16
; CHECK-NEXT:   %4 = insertvalue { x86_fp80 } undef, x86_fp80 %3, 0
; CHECK-NEXT:   ret { x86_fp80 } %4
; CHECK-NEXT: }
