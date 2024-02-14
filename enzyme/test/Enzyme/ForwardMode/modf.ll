; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

declare double @__enzyme_fwddiff(i8*, ...)
declare float @__enzyme_fwddifff(i8*, ...)
declare x86_fp80 @__enzyme_fwddiffl(i8*, ...)

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
  %call = call double (i8*, ...) @__enzyme_fwddiff(i8* bitcast (double (double)* @testint to i8*), double %x, double %dx)
  ret double %call
}
define double @dtestfrac(double %x, double %dx) {
entry:
  %call = call double (i8*, ...) @__enzyme_fwddiff(i8* bitcast (double (double)* @testfrac to i8*), double %x, double %dx)
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
  %call = call float (i8*, ...) @__enzyme_fwddifff(i8* bitcast (float (float)* @testintf to i8*), float %x, float %dx)
  ret float %call
}
define float @dtestfracf(float %x, float %dx) {
entry:
  %call = call float (i8*, ...) @__enzyme_fwddifff(i8* bitcast (float (float)* @testfracf to i8*), float %x, float %dx)
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
  %call = call x86_fp80 (i8*, ...) @__enzyme_fwddiffl(i8* bitcast (x86_fp80 (x86_fp80)* @testintl to i8*), x86_fp80 %x, x86_fp80 %dx)
  ret x86_fp80 %call
}
define x86_fp80 @dtestfracl(x86_fp80 %x, x86_fp80 %dx) {
entry:
  %call = call x86_fp80 (i8*, ...) @__enzyme_fwddiffl(i8* bitcast (x86_fp80 (x86_fp80)* @testfracl to i8*), x86_fp80 %x, x86_fp80 %dx)
  ret x86_fp80 %call
}

; tests

; CHECK: define internal double @fwddiffetestint(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret double 0.000000e+00
; CHECK-NEXT: }

; CHECK: define internal double @fwddiffetestfrac(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret double %"x'"
; CHECK-NEXT: }

; CHECK: define internal float @fwddiffetestintf(float %x, float %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret float 0.000000e+00
; CHECK-NEXT: }

; CHECK: define internal float @fwddiffetestfracf(float %x, float %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret float %"x'"
; CHECK-NEXT: }

; CHECK: define internal x86_fp80 @fwddiffetestintl(x86_fp80 %x, x86_fp80 %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret x86_fp80 0xK00000000000000000000
; CHECK-NEXT: }

; CHECK: define internal x86_fp80 @fwddiffetestfracl(x86_fp80 %x, x86_fp80 %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret x86_fp80 %"x'"
; CHECK-NEXT: }

