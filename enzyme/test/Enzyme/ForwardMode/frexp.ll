; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

declare double @__enzyme_fwddiff(i8*, ...)
declare float @__enzyme_fwddifff(i8*, ...)
declare x86_fp80 @__enzyme_fwddiffl(i8*, ...)

declare double @frexp(double, i32*)
define double @test(double %x) {
entry:
  %exp = alloca i32, align 4
  %call = call double @frexp(double %x, i32* %exp)
  ret double %call
}

define double @dtest(double %x, double %dx) {
entry:
  %call = call double (i8*, ...) @__enzyme_fwddiff(i8* bitcast (double (double)* @test to i8*), double %x, double %dx)
  ret double %call
}

declare float @frexpf(float, i32*)
define float @testf(float %x) {
entry:
  %exp = alloca i32, align 4
  %call = call float @frexpf(float %x, i32* %exp)
  ret float %call
}

define float @dtestf(float %x, float %dx) {
entry:
  %call = call float (i8*, ...) @__enzyme_fwddifff(i8* bitcast (float (float)* @testf to i8*), float %x, float %dx)
  ret float %call
}

declare x86_fp80 @frexpl(x86_fp80, i32*)
define x86_fp80 @testl(x86_fp80 %x) {
entry:
  %exp = alloca i32, align 4
  %call = call x86_fp80 @frexpl(x86_fp80 %x, i32* %exp)
  ret x86_fp80 %call
}

define x86_fp80 @dtestl(x86_fp80 %x, x86_fp80 %dx) {
entry:
  %call = call x86_fp80 (i8*, ...) @__enzyme_fwddiffl(i8* bitcast (x86_fp80 (x86_fp80)* @testl to i8*), x86_fp80 %x, x86_fp80 %dx)
  ret x86_fp80 %call
}

; CHECK: define internal double @fwddiffetest(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = bitcast double %x to i64
; CHECK-NEXT:   %1 = and i64 9218868437227405312, %0
; CHECK-NEXT:   %2 = bitcast i64 %1 to double
; CHECK-NEXT:   %3 = fmul fast double %2, 2.000000e+00
; CHECK-NEXT:   %4 = fdiv fast double %"x'", %3
; CHECK-NEXT:   ret double %4
; CHECK-NEXT: }

; CHECK: define internal float @fwddiffetestf(float %x, float %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = bitcast float %x to i32
; CHECK-NEXT:   %1 = and i32 2139095040, %0
; CHECK-NEXT:   %2 = bitcast i32 %1 to float
; CHECK-NEXT:   %3 = fmul fast float %2, 2.000000e+00
; CHECK-NEXT:   %4 = fdiv fast float %"x'", %3
; CHECK-NEXT:   ret float %4
; CHECK-NEXT: }

; CHECK: define internal x86_fp80 @fwddiffetestl(x86_fp80 %x, x86_fp80 %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = bitcast x86_fp80 %x to i80
; CHECK-NEXT:   %1 = and i80 604453686435277732577280, %0
; CHECK-NEXT:   %2 = bitcast i80 %1 to x86_fp80
; CHECK-NEXT:   %3 = fmul fast x86_fp80 %2, 0xK40008000000000000000
; CHECK-NEXT:   %4 = fdiv fast x86_fp80 %"x'", %3
; CHECK-NEXT:   ret x86_fp80 %4
; CHECK-NEXT: }
