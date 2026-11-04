; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -S | %lli - | FileCheck %s --check-prefix=EVAL; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | %lli - | FileCheck %s --check-prefix=EVAL

; tanh'(x) = sech(x)^2, computed as 4u / (1 + u)^2 with u = exp(-2|x|).
; 1 - tanh(x)^2 is 0 once tanh(x) rounds to 1 (|x| > 19 in double, > 9 in
; float), and 1 / cosh(x)^2 gives NaN in second derivatives once cosh
; overflows. The second derivative is taken forward over reverse.

; EVAL: double 0.5: 7.8645e-01 2nd -7.2686e-01
; EVAL: double 20: 1.6993e-17 2nd -3.3987e-17
; EVAL: double -20: 1.6993e-17 2nd 3.3987e-17
; EVAL: double 800: 0.0000e+00 2nd {{-?}}0.0000e+00
; EVAL: float 0.5: 7.8645e-01
; EVAL: float 10: 8.2446e-09

define double @tester(double %x) {
entry:
  %0 = tail call fast double @tanh(double %x)
  ret double %0
}

define double @grad(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @tester, double %x)
  ret double %0
}

define double @hess(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @grad, double %x, double 1.0)
  ret double %0
}

define float @testerf(float %x) {
entry:
  %0 = tail call fast float @tanhf(float %x)
  ret float %0
}

define float @gradf(float %x) {
entry:
  %0 = tail call float (float (float)*, ...) @__enzyme_autodiff.f(float (float)* nonnull @testerf, float %x)
  ret float %0
}

@fmt = private unnamed_addr constant [26 x i8] c"double %g: %.4e 2nd %.4e\0A\00"
@fmtf = private unnamed_addr constant [16 x i8] c"float %g: %.4e\0A\00"

define void @evald(double %x) {
entry:
  %g = call double @grad(double %x)
  %h = call double @hess(double %x)
  %r = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([26 x i8], [26 x i8]* @fmt, i64 0, i64 0), double %x, double %g, double %h)
  ret void
}

define void @evalf(float %x) {
entry:
  %g = call float @gradf(float %x)
  %xd = fpext float %x to double
  %gd = fpext float %g to double
  %r = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([16 x i8], [16 x i8]* @fmtf, i64 0, i64 0), double %xd, double %gd)
  ret void
}

define i32 @main() {
entry:
  call void @evald(double 0.5)
  call void @evald(double 20.0)
  call void @evald(double -20.0)
  call void @evald(double 800.0)
  call void @evalf(float 0.5)
  call void @evalf(float 10.0)
  ret i32 0
}

declare double @tanh(double)
declare float @tanhf(float)
declare i32 @printf(i8*, ...)
declare double @__enzyme_autodiff(double (double)*, ...)
declare float @__enzyme_autodiff.f(float (float)*, ...)
declare double @__enzyme_fwddiff(double (double)*, ...)

; CHECK: define internal { double } @diffetester(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call fast double @llvm.fabs.f64(double %x)
; CHECK-NEXT:   %1 = fmul fast double -2.000000e+00, %0
; CHECK-NEXT:   %2 = call fast double @llvm.exp.f64(double %1)
; CHECK-NEXT:   %3 = fmul fast double 4.000000e+00, %2
; CHECK-NEXT:   %4 = fmul fast double %differeturn, %3
; CHECK-NEXT:   %5 = fadd fast double 1.000000e+00, %2
; CHECK-NEXT:   %6 = fmul fast double %5, %5
; CHECK-NEXT:   %7 = fdiv fast double %4, %6
; CHECK-NEXT:   %8 = insertvalue { double } undef, double %7, 0
; CHECK-NEXT:   ret { double } %8
; CHECK-NEXT: }

; CHECK: define internal { float } @diffetesterf(float %x, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call fast float @llvm.fabs.f32(float %x)
; CHECK-NEXT:   %1 = fmul fast float -2.000000e+00, %0
; CHECK-NEXT:   %2 = call fast float @llvm.exp.f32(float %1)
; CHECK-NEXT:   %3 = fmul fast float 4.000000e+00, %2
; CHECK-NEXT:   %4 = fmul fast float %differeturn, %3
; CHECK-NEXT:   %5 = fadd fast float 1.000000e+00, %2
; CHECK-NEXT:   %6 = fmul fast float %5, %5
; CHECK-NEXT:   %7 = fdiv fast float %4, %6
; CHECK-NEXT:   %8 = insertvalue { float } undef, float %7, 0
; CHECK-NEXT:   ret { float } %8
; CHECK-NEXT: }
