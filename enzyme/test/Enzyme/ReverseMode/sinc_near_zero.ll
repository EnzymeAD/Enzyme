; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | %lli - | FileCheck %s --check-prefix=EVAL
; EVAL: -3.33333333e-05 -4.35397775e-01 -3.28986810e-04 -9.02028130e-01
; EVAL-NEXT: -3.28986813e+00 -3.28986813e+00 -3.28986813e+00

; The sinc and sincn derivative rules must stay accurate near zero, where the
; closed form (cos(x) - sinc(x)) / x cancels, and must remain differentiable at
; zero so that forward-over-forward and reverse-over-forward second derivatives
; equal -1/3 and -pi^2/3 rather than 0 or NaN.

@.fmt1 = private unnamed_addr constant [41 x i8] c"%.8e %.8e %.8e %.8e\0A\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00", align 1
@.fmt2 = private unnamed_addr constant [41 x i8] c"%.8e %.8e %.8e\0A\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00", align 1

declare double @llvm.sin.f64(double)

; Bodies are provided only so that the evaluation run can link; the derivative
; rules are selected by name and never differentiate these bodies.
define double @sinc(double %x) nounwind readnone {
entry:
  %zero = fcmp oeq double %x, 0.000000e+00
  %s = call double @llvm.sin.f64(double %x)
  %q = fdiv double %s, %x
  %r = select i1 %zero, double 1.000000e+00, double %q
  ret double %r
}

define double @sincn(double %x) nounwind readnone {
entry:
  %px = fmul double %x, 0x400921FB54442D18
  %r = call double @sinc(double %px)
  ret double %r
}
declare double @__enzyme_fwddiff(...)
declare double @__enzyme_autodiff(...)
declare i32 @printf(ptr, ...)

define double @unnormalized(double %x) {
entry:
  %r = call double @sinc(double %x)
  ret double %r
}

define double @normalized(double %x) {
entry:
  %r = call double @sincn(double %x)
  ret double %r
}

define double @normalized_derivative(double %x) {
entry:
  %d = call double (...) @__enzyme_fwddiff(ptr @normalized, double %x, double 1.000000e+00)
  ret double %d
}

define double @normalized_gradient(double %x) {
entry:
  %d = call double (...) @__enzyme_autodiff(ptr @normalized, double %x)
  ret double %d
}

; CHECK-LABEL: define i32 @main(
define i32 @main() {
entry:
  ; CHECK: call fast double @fwddiffeunnormalized(double 1.000000e-04, double 1.000000e+00)
  %d_sinc_small = call double (...) @__enzyme_fwddiff(ptr @unnormalized, double 1.000000e-04, double 1.000000e+00)
  ; CHECK: call { double } @diffeunnormalized(double 2.000000e+00, double 1.000000e+00)
  %d_sinc_large = call double (...) @__enzyme_autodiff(ptr @unnormalized, double 2.000000e+00)
  ; CHECK: call fast double @fwddiffenormalized(double 1.000000e-04, double 1.000000e+00)
  %d_sincn_small = call double (...) @__enzyme_fwddiff(ptr @normalized, double 1.000000e-04, double 1.000000e+00)
  ; CHECK: call { double } @diffenormalized(double 3.000000e-01, double 1.000000e+00)
  %d_sincn_mid = call double (...) @__enzyme_autodiff(ptr @normalized, double 3.000000e-01)
  call i32 (ptr, ...) @printf(ptr @.fmt1, double %d_sinc_small, double %d_sinc_large, double %d_sincn_small, double %d_sincn_mid)
  ; CHECK: call fast double @fwddiffenormalized_derivative(double 0.000000e+00, double 1.000000e+00)
  %dd_fwd_fwd = call double (...) @__enzyme_fwddiff(ptr @normalized_derivative, double 0.000000e+00, double 1.000000e+00)
  ; CHECK: call { double } @diffenormalized_derivative(double 0.000000e+00, double 1.000000e+00)
  %dd_rev_fwd = call double (...) @__enzyme_autodiff(ptr @normalized_derivative, double 0.000000e+00)
  ; CHECK: call fast double @fwddiffenormalized_gradient(double 0.000000e+00, double 1.000000e+00)
  %dd_fwd_rev = call double (...) @__enzyme_fwddiff(ptr @normalized_gradient, double 0.000000e+00, double 1.000000e+00)
  call i32 (ptr, ...) @printf(ptr @.fmt2, double %dd_fwd_fwd, double %dd_rev_fwd, double %dd_fwd_rev)
  ret i32 0
}

; CHECK-LABEL: define internal { double } @diffeunnormalized(
; CHECK: %[[abs:.+]] = call fast double @llvm.fabs.f64(double %x)
; CHECK-NEXT: %[[small:.+]] = fcmp fast olt double %[[abs]], 4.400000e-01
; CHECK: %[[safe:.+]] = select {{(fast )?}}i1 %[[small]], double 1.000000e+00, double %x
; CHECK-NEXT: %[[cos:.+]] = call fast double @llvm.cos.f64(double %[[safe]])
; CHECK-NEXT: %[[sinc:.+]] = call fast double @sinc(double %[[safe]])
; CHECK-NEXT: %[[num:.+]] = fsub fast double %[[cos]], %[[sinc]]
; CHECK-NEXT: %[[closed:.+]] = fdiv fast double %[[num]], %[[safe]]
; CHECK-NEXT: %[[deriv:.+]] = select {{(fast )?}}i1 %[[small]], double %{{.+}}, double %[[closed]]
