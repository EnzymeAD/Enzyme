; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -enzyme-inline=1 -passes="enzyme,default<O2>" -S | %lli -

declare double @sin(double)
declare void @abort()
declare double @llvm.fabs.f64(double)
declare float @llvm.fabs.f32(float)
declare double @__enzyme_autodiff(...)
declare float @__enzyme_autodiff_f32(...)
declare double @__enzyme_fwddiff(...)
declare float @__enzyme_fwddiff_f32(...)
declare { ptr, double } @__enzyme_augmentfwd(...)

; Body intentionally divides by zero at x = 0 (#2450); enzyme_math="sinc" must
; dispatch to the sinc rule instead of differentiating the body.
define double @my_sinc(double %x) "enzyme_math"="sinc" {
entry:
  %s = call double @sin(double %x)
  %q = fdiv double %s, %x
  ret double %q
}

; Body intentionally has derivative 2.0 instead of 1 / (1 + x).
define double @my_log1p(double %x) "enzyme_math"="log1p" {
entry:
  %wrong = fmul double %x, 2.0
  ret double %wrong
}

declare double @decl_log1p(double) "enzyme_math"="log1p"

; Constant body returns 10.0 (exp10f(1.0)), so differentiating the body would
; yield 0.0 instead of 10.0 * ln(10).
define float @my_exp10f(float %x) "enzyme_math"="exp10f" {
entry:
  ret float 1.000000e+01
}

; Fallback implementation carrying an unregistered enzyme_math name must have
; its body differentiated rather than recursing into a self-call wrapper.
define double @fallback_unregistered(double %x) "enzyme_math"="cblas_zdotc_sub64_" {
entry:
  %sq = fmul double %x, %x
  ret double %sq
}

define i32 @main() {
entry:
  ; Reverse mode on my_sinc at x = 0 -> 0.0 (not NaN)
  %dsinc0 = call double (...) @__enzyme_autodiff(ptr @my_sinc, double 0.0)
  %isnan0 = fcmp uno double %dsinc0, 0.0
  %bad0 = fcmp one double %dsinc0, 0.0
  %fail0 = or i1 %isnan0, %bad0
  br i1 %fail0, label %fail, label %check_fwd_sinc

check_fwd_sinc:
  ; Forward mode on my_sinc at x = 0 -> 0.0 (not NaN)
  %fsinc0 = call double (...) @__enzyme_fwddiff(ptr @my_sinc, double 0.0, double 1.0)
  %isnan1 = fcmp uno double %fsinc0, 0.0
  %bad1 = fcmp one double %fsinc0, 0.0
  %fail1 = or i1 %isnan1, %bad1
  br i1 %fail1, label %fail, label %check_rev_log1p

check_rev_log1p:
  ; Reverse mode on my_log1p(2.0) -> 1/3
  %dlog = call double (...) @__enzyme_autodiff(ptr @my_log1p, double 2.0)
  %target = fdiv double 1.0, 3.0
  %err2 = fsub double %dlog, %target
  %abs2 = call double @llvm.fabs.f64(double %err2)
  %bad2 = fcmp ogt double %abs2, 1.0e-12
  br i1 %bad2, label %fail, label %check_fwd_log1p

check_fwd_log1p:
  ; Forward mode on my_log1p(2.0) -> 1/3
  %flog = call double (...) @__enzyme_fwddiff(ptr @my_log1p, double 2.0, double 1.0)
  %err3 = fsub double %flog, %target
  %abs3 = call double @llvm.fabs.f64(double %err3)
  %bad3 = fcmp ogt double %abs3, 1.0e-12
  br i1 %bad3, label %fail, label %check_split_log1p

check_split_log1p:
  ; Split reverse mode on my_log1p(2.0) -> primal 4.0, derivative 1/3
  %aug = call { ptr, double } (...) @__enzyme_augmentfwd(ptr @my_log1p, double 2.0)
  %tape = extractvalue { ptr, double } %aug, 0
  %primal = extractvalue { ptr, double } %aug, 1
  %bad_primal = fcmp one double %primal, 4.0
  br i1 %bad_primal, label %fail, label %check_split_grad

check_split_grad:
  %slog = call double (...) @__enzyme_autodiff(ptr @my_log1p, metadata !"enzyme_tape", ptr %tape, double 2.0)
  %err4 = fsub double %slog, %target
  %abs4 = call double @llvm.fabs.f64(double %err4)
  %bad4 = fcmp ogt double %abs4, 1.0e-12
  br i1 %bad4, label %fail, label %check_decl_log1p

check_decl_log1p:
  ; Empty declaration with enzyme_math="log1p" -> 1/3
  %decl_grad = call double (...) @__enzyme_autodiff(ptr @decl_log1p, double 2.0)
  %err5 = fsub double %decl_grad, %target
  %abs5 = call double @llvm.fabs.f64(double %err5)
  %bad5 = fcmp ogt double %abs5, 1.0e-12
  br i1 %bad5, label %fail, label %check_exp10f

check_exp10f:
  ; exp10f(1.0) -> 10.0 * ln(10) = 23.0258509f
  %dexp10f = call float (...) @__enzyme_autodiff_f32(ptr @my_exp10f, float 1.000000e+00)
  %fexp10f = call float (...) @__enzyme_fwddiff_f32(ptr @my_exp10f, float 1.000000e+00, float 1.000000e+00)
  %err6 = fsub float %dexp10f, 0x4037069E60000000
  %abs6 = call float @llvm.fabs.f32(float %err6)
  %bad6 = fcmp ogt float %abs6, 9.765625e-04
  %err7 = fsub float %fexp10f, 0x4037069E60000000
  %abs7 = call float @llvm.fabs.f32(float %err7)
  %bad7 = fcmp ogt float %abs7, 9.765625e-04
  %fail_exp10f = or i1 %bad6, %bad7
  br i1 %fail_exp10f, label %fail, label %check_fallback

check_fallback:
  ; Unregistered enzyme_math falls back to differentiating the body: d/dx(x^2) at 3 = 6
  %dfall = call double (...) @__enzyme_autodiff(ptr @fallback_unregistered, double 3.0)
  %ffall = call double (...) @__enzyme_fwddiff(ptr @fallback_unregistered, double 3.0, double 1.0)
  %err8 = fsub double %dfall, 6.0
  %abs8 = call double @llvm.fabs.f64(double %err8)
  %bad8 = fcmp ogt double %abs8, 1.0e-12
  %err9 = fsub double %ffall, 6.0
  %abs9 = call double @llvm.fabs.f64(double %err9)
  %bad9 = fcmp ogt double %abs9, 1.0e-12
  %fail_fall = or i1 %bad8, %bad9
  br i1 %fail_fall, label %fail, label %ok

fail:
  call void @abort()
  unreachable

ok:
  ret i32 0
}

; CHECK-LABEL: define internal { double } @diffemy_sinc(
; CHECK: fcmp fast oeq double %x, 0.000000e+00
; CHECK: select {{.*}}i1 %{{.+}}, double 0.000000e+00, double %{{.+}}
; CHECK: ret { double }

; CHECK-LABEL: define internal { double } @diffemy_log1p(
; CHECK: %[[DRET1:.+]] = load double, ptr %{{.+}}
; CHECK: %[[DENOM1:.+]] = fadd fast double %x, 1.000000e+00
; CHECK: fdiv fast double %[[DRET1]], %[[DENOM1]]
; CHECK: ret { double }

; CHECK-LABEL: define internal double @fwddiffemy_log1p(
; CHECK: %[[FDENOM:.+]] = fadd fast double %x, 1.000000e+00
; CHECK: fdiv fast double %{{.+}}, %[[FDENOM]]
; CHECK: ret double

; CHECK-LABEL: define internal { double } @diffedecl_log1p(
; CHECK: %[[DRET2:.+]] = load double, ptr %{{.+}}
; CHECK: %[[DENOM2:.+]] = fadd fast double %{{.+}}, 1.000000e+00
; CHECK: fdiv fast double %[[DRET2]], %[[DENOM2]]
; CHECK: ret { double }
