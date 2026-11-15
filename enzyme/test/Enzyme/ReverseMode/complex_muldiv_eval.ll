; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | %lli - | FileCheck %s --check-prefix=EVAL
; CHECK-NOT: @diffe__muldc3(
; CHECK-NOT: @fwddiffe__muldc3(
; CHECK-NOT: @diffe__divdc3(
; CHECK-NOT: @fwddiffe__divdc3(
; EVAL: -6.00 -17.00 -20.00 -35.00 -65.00
; EVAL-NEXT: 0.72 -0.04 -7.84 1.88 -0.08

; Verify forward and reverse derivatives of __muldc3 and __divdc3 numerically
; against exact analytic values for L(z) = 2*Re(z) - 3*Im(z).

@.fmt = private unnamed_addr constant [26 x i8] c"%.2f %.2f %.2f %.2f %.2f\0A\00", align 1

; Provide self-contained definitions so lli does not depend on compiler-rt
; exporting __muldc3/__divdc3 dynamically on macOS.
define { double, double } @__muldc3(double %a, double %b, double %c, double %d) noinline {
entry:
  %ac = fmul double %a, %c
  %bd = fmul double %b, %d
  %ad = fmul double %a, %d
  %bc = fmul double %b, %c
  %re = fsub double %ac, %bd
  %im = fadd double %ad, %bc
  %s0 = insertvalue { double, double } undef, double %re, 0
  %s1 = insertvalue { double, double } %s0, double %im, 1
  ret { double, double } %s1
}

define { double, double } @__divdc3(double %a, double %b, double %c, double %d) noinline {
entry:
  %cc = fmul double %c, %c
  %dd = fmul double %d, %d
  %denom = fadd double %cc, %dd
  %ac = fmul double %a, %c
  %bd = fmul double %b, %d
  %bc = fmul double %b, %c
  %ad = fmul double %a, %d
  %num_re = fadd double %ac, %bd
  %num_im = fsub double %bc, %ad
  %re = fdiv double %num_re, %denom
  %im = fdiv double %num_im, %denom
  %s0 = insertvalue { double, double } undef, double %re, 0
  %s1 = insertvalue { double, double } %s0, double %im, 1
  ret { double, double } %s1
}

declare { double, double, double, double } @__enzyme_autodiff(...)
declare double @__enzyme_fwddiff(...)
declare i32 @printf(ptr, ...)

define double @mul_loss(double %xre, double %xim, double %yre, double %yim) {
entry:
  %prod = call { double, double } @__muldc3(double %xre, double %xim, double %yre, double %yim)
  %re = extractvalue { double, double } %prod, 0
  %im = extractvalue { double, double } %prod, 1
  %w_re = fmul double 2.000000e+00, %re
  %w_im = fmul double 3.000000e+00, %im
  %loss = fsub double %w_re, %w_im
  ret double %loss
}

define double @div_loss(double %xre, double %xim, double %yre, double %yim) {
entry:
  %quot = call { double, double } @__divdc3(double %xre, double %xim, double %yre, double %yim)
  %re = extractvalue { double, double } %quot, 0
  %im = extractvalue { double, double } %quot, 1
  %w_re = fmul double 2.000000e+00, %re
  %w_im = fmul double 3.000000e+00, %im
  %loss = fsub double %w_re, %w_im
  ret double %loss
}

; CHECK-LABEL: define i32 @main(
define i32 @main() {
entry:
  ; CHECK: call { double, double, double, double } @diffemul_loss(
  %g_mul = call { double, double, double, double } (...) @__enzyme_autodiff(ptr @mul_loss, double 5.000000e+00, double 1.000000e+01, double 3.000000e+00, double 4.000000e+00)
  %gm0 = extractvalue { double, double, double, double } %g_mul, 0
  %gm1 = extractvalue { double, double, double, double } %g_mul, 1
  %gm2 = extractvalue { double, double, double, double } %g_mul, 2
  %gm3 = extractvalue { double, double, double, double } %g_mul, 3
  ; CHECK: call fast double @fwddiffemul_loss(
  %f_mul = call double (...) @__enzyme_fwddiff(ptr @mul_loss, double 5.000000e+00, double 1.000000e+00, double 1.000000e+01, double 2.000000e+00, double 3.000000e+00, double 3.000000e+00, double 4.000000e+00, double -1.000000e+00)
  call i32 (ptr, ...) @printf(ptr @.fmt, double %gm0, double %gm1, double %gm2, double %gm3, double %f_mul)

  ; CHECK: call { double, double, double, double } @diffediv_loss(
  %g_div = call { double, double, double, double } (...) @__enzyme_autodiff(ptr @div_loss, double 2.500000e+01, double 5.000000e+01, double 3.000000e+00, double 4.000000e+00)
  %gd0 = extractvalue { double, double, double, double } %g_div, 0
  %gd1 = extractvalue { double, double, double, double } %g_div, 1
  %gd2 = extractvalue { double, double, double, double } %g_div, 2
  %gd3 = extractvalue { double, double, double, double } %g_div, 3
  ; CHECK: call fast double @fwddiffediv_loss(
  %f_div = call double (...) @__enzyme_fwddiff(ptr @div_loss, double 2.500000e+01, double 5.000000e+00, double 5.000000e+01, double -1.000000e+01, double 3.000000e+00, double 1.000000e+00, double 4.000000e+00, double 2.000000e+00)
  call i32 (ptr, ...) @printf(ptr @.fmt, double %gd0, double %gd1, double %gd2, double %gd3, double %f_div)
  ret i32 0
}
