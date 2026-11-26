; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | %lli - | FileCheck %s --check-prefix=EVAL
; EVAL: 1.127626 0.521095

; The derivative of sinh calls the companion cosh declared with the callee's
; attributes. Identity attributes such as enzyme_math must not be copied, or
; the companion would be differentiated with sinh's rule and the second
; derivative of sinh would be cosh instead of sinh.

@.fmt = private unnamed_addr constant [7 x i8] c"%f %f\0A\00", align 1

declare double @sinh(double)

; The body only lets the evaluation run link; the rule is selected by the
; enzyme_math attribute and never differentiates it.
define double @my_sinh(double %x) #0 {
entry:
  %r = call double @sinh(double %x)
  ret double %r
}
declare double @__enzyme_autodiff(...)
declare double @__enzyme_fwddiff(...)
declare i32 @printf(ptr, ...)

define double @tester(double %x) {
entry:
  %y = call double @my_sinh(double %x)
  ret double %y
}

define double @first(double %x) {
entry:
  %d = call double (...) @__enzyme_autodiff(ptr @tester, double %x)
  ret double %d
}

; CHECK-LABEL: define i32 @main(
define i32 @main() {
entry:
  ; CHECK: call double @first(double 5.000000e-01)
  %d1 = call double @first(double 5.000000e-01)
  ; CHECK: call fast double @fwddiffefirst(double 5.000000e-01, double 1.000000e+00)
  %d2 = call double (...) @__enzyme_fwddiff(ptr @first, double 5.000000e-01, double 1.000000e+00)
  call i32 (ptr, ...) @printf(ptr @.fmt, double %d1, double %d2)
  ret i32 0
}

; CHECK-LABEL: define internal { double } @diffetester(
; CHECK: call fast double @cosh(double %x)

; CHECK: declare double @cosh(double) #[[COSHATTRS:[0-9]+]]

; CHECK-LABEL: define internal { double } @fwddiffediffetester(
; CHECK: call fast double @sinh(double %x)

; CHECK: attributes #[[COSHATTRS]] = { {{[a-z_() ]*}} }

attributes #0 = { nounwind readnone "enzyme_math"="sinh" }
