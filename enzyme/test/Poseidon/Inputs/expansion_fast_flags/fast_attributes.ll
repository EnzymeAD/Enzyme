; fast.ll in functions that carry every fast-math attribute, for
; expansion_fast_math_attributes.cpp.

define double @ex2_add(double %a, double %b) #0 {
  %r = fadd fast double %a, %b
  ret double %r
}

define double @ex2_mul(double %a, double %b) #0 {
  %r = fmul fast double %a, %b
  ret double %r
}

define double @ex2_div(double %a, double %b) #0 {
  %r = fdiv fast double %a, %b
  ret double %r
}

define double @ex2_fma(double %a, double %b, double %c) #0 {
  %r = call fast double @llvm.fmuladd.f64(double %a, double %b, double %c)
  ret double %r
}

define double @ex3_mul(double %a, double %b) #0 {
  %r = fmul fast double %a, %b
  ret double %r
}

define double @ex4_add(double %a, double %b) #0 {
  %r = fadd fast double %a, %b
  ret double %r
}

declare double @llvm.fmuladd.f64(double, double, double)

attributes #0 = { "unsafe-fp-math"="true" "no-nans-fp-math"="true" "no-infs-fp-math"="true" "approx-func-fp-math"="true" "no-signed-zeros-fp-math"="true" }
