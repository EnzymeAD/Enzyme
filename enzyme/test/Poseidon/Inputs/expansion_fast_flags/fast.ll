; FP64 operations carrying every fast-math flag, for expansion_fast_flags.cpp.

define double @ex2_add(double %a, double %b) {
  %r = fadd fast double %a, %b
  ret double %r
}

define double @ex2_mul(double %a, double %b) {
  %r = fmul fast double %a, %b
  ret double %r
}

define double @ex2_div(double %a, double %b) {
  %r = fdiv fast double %a, %b
  ret double %r
}

define double @ex2_fma(double %a, double %b, double %c) {
  %r = call fast double @llvm.fmuladd.f64(double %a, double %b, double %c)
  ret double %r
}

define double @ex3_mul(double %a, double %b) {
  %r = fmul fast double %a, %b
  ret double %r
}

define double @ex4_add(double %a, double %b) {
  %r = fadd fast double %a, %b
  ret double %r
}

declare double @llvm.fmuladd.f64(double, double, double)
