; RUN: if [ %llvmver -lt 16 ]; then not %opt < %s %loadEnzyme -enzyme -S 2>&1 | FileCheck %s; fi
; RUN: not %opt < %s %newLoadEnzyme -passes="enzyme" -S 2>&1 | FileCheck %s

; enzyme_out on a pointer argument is a misuse: the derivative of memory must be
; requested with enzyme_dup and a shadow pointer. Enzyme must say so up front
; instead of failing later inside the derivative.

define double @square(double* %x) {
entry:
  %v = load double, double* %x, align 8
  %m = fmul double %v, %v
  ret double %m
}

declare double @__enzyme_autodiff(...)

define double @dsquare(double* %x) {
entry:
  %r = call double (...) @__enzyme_autodiff(double (double*)* @square, metadata !"enzyme_out", double* %x)
  ret double %r
}

; CHECK: Enzyme: enzyme_out applied to argument 0 of square, which is a pointer (or reference) rather than a value passed by value. Pass the pointer with enzyme_dup followed by a shadow pointer that receives the derivative, or with enzyme_const if it is inactive
