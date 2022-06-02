; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -S | FileCheck %s

declare [4 x double] @__enzyme_batch(...)

define double @add3(double %x, double %a) {
entry:
  %add = fadd double %x, %a
  ret double %add
}

define double @square(double %x, double %a) {
entry:
  %call = call double @add3(double %a, double 3.0)
  %mul = fmul double %x, %call
  ret double %mul
}

define [4 x double] @dsquare(double %x1, double %x2, double %x3, double %x4, double %a) {
entry:
  %call = call [4 x double] (...) @__enzyme_batch(double (double, double)* @square, metadata !"enzyme_width", i64 4, metadata !"enzyme_vector", double %x1, double %x2, double %x3, double %x4, metadata !"enzyme_scalar", double %a)
  ret [4 x double] %call
}


; CHECK: define [4 x double] @batch_square([4 x double] %0, double %1)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %2 = extractvalue [4 x double] %0, 0
; CHECK-NEXT:   %3 = extractvalue [4 x double] %0, 1
; CHECK-NEXT:   %4 = extractvalue [4 x double] %0, 2
; CHECK-NEXT:   %5 = extractvalue [4 x double] %0, 3
; CHECK-NEXT:   %6 = call double @add3(double %1, double 3.000000e+00)
; CHECK-NEXT:   %7 = fmul double %2, %6
; CHECK-NEXT:   %8 = fmul double %3, %6
; CHECK-NEXT:   %9 = fmul double %4, %6
; CHECK-NEXT:   %10 = fmul double %5, %6
; CHECK-NEXT:   %mrv = insertvalue [4 x double] undef, double %7, 0
; CHECK-NEXT:   %mrv1 = insertvalue [4 x double] %mrv, double %8, 1
; CHECK-NEXT:   %mrv2 = insertvalue [4 x double] %mrv1, double %9, 2
; CHECK-NEXT:   %mrv3 = insertvalue [4 x double] %mrv2, double %10, 3
; CHECK-NEXT:   ret [4 x double] %mrv3
; CHECK-NEXT: }