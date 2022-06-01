; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -S | FileCheck %s

declare [4 x double] @__enzyme_batch(...)

define double @square_add(double %x, double %y) {
entry:
  %mul = fmul double %x, %x
  %add = fadd double %mul, %y
  ret double %add
}

define [4 x double] @dsquare(double %x1, double %x2, double %x3, double %x4, double %y) {
entry:
  %call = call [4 x double] (...) @__enzyme_batch(double (double, double)* @square_add, metadata !"enzyme_width", i64 4, metadata !"enzyme_vector", double %x1, double %x2, double %x3, double %x4, metadata !"enzyme_scalar", double %y)
  ret [4 x double] %call
}


; CHECK: define [4 x double] @batch_square_add([4 x double] %0, double %1)
; CHECK-NEXT: entry:
; CHECK:   %2 = extractvalue [4 x double] %0, 0
; CHECK:   %3 = extractvalue [4 x double] %0, 1
; CHECK:   %4 = extractvalue [4 x double] %0, 2
; CHECK:   %5 = extractvalue [4 x double] %0, 3
; CHECK:   %6 = fmul double %2, %2
; CHECK:   %7 = fmul double %3, %3
; CHECK:   %8 = fmul double %4, %4
; CHECK:   %9 = fmul double %5, %5
; CHECK:   %10 = fadd double %6, %1
; CHECK:   %11 = fadd double %7, %1
; CHECK:   %12 = fadd double %8, %1
; CHECK:   %13 = fadd double %9, %1
; CHECK:   %mrv = insertvalue [4 x double] undef, double %10, 0
; CHECK:   %mrv1 = insertvalue [4 x double] %mrv, double %11, 1
; CHECK:   %mrv2 = insertvalue [4 x double] %mrv1, double %12, 2
; CHECK:   %mrv3 = insertvalue [4 x double] %mrv2, double %13, 3
; CHECK:   ret [4 x double] %mrv3
; CHECK: }