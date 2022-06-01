; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -S | FileCheck %s

declare [4 x double] @__enzyme_batch(...)

define double @square(double %x) {
entry:
  %mul = fmul double %x, %x
  ret double %mul
}

define [4 x double] @dsquare(double %x1, double %x2, double %x3, double %x4) {
entry:
  %call = call [4 x double] (...) @__enzyme_batch(double (double)* @square, metadata !"enzyme_width", i64 4, metadata !"enzyme_vector", double %x1, double %x2, double %x3, double %x4)
  ret [4 x double] %call
}


; CHECK: define [4 x double] @batch_square([4 x double] %0)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %1 = extractvalue [4 x double] %0, 0
; CHECK-NEXT:   %2 = extractvalue [4 x double] %0, 1
; CHECK-NEXT:   %3 = extractvalue [4 x double] %0, 2
; CHECK-NEXT:   %4 = extractvalue [4 x double] %0, 3
; CHECK-NEXT:   %5 = fmul double %1, %1
; CHECK-NEXT:   %6 = fmul double %2, %2
; CHECK-NEXT:   %7 = fmul double %3, %3
; CHECK-NEXT:   %8 = fmul double %4, %4
; CHECK-NEXT:   %mrv = insertvalue [4 x double] undef, double %5, 0
; CHECK-NEXT:   %mrv1 = insertvalue [4 x double] %mrv, double %6, 1
; CHECK-NEXT:   %mrv2 = insertvalue [4 x double] %mrv1, double %7, 2
; CHECK-NEXT:   %mrv3 = insertvalue [4 x double] %mrv2, double %8, 3
; CHECK-NEXT:   ret [4 x double] %mrv3
; CHECK-NEXT: }