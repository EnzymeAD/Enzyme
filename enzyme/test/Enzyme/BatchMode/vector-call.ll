; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -S | FileCheck %s

declare [4 x double] @__enzyme_batch(...)


define double @add3(double %x, double %a) {
entry:
  %add = fadd double %x, %a
  ret double %add
}

define double @square(double %x, double %a) {
entry:
  %mul = fmul double %x, %x
  %call = call double @add3(double %mul, double %a)
  ret double %call
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
; CHECK-NEXT:   %6 = fmul double %2, %2
; CHECK-NEXT:   %7 = fmul double %3, %3
; CHECK-NEXT:   %8 = fmul double %4, %4
; CHECK-NEXT:   %9 = fmul double %5, %5
; CHECK-NEXT:   %10 = insertvalue [4 x double] undef, double %6, 0
; CHECK-NEXT:   %11 = insertvalue [4 x double] undef, double %7, 1
; CHECK-NEXT:   %12 = insertvalue [4 x double] undef, double %8, 2
; CHECK-NEXT:   %13 = insertvalue [4 x double] undef, double %9, 3
; CHECK-NEXT:   %14 = call [4 x double] @batch_add3([4 x double] undef, double %1)
; CHECK-NEXT:   %15 = extractvalue [4 x double] %14, 0
; CHECK-NEXT:   %16 = extractvalue [4 x double] %14, 1
; CHECK-NEXT:   %17 = extractvalue [4 x double] %14, 2
; CHECK-NEXT:   %18 = extractvalue [4 x double] %14, 3
; CHECK-NEXT:   %mrv = insertvalue [4 x double] undef, double %15, 0
; CHECK-NEXT:   %mrv1 = insertvalue [4 x double] %mrv, double %16, 1
; CHECK-NEXT:   %mrv2 = insertvalue [4 x double] %mrv1, double %17, 2
; CHECK-NEXT:   %mrv3 = insertvalue [4 x double] %mrv2, double %18, 3
; CHECK-NEXT:   ret [4 x double] %mrv3
; CHECK-NEXT: }

; CHECK: define [4 x double] @batch_add3([4 x double] %0, double %1)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %2 = extractvalue [4 x double] %0, 0
; CHECK-NEXT:   %3 = extractvalue [4 x double] %0, 1
; CHECK-NEXT:   %4 = extractvalue [4 x double] %0, 2
; CHECK-NEXT:   %5 = extractvalue [4 x double] %0, 3
; CHECK-NEXT:   %6 = fadd double %2, %1
; CHECK-NEXT:   %7 = fadd double %3, %1
; CHECK-NEXT:   %8 = fadd double %4, %1
; CHECK-NEXT:   %9 = fadd double %5, %1
; CHECK-NEXT:   %mrv = insertvalue [4 x double] undef, double %6, 0
; CHECK-NEXT:   %mrv1 = insertvalue [4 x double] %mrv, double %7, 1
; CHECK-NEXT:   %mrv2 = insertvalue [4 x double] %mrv1, double %8, 2
; CHECK-NEXT:   %mrv3 = insertvalue [4 x double] %mrv2, double %9, 3
; CHECK-NEXT:   ret [4 x double] %mrv3
; CHECK-NEXT: }