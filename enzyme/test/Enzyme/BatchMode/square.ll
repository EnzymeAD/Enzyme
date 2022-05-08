; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -early-cse -S | FileCheck %s

declare [3 x double] @__enzyme_batch(...)

define double @square(double %x) {
entry:
  %mul = fmul fast double %x, %x
  ret double %mul
}

define [3 x double] @dsquare(double %x) {
entry:
  %0 = tail call [3 x double] (...) @__enzyme_batch(double (double)* nonnull @square, metadata !"enzyme_width", i64 3, double %x, double 1.0, double 10.0, double 100.0)
  ret [3 x double] %0
}


; CHECK: define internal [3 x double] @batch3square(double %x, [3 x double] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue [3 x double] %"x'", 0
; CHECK-NEXT:   %1 = fmul fast double %0, %0
; CHECK-NEXT:   %2 = insertvalue [3 x double] undef, double %1, 0
; CHECK-NEXT:   %3 = extractvalue [3 x double] %"x'", 1
; CHECK-NEXT:   %4 = fmul fast double %3, %3
; CHECK-NEXT:   %5 = insertvalue [3 x double] %2, double %4, 1
; CHECK-NEXT:   %6 = extractvalue [3 x double] %"x'", 2
; CHECK-NEXT:   %7 = fmul fast double %6, %6
; CHECK-NEXT:   %8 = insertvalue [3 x double] %5, double %7, 2
; CHECK-NEXT:   ret [3 x double] %8
; CHECK-NEXT: }