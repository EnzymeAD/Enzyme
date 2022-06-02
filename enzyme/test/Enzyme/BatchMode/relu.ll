; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -S | FileCheck %s

define double @relu(double %x, double %a) {
entry:
  %cmp = fcmp fast ogt double %x, 0.000000e+00
  br i1 %cmp, label %cond.true, label %cond.end

cond.true:                                        ; preds = %entry
  %ax = fmul double %x, %a
  ret double %ax

cond.end:                                         ; preds = %entry, %cond.true
  ret double %x
}

define [4 x double] @vecrelu(double %x, double %a1, double %a2, double %a3, double %a4) {
entry:
  %0 = tail call [4 x double] (...) @__enzyme_batch(double (double, double)* nonnull @relu, metadata !"enzyme_width", i64 4, metadata !"enzyme_scalar", double %x, metadata !"enzyme_vector", double %a1, double %a2, double %a3, double %a4)
  ret [4 x double] %0
}

declare [4 x double] @__enzyme_batch(...)


; CHECK: define [4 x double] @batch_relu(double %0, [4 x double] %1)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %2 = extractvalue [4 x double] %1, 0
; CHECK-NEXT:   %3 = extractvalue [4 x double] %1, 1
; CHECK-NEXT:   %4 = extractvalue [4 x double] %1, 2
; CHECK-NEXT:   %5 = extractvalue [4 x double] %1, 3
; CHECK-NEXT:   %6 = fcmp fast ogt double %0, 0.000000e+00
; CHECK-NEXT:   br i1 %6, label %cond.true, label %cond.end

; CHECK: cond.true:                                        ; preds = %entry
; CHECK-NEXT:   %7 = fmul double %0, %2
; CHECK-NEXT:   %8 = fmul double %0, %3
; CHECK-NEXT:   %9 = fmul double %0, %4
; CHECK-NEXT:   %10 = fmul double %0, %5
; CHECK-NEXT:   %mrv = insertvalue [4 x double] undef, double %7, 0
; CHECK-NEXT:   %mrv1 = insertvalue [4 x double] %mrv, double %8, 1
; CHECK-NEXT:   %mrv2 = insertvalue [4 x double] %mrv1, double %9, 2
; CHECK-NEXT:   %mrv3 = insertvalue [4 x double] %mrv2, double %10, 3
; CHECK-NEXT:   ret [4 x double] %mrv3

; CHECK: cond.end:                                         ; preds = %entry
; CHECK-NEXT:   %mrv4 = insertvalue [4 x double] undef, double %0, 0
; CHECK-NEXT:   %mrv5 = insertvalue [4 x double] %mrv4, double %0, 1
; CHECK-NEXT:   %mrv6 = insertvalue [4 x double] %mrv5, double %0, 2
; CHECK-NEXT:   %mrv7 = insertvalue [4 x double] %mrv6, double %0, 3
; CHECK-NEXT:   ret [4 x double] %mrv7
; CHECK-NEXT: }