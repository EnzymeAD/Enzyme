; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -S | FileCheck %s

define void @addOneMem(double* nocapture %x) {
entry:
  %0 = load double, double* %x, align 8
  %add = fadd fast double %0, 1.000000e+00
  store double %add, double* %x, align 8
  ret void
}

define void @test(double* %x1, double* %x2, double* %x3, double* %x4) {
entry:
  tail call void (...) @__enzyme_batch(void (double*)* nonnull @addOneMem, metadata !"enzyme_width", i64 4, metadata !"enzyme_vector", double* %x1, double* %x2, double* %x3, double* %x4)
  ret void
}

declare void @__enzyme_batch(...)


; CHECK: ddefine void @batch_addOneMem([4 x double*] %0)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %1 = extractvalue [4 x double*] %0, 0
; CHECK-NEXT:   %2 = extractvalue [4 x double*] %0, 1
; CHECK-NEXT:   %3 = extractvalue [4 x double*] %0, 2
; CHECK-NEXT:   %4 = extractvalue [4 x double*] %0, 3
; CHECK-NEXT:   %5 = load double, double* %1, align 8
; CHECK-NEXT:   %6 = load double, double* %2, align 8
; CHECK-NEXT:   %7 = load double, double* %3, align 8
; CHECK-NEXT:   %8 = load double, double* %4, align 8
; CHECK-NEXT:   %9 = fadd fast double %5, 1.000000e+00
; CHECK-NEXT:   %10 = fadd fast double %6, 1.000000e+00
; CHECK-NEXT:   %11 = fadd fast double %7, 1.000000e+00
; CHECK-NEXT:   %12 = fadd fast double %8, 1.000000e+00
; CHECK-NEXT:   store double %9, double* %1, align 8
; CHECK-NEXT:   store double %10, double* %2, align 8
; CHECK-NEXT:   store double %11, double* %3, align 8
; CHECK-NEXT:   store double %12, double* %4, align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }