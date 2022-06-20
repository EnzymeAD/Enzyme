; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -early-cse  -enzyme-vectorize-at-leaf-nodes -S | FileCheck %s

; Function Attrs: nounwind
declare <3 x double> @__enzyme_fwddiff(double (double)*, ...)

define double @square(double %x) {
entry:
  %mul = fmul fast double %x, %x
  ret double %mul
}

define <3 x double> @dsquare(double %x) {
entry:
  %0 = tail call <3 x double> (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @square, metadata !"enzyme_width", i64 3, double %x, <3 x double> <double 1.0, double 10.0, double 100.0>)
  ret <3 x double> %0
}


; CHECK: define internal <3 x double> @fwddiffe3square(double %x, <3 x double> %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %.splatinsert = insertelement <3 x double> poison, double %x, i32 0
; CHECK-NEXT:   %.splat = shufflevector <3 x double> %.splatinsert, <3 x double> poison, <3 x i32> zeroinitializer
; CHECK-NEXT:   %0 = fmul fast <3 x double> %"x'", %.splat
; CHECK-NEXT:   %1 = fadd fast <3 x double> %0, %0
; CHECK-NEXT:   ret <3 x double> %1
; CHECK-NEXT: }