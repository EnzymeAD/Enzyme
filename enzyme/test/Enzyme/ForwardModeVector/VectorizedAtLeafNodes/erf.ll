; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -enzyme-vectorize-at-leaf-nodes -mem2reg -sroa -instsimplify -simplifycfg -adce -S | FileCheck %s

; Function Attrs: nounwind
declare <3 x double> @__enzyme_fwddiff(double (double)*, ...)

declare double @erf(double)

define double @tester(double %x) {
entry:
  %call = call double @erf(double %x)
  ret double %call
}

define <3 x double> @test_derivative(double %x) {
entry:
  %0 = tail call <3 x double> (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @tester, metadata !"enzyme_width", i64 3, double %x, <3 x double> <double 1.0, double 2.0, double 3.0>)
  ret <3 x double> %0
}


; CHECK: define internal <3 x double> @fwddiffe3tester(double %x, <3 x double> %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = fmul fast double %x, %x
; CHECK-NEXT:   %1 = fneg fast double %0
; CHECK-NEXT:   %2 = call fast double @llvm.exp.f64(double %1)
; CHECK-NEXT:   %3 = fmul fast double %2, 0x3FF20DD750429B6D
; CHECK-NEXT:   %.splatinsert = insertelement <3 x double> poison, double %3, i32 0
; CHECK-NEXT:   %.splat = shufflevector <3 x double> %.splatinsert, <3 x double> poison, <3 x i32> zeroinitializer
; CHECK-NEXT:   %4 = fmul fast <3 x double> %.splat, %"x'"
; CHECK-NEXT:   ret <3 x double> %4
; CHECK-NEXT: }