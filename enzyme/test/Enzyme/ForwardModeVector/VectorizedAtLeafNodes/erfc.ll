; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -enzyme-vectorize-at-leaf-nodes -mem2reg -sroa -instsimplify -simplifycfg -adce -S | FileCheck %s

; Function Attrs: nounwind
declare <2 x double> @__enzyme_fwddiff(double (double)*, ...)

declare double @erfc(double)

define double @tester(double %x) {
entry:
  %call = call double @erfc(double %x)
  ret double %call
}

define <2 x double> @test_derivative(double %x) {
entry:
  %0 = tail call <2 x double> (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @tester, metadata !"enzyme_width", i64 2, double %x, <2 x double> <double 1.0, double 1.5>)
  ret <2 x double> %0
}


; CHECK: define internal <2 x double> @fwddiffe2tester(double %x, <2 x double> %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = fmul fast double %x, %x
; CHECK-NEXT:   %1 = fneg fast double %0
; CHECK-NEXT:   %2 = call fast double @llvm.exp.f64(double %1)
; CHECK-NEXT:   %3 = fmul fast double %2, 0xBFF20DD750429B6D
; CHECK-NEXT:   %.splatinsert = insertelement <2 x double> poison, double %3, i32 0
; CHECK-NEXT:   %.splat = shufflevector <2 x double> %.splatinsert, <2 x double> poison, <2 x i32> zeroinitializer
; CHECK-NEXT:   %4 = fmul fast <2 x double> %.splat, %"x'"
; CHECK-NEXT:   ret <2 x double> %4
; CHECK-NEXT: }