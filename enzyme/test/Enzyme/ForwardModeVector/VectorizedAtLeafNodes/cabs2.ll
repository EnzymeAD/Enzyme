; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -enzyme-vectorize-at-leaf-nodes -mem2reg -sroa -instsimplify -simplifycfg -adce -S | FileCheck %s

; Function Attrs: nounwind readnone willreturn
declare double @cabs([2 x double]) #7

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %agg0 = insertvalue [2 x double] undef, double %x, 0
  %agg1 = insertvalue [2 x double] %agg0, double %y, 1
  %call = call double @cabs([2 x double] %agg1)
  ret double %call
}

define <3 x double> @test_derivative(double %x, double %y) {
entry:
  %0 = tail call <3 x double> (double (double, double)*, ...) @__enzyme_fwddiff(double (double, double)* nonnull @tester, metadata !"enzyme_width", i64 3, double %x, <3 x double> <double 1.0, double 1.3, double 2.0>, double %y, <3 x double> <double 1.0, double 0.0, double 2.0>)
  ret <3 x double> %0
}

; Function Attrs: nounwind
declare <3 x double> @__enzyme_fwddiff(double (double, double)*, ...)


; CHECK: define internal <3 x double> @fwddiffe3tester(double %x, <3 x double> %"x'", double %y, <3 x double> %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %agg0 = insertvalue [2 x double] undef, double %x, 0
; CHECK-NEXT:   %agg1 = insertvalue [2 x double] %agg0, double %y, 1
; CHECK-NEXT:   %0 = call fast double @cabs([2 x double] %agg1)
; CHECK-NEXT:   %.splatinsert = insertelement <3 x double> poison, double %x, i32 0
; CHECK-NEXT:   %.splat = shufflevector <3 x double> %.splatinsert, <3 x double> poison, <3 x i32> zeroinitializer
; CHECK-NEXT:   %.splatinsert1 = insertelement <3 x double> poison, double %y, i32 0
; CHECK-NEXT:   %.splat2 = shufflevector <3 x double> %.splatinsert1, <3 x double> poison, <3 x i32> zeroinitializer
; CHECK-NEXT:   %.splatinsert3 = insertelement <3 x double> poison, double %0, i32 0
; CHECK-NEXT:   %.splat4 = shufflevector <3 x double> %.splatinsert3, <3 x double> poison, <3 x i32> zeroinitializer
; CHECK-NEXT:   %1 = fdiv fast <3 x double> %"x'", %.splat4
; CHECK-NEXT:   %2 = fmul fast <3 x double> %.splat, %1
; CHECK-NEXT:   %3 = fdiv fast <3 x double> %"y'", %.splat4
; CHECK-NEXT:   %4 = fmul fast <3 x double> %.splat2, %3
; CHECK-NEXT:   %5 = fadd fast <3 x double> %2, %4
; CHECK-NEXT:   ret <3 x double> %5
; CHECK-NEXT: }