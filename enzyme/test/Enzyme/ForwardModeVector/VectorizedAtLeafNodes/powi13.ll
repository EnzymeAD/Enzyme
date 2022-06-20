; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -enzyme-vectorize-at-leaf-nodes -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

; Function Attrs: nounwind
declare <3 x double> @__enzyme_fwddiff(double (double, i32)*, ...)

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x, i32 %y) {
entry:
  %0 = tail call fast double @llvm.powi.f64.i32(double %x, i32 %y)
  ret double %0
}

define <3 x double> @test_derivative(double %x, i32 %y) {
entry:
  %0 = tail call <3 x double> (double (double, i32)*, ...) @__enzyme_fwddiff(double (double, i32)* nonnull @tester, metadata !"enzyme_width", i64 3, double %x, <3 x double> <double 1.0, double 2.0, double 3.0>, i32 %y)
  ret <3 x double> %0
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.powi.f64.i32(double, i32)


; CHECK: define internal <3 x double> @fwddiffe3tester(double %x, <3 x double> %"x'", i32 %y)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = sub i32 %y, 1
; CHECK-NEXT:   %1 = call fast double @llvm.powi.f64(double %x, i32 %0)
; CHECK-NEXT:   %2 = sitofp i32 %y to double
; CHECK-NEXT:   %3 = icmp eq i32 0, %y
; CHECK-NEXT:   %.splatinsert = insertelement <3 x double> poison, double %1, i32 0
; CHECK-NEXT:   %.splat = shufflevector <3 x double> %.splatinsert, <3 x double> poison, <3 x i32> zeroinitializer
; CHECK-NEXT:   %.splatinsert1 = insertelement <3 x double> poison, double %2, i32 0
; CHECK-NEXT:   %.splat2 = shufflevector <3 x double> %.splatinsert1, <3 x double> poison, <3 x i32> zeroinitializer
; CHECK-NEXT:   %4 = fmul fast <3 x double> %"x'", %.splat
; CHECK-NEXT:   %5 = fmul fast <3 x double> %4, %.splat2
; CHECK-NEXT:   %6 = select fast i1 %3, <3 x double> zeroinitializer, <3 x double> %5
; CHECK-NEXT:   ret <3 x double> %6
; CHECK-NEXT: }