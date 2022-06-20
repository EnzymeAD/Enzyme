; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -enzyme-vectorize-at-leaf-nodes -mem2reg -sroa -instsimplify -simplifycfg -adce -S | FileCheck %s

declare { <3 x double>, <3 x double> } @__enzyme_fwddiff({ double, double } ({ double, double })*, ...)

declare { double, double } @Faddeeva_erfi({ double, double }, double)

define { double, double } @tester({ double, double } %in) {
entry:
  %call = call { double, double } @Faddeeva_erfi({ double, double } %in, double 0.000000e+00)
  ret { double, double } %call
}

define { <3 x double>, <3 x double> } @test_derivative({ double, double } %x) {
entry:
  %0 = tail call { <3 x double>, <3 x double> } ({ double, double } ({ double, double })*, ...) @__enzyme_fwddiff({ double, double } ({ double, double })* nonnull @tester,  metadata !"enzyme_width", i64 3, { double, double } %x, { <3 x double>, <3 x double> } { <3 x double> <double 1.0, double 2.0, double 3.0>, <3 x double> <double 0.0, double 1.0, double 2.0> })
  ret { <3 x double>, <3 x double> } %0
}


; CHECK: define internal { <3 x double>, <3 x double> } @fwddiffe3tester({ double, double } %in, { <3 x double>, <3 x double> } %"in'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue { double, double } %in, 0
; CHECK-NEXT:   %1 = extractvalue { double, double } %in, 1
; CHECK-NEXT:   %2 = fmul fast double %0, %0
; CHECK-NEXT:   %3 = fmul fast double %1, %1
; CHECK-NEXT:   %4 = fsub fast double %2, %3
; CHECK-NEXT:   %5 = fmul fast double %0, %1
; CHECK-NEXT:   %6 = fadd fast double %5, %5
; CHECK-NEXT:   %7 = call fast double @llvm.exp.f64(double %4)
; CHECK-NEXT:   %8 = call fast double @llvm.cos.f64(double %6)
; CHECK-NEXT:   %9 = fmul fast double %7, %8
; CHECK-NEXT:   %10 = call fast double @llvm.sin.f64(double %6)
; CHECK-NEXT:   %11 = fmul fast double %7, %10
; CHECK-NEXT:   %12 = fmul fast double %9, 0x3FF20DD750429B6D
; CHECK-NEXT:   %13 = fmul fast double %11, 0x3FF20DD750429B6D
; CHECK-NEXT:   %.splatinsert = insertelement <3 x double> poison, double %12, i32 0
; CHECK-NEXT:   %.splat = shufflevector <3 x double> %.splatinsert, <3 x double> poison, <3 x i32> zeroinitializer
; CHECK-NEXT:   %14 = insertvalue { <3 x double>, <3 x double> } undef, <3 x double> %.splat, 0
; CHECK-NEXT:   %.splatinsert1 = insertelement <3 x double> poison, double %13, i32 0
; CHECK-NEXT:   %.splat2 = shufflevector <3 x double> %.splatinsert1, <3 x double> poison, <3 x i32> zeroinitializer
; CHECK-NEXT:   %15 = insertvalue { <3 x double>, <3 x double> } %14, <3 x double> %.splat2, 1
; CHECK-NEXT:   %.splatinsert3 = insertelement <3 x double> poison, double %12, i32 0
; CHECK-NEXT:   %.splat4 = shufflevector <3 x double> %.splatinsert3, <3 x double> poison, <3 x i32> zeroinitializer
; CHECK-NEXT:   %.splatinsert5 = insertelement <3 x double> poison, double %13, i32 0
; CHECK-NEXT:   %.splat6 = shufflevector <3 x double> %.splatinsert5, <3 x double> poison, <3 x i32> zeroinitializer
; CHECK-NEXT:   %16 = extractvalue { <3 x double>, <3 x double> } %"in'", 0
; CHECK-NEXT:   %17 = extractvalue { <3 x double>, <3 x double> } %"in'", 1
; CHECK-NEXT:   %18 = fmul fast <3 x double> %.splat4, %16
; CHECK-NEXT:   %19 = fmul fast <3 x double> %.splat6, %17
; CHECK-NEXT:   %20 = fsub fast <3 x double> %18, %19
; CHECK-NEXT:   %21 = insertvalue { <3 x double>, <3 x double> } %15, <3 x double> %20, 0
; CHECK-NEXT:   %22 = fmul fast <3 x double> %.splat6, %16
; CHECK-NEXT:   %23 = fmul fast <3 x double> %.splat4, %17
; CHECK-NEXT:   %24 = fadd fast <3 x double> %22, %23
; CHECK-NEXT:   %25 = insertvalue { <3 x double>, <3 x double> } %21, <3 x double> %24, 1
; CHECK-NEXT:   ret { <3 x double>, <3 x double> } %25
; CHECK-NEXT: }