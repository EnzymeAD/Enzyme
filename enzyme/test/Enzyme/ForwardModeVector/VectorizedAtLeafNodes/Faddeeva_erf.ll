; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -enzyme-vectorize-at-leaf-nodes -mem2reg -sroa -instsimplify -simplifycfg -adce -S | FileCheck %s

declare { <3 x double>, <3 x double> } @__enzyme_fwddiff({ double, double } ({ double, double })*, ...)

declare { double, double } @Faddeeva_erf({ double, double }, double)

define { double, double } @tester({ double, double } %in) {
entry:
  %call = call { double, double } @Faddeeva_erf({ double, double } %in, double 0.000000e+00)
  ret { double, double } %call
}

define { <3 x double>, <3 x double> } @test_derivative({ double, double } %x) {
entry:
  %0 = tail call { <3 x double>, <3 x double> } ({ double, double } ({ double, double })*, ...) @__enzyme_fwddiff({ double, double } ({ double, double })*  @tester,  metadata !"enzyme_width", i64 3, { double, double } %x, { <3 x double>, <3 x double> } { <3 x double> <double 1.0, double 0.0, double 2.0>, <3 x double> <double 0.0, double 1.0, double 2.5>})
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
; CHECK-NEXT:   %7 = fneg fast double %4
; CHECK-NEXT:   %8 = fneg fast double %6
; CHECK-NEXT:   %9 = call fast double @llvm.exp.f64(double %7)
; CHECK-NEXT:   %10 = call fast double @llvm.cos.f64(double %8)
; CHECK-NEXT:   %11 = fmul fast double %9, %10
; CHECK-NEXT:   %12 = call fast double @llvm.sin.f64(double %8)
; CHECK-NEXT:   %13 = fmul fast double %9, %12
; CHECK-NEXT:   %14 = fmul fast double %11, 0x3FF20DD750429B6D
; CHECK-NEXT:   %15 = fmul fast double %13, 0x3FF20DD750429B6D
; CHECK-NEXT:   %.splatinsert = insertelement <3 x double> poison, double %14, i32 0
; CHECK-NEXT:   %.splat = shufflevector <3 x double> %.splatinsert, <3 x double> poison, <3 x i32> zeroinitializer
; CHECK-NEXT:   %16 = insertvalue { <3 x double>, <3 x double> } undef, <3 x double> %.splat, 0
; CHECK-NEXT:   %.splatinsert1 = insertelement <3 x double> poison, double %15, i32 0
; CHECK-NEXT:   %.splat2 = shufflevector <3 x double> %.splatinsert1, <3 x double> poison, <3 x i32> zeroinitializer
; CHECK-NEXT:   %17 = insertvalue { <3 x double>, <3 x double> } %16, <3 x double> %.splat2, 1
; CHECK-NEXT:   %.splatinsert3 = insertelement <3 x double> poison, double %14, i32 0
; CHECK-NEXT:   %.splat4 = shufflevector <3 x double> %.splatinsert3, <3 x double> poison, <3 x i32> zeroinitializer
; CHECK-NEXT:   %.splatinsert5 = insertelement <3 x double> poison, double %15, i32 0
; CHECK-NEXT:   %.splat6 = shufflevector <3 x double> %.splatinsert5, <3 x double> poison, <3 x i32> zeroinitializer
; CHECK-NEXT:   %18 = extractvalue { <3 x double>, <3 x double> } %"in'", 0
; CHECK-NEXT:   %19 = extractvalue { <3 x double>, <3 x double> } %"in'", 1
; CHECK-NEXT:   %20 = fmul fast <3 x double> %.splat4, %18
; CHECK-NEXT:   %21 = fmul fast <3 x double> %.splat6, %19
; CHECK-NEXT:   %22 = fsub fast <3 x double> %20, %21
; CHECK-NEXT:   %23 = insertvalue { <3 x double>, <3 x double> } %17, <3 x double> %22, 0
; CHECK-NEXT:   %24 = fmul fast <3 x double> %.splat6, %18
; CHECK-NEXT:   %25 = fmul fast <3 x double> %.splat4, %19
; CHECK-NEXT:   %26 = fadd fast <3 x double> %24, %25
; CHECK-NEXT:   %27 = insertvalue { <3 x double>, <3 x double> } %23, <3 x double> %26, 1
; CHECK-NEXT:   ret { <3 x double>, <3 x double> } %27
; CHECK-NEXT: }