; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -enzyme-vectorize-at-leaf-nodes -mem2reg -sroa -instsimplify -simplifycfg -adce -S | FileCheck %s

define double @tester(double %x, double %y) {
entry:
  %call = call double @hypot(double %x, double %y)
  ret double %call
}

define double @tester2(double %x) {
entry:
  %call = call double @hypot(double %x, double 2.000000e+00)
  ret double %call
}


define <3 x double> @test_derivative(double %x, double %y) {
entry:
  %0 = tail call <3 x double> (...) @__enzyme_fwddiff(double (double, double)* nonnull @tester, metadata !"enzyme_width", i64 3,  double %x, <3 x double> <double 1.0, double 2.0, double 3.0>, double %y, <3 x double> <double 1.0, double 2.0, double 3.0>)
  %1 = tail call <3 x double> (...) @__enzyme_fwddiff(double (double)* nonnull @tester2, metadata !"enzyme_width", i64 3, double %x, <3 x double> <double 1.0, double 2.0, double 3.0>)
  ret <3 x double> %0
}

declare double @hypot(double, double)

declare <3 x double> @__enzyme_fwddiff(...)


; CHECK: define internal <3 x double> @fwddiffe3tester(double %x, <3 x double> %"x'", double %y, <3 x double> %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractelement <3 x double> %"x'", i32 0
; CHECK-NEXT:   %1 = fmul fast double %0, %x
; CHECK-NEXT:   %2 = extractelement <3 x double> %"x'", i32 1
; CHECK-NEXT:   %3 = fmul fast double %2, %x
; CHECK-NEXT:   %4 = extractelement <3 x double> %"x'", i32 2
; CHECK-NEXT:   %5 = fmul fast double %4, %x
; CHECK-NEXT:   %6 = call fast double @hypot(double %x, double %y) #1
; CHECK-NEXT:   %7 = fdiv fast double %1, %6
; CHECK-NEXT:   %8 = fdiv fast double %3, %6
; CHECK-NEXT:   %9 = fdiv fast double %5, %6
; CHECK-NEXT:   %10 = extractelement <3 x double> %"y'", i32 0
; CHECK-NEXT:   %11 = fmul fast double %10, %y
; CHECK-NEXT:   %12 = extractelement <3 x double> %"y'", i32 1
; CHECK-NEXT:   %13 = fmul fast double %12, %y
; CHECK-NEXT:   %14 = extractelement <3 x double> %"y'", i32 2
; CHECK-NEXT:   %15 = fmul fast double %14, %y
; CHECK-NEXT:   %16 = call fast double @hypot(double %x, double %y) #1
; CHECK-NEXT:   %17 = fdiv fast double %11, %16
; CHECK-NEXT:   %18 = fdiv fast double %13, %16
; CHECK-NEXT:   %19 = fdiv fast double %15, %16
; CHECK-NEXT:   %20 = fadd fast double %7, %17
; CHECK-NEXT:   %21 = insertelement <3 x double> undef, double %20, i32 0
; CHECK-NEXT:   %22 = fadd fast double %8, %18
; CHECK-NEXT:   %23 = insertelement <3 x double> %21, double %22, i32 1
; CHECK-NEXT:   %24 = fadd fast double %9, %19
; CHECK-NEXT:   %25 = insertelement <3 x double> %23, double %24, i32 2
; CHECK-NEXT:   ret <3 x double> %25
; CHECK-NEXT: }

; CHECK: define internal <3 x double> @fwddiffe3tester2(double %x, <3 x double> %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractelement <3 x double> %"x'", i32 0
; CHECK-NEXT:   %1 = fmul fast double %0, %x
; CHECK-NEXT:   %2 = extractelement <3 x double> %"x'", i32 1
; CHECK-NEXT:   %3 = fmul fast double %2, %x
; CHECK-NEXT:   %4 = extractelement <3 x double> %"x'", i32 2
; CHECK-NEXT:   %5 = fmul fast double %4, %x
; CHECK-NEXT:   %6 = call fast double @hypot(double %x, double 2.000000e+00) #1
; CHECK-NEXT:   %7 = fdiv fast double %1, %6
; CHECK-NEXT:   %8 = insertelement <3 x double> undef, double %7, i32 0
; CHECK-NEXT:   %9 = fdiv fast double %3, %6
; CHECK-NEXT:   %10 = insertelement <3 x double> %8, double %9, i32 1
; CHECK-NEXT:   %11 = fdiv fast double %5, %6
; CHECK-NEXT:   %12 = insertelement <3 x double> %10, double %11, i32 2
; CHECK-NEXT:   ret <3 x double> %12
; CHECK-NEXT: }