; RUN: %opt < %s %loadEnzyme -enzyme -sroa -simplifycfg -instcombine -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define [2 x double] @meta(double %x) {
entry:
  %0 = tail call fast [2 x double] @__fd_sincos_1(double %x)
  ret [2 x double] %0
}

define double @tester(double %x) {
entry:
  %0 = tail call fast [2 x double] @meta(double %x)
  %e = extractvalue [2 x double] %0, 0
  ret double %e
}

define [3 x double] @test_derivative(double %x) {
entry:
  %0 = tail call [3 x double] (...) @__enzyme_autodiff(double (double)* nonnull @tester, metadata !"enzyme_width", i64 3, double %x)
  ret [3 x double] %0
}

declare [2 x double] @__fd_sincos_1(double)

; Function Attrs: nounwind
declare [3 x double] @__enzyme_autodiff(...)

; CHECK: define internal { [3 x double] } @diffe3meta(double %x, [3 x [2 x double]] %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %differeturn.fca.0.0.extract = extractvalue [3 x [2 x double]] %differeturn, 0, 0
; CHECK-NEXT:   %differeturn.fca.0.1.extract = extractvalue [3 x [2 x double]] %differeturn, 0, 1
; CHECK-NEXT:   %differeturn.fca.1.0.extract = extractvalue [3 x [2 x double]] %differeturn, 1, 0
; CHECK-NEXT:   %differeturn.fca.1.1.extract = extractvalue [3 x [2 x double]] %differeturn, 1, 1
; CHECK-NEXT:   %differeturn.fca.2.0.extract = extractvalue [3 x [2 x double]] %differeturn, 2, 0
; CHECK-NEXT:   %differeturn.fca.2.1.extract = extractvalue [3 x [2 x double]] %differeturn, 2, 1
; CHECK-NEXT:   %0 = call fast [2 x double] @__fd_sincos_1(double %x)
; CHECK-NEXT:   %1 = extractvalue [2 x double] %0, 1
; CHECK-NEXT:   %2 = fmul fast double %1, %differeturn.fca.0.0.extract
; CHECK-NEXT:   %3 = fmul fast double %1, %differeturn.fca.1.0.extract
; CHECK-NEXT:   %4 = fmul fast double %1, %differeturn.fca.2.0.extract
; CHECK-NEXT:   %5 = extractvalue [2 x double] %0, 0
; CHECK-NEXT:   %6 = fmul fast double %5, %differeturn.fca.0.1.extract
; CHECK-NEXT:   %7 = fsub fast double %2, %6
; CHECK-NEXT:   %8 = fmul fast double %5, %differeturn.fca.1.1.extract
; CHECK-NEXT:   %9 = fsub fast double %3, %8
; CHECK-NEXT:   %10 = fmul fast double %5, %differeturn.fca.2.1.extract
; CHECK-NEXT:   %11 = fsub fast double %4, %10
; CHECK-NEXT:   %.fca.0.insert6 = insertvalue [3 x double] undef, double %7, 0
; CHECK-NEXT:   %.fca.1.insert9 = insertvalue [3 x double] %.fca.0.insert6, double %9, 1
; CHECK-NEXT:   %.fca.2.insert12 = insertvalue [3 x double] %.fca.1.insert9, double %11, 2
; CHECK-NEXT:   %12 = insertvalue { [3 x double] } undef, [3 x double] %.fca.2.insert12, 0
; CHECK-NEXT:   ret { [3 x double] } %12
; CHECK-NEXT: }
