; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -sroa -simplifycfg -instsimplify -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(sroa,%simplifycfg,instsimplify)" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define [2 x double] @meta(double %x) {
entry:
  %0 = tail call [2 x double] @__fd_sincos_1(double %x)
  ret [2 x double] %0
}

define double @tester(double %x) {
entry:
  %0 = tail call [2 x double] @meta(double %x)
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
; CHECK-NEXT:   %0 = call {{(fast )?}}[2 x double] @__fd_sincos_1(double %x)
; CHECK-NEXT:   %1 = extractvalue [2 x double] %0, 1
; CHECK-NEXT:   %2 = extractvalue [3 x [2 x double]] %differeturn, 0, 0
; CHECK-NEXT:   %3 = extractvalue [3 x [2 x double]] %differeturn, 1, 0
; CHECK-NEXT:   %4 = extractvalue [3 x [2 x double]] %differeturn, 2, 0
; CHECK-NEXT:   %5 = fmul fast double %1, %2
; CHECK-NEXT:   %6 = fmul fast double %1, %3
; CHECK-NEXT:   %7 = fmul fast double %1, %4
; CHECK-NEXT:   %8 = extractvalue [2 x double] %0, 0
; CHECK-NEXT:   %9 = extractvalue [3 x [2 x double]] %differeturn, 0, 1
; CHECK-NEXT:   %10 = extractvalue [3 x [2 x double]] %differeturn, 1, 1
; CHECK-NEXT:   %11 = extractvalue [3 x [2 x double]] %differeturn, 2, 1
; CHECK-NEXT:   %12 = fmul fast double %8, %9
; CHECK-NEXT:   %13 = fmul fast double %8, %10
; CHECK-NEXT:   %14 = fmul fast double %8, %11
; CHECK-NEXT:   %15 = {{(fsub fast double \-0.000000e\+00,|fneg fast double)}} %12
; CHECK-NEXT:   %16 = {{(fsub fast double \-0.000000e\+00,|fneg fast double)}} %13
; CHECK-NEXT:   %17 = {{(fsub fast double \-0.000000e\+00,|fneg fast double)}} %14
; CHECK-NEXT:   %18 = fadd fast double %5, %15
; CHECK-NEXT:   %19 = fadd fast double %6, %16
; CHECK-NEXT:   %20 = fadd fast double %7, %17
; CHECK-NEXT:   %.fca.0.insert6 = insertvalue [3 x double] {{(undef|poison)}}, double %18, 0
; CHECK-NEXT:   %.fca.1.insert9 = insertvalue [3 x double] %.fca.0.insert6, double %19, 1
; CHECK-NEXT:   %.fca.2.insert12 = insertvalue [3 x double] %.fca.1.insert9, double %20, 2
; CHECK-NEXT:   %21 = insertvalue { [3 x double] } {{(undef|poison)}}, [3 x double] %.fca.2.insert12, 0
; CHECK-NEXT:   ret { [3 x double] } %21
; CHECK-NEXT: }
