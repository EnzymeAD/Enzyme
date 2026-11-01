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
; CHECK-NEXT:   %5 = fmul double %1, %2
; CHECK-NEXT:   %6 = fmul double %1, %3
; CHECK-NEXT:   %7 = fmul double %1, %4
; CHECK-NEXT:   %8 = extractvalue [2 x double] %0, 0
; CHECK-NEXT:   %9 = extractvalue [3 x [2 x double]] %differeturn, 0, 1
; CHECK-NEXT:   %10 = extractvalue [3 x [2 x double]] %differeturn, 1, 1
; CHECK-NEXT:   %11 = extractvalue [3 x [2 x double]] %differeturn, 2, 1
; CHECK-NEXT:   %12 = fmul double %8, %9
; CHECK-NEXT:   %13 = fmul double %8, %10
; CHECK-NEXT:   %14 = fmul double %8, %11
; CHECK-NEXT:   %15 = {{(fsub double \-0.000000e\+00,|fneg double)}} %12
; CHECK-NEXT:   %16 = {{(fsub double \-0.000000e\+00,|fneg double)}} %13
; CHECK-NEXT:   %17 = {{(fsub double \-0.000000e\+00,|fneg double)}} %14
; CHECK-NEXT:   %18 = fadd double %5, %15
; CHECK-NEXT:   %19 = fadd double %6, %16
; CHECK-NEXT:   %20 = fadd double %7, %17
; CHECK-NEXT:   %21 = fadd double 0.000000e+00, %18
; CHECK-NEXT:   %22 = fadd double 0.000000e+00, %19
; CHECK-NEXT:   %23 = fadd double 0.000000e+00, %20
; CHECK-NEXT:   %.fca.0.insert6 = insertvalue [3 x double] {{(undef|poison)}}, double %21, 0
; CHECK-NEXT:   %.fca.1.insert9 = insertvalue [3 x double] %.fca.0.insert6, double %22, 1
; CHECK-NEXT:   %.fca.2.insert12 = insertvalue [3 x double] %.fca.1.insert9, double %23, 2
; CHECK-NEXT:   %24 = insertvalue { [3 x double] } {{(undef|poison)}}, [3 x double] %.fca.2.insert12, 0
; CHECK-NEXT:   ret { [3 x double] } %24
; CHECK-NEXT: }
