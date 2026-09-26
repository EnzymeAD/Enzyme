; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -sroa -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,sroa,instsimplify,%simplifycfg)" -S | FileCheck %s

; The derivative multiplies by pi rounded to double, 0x400921FB54442D18.

; Function Attrs: nounwind readnone uwtable
define [2 x double] @meta(double %x) {
entry:
  %0 = tail call [2 x double] @sincospi(double %x)
  ret [2 x double] %0
}

define double @tester(double %x) {
entry:
  %0 = tail call [2 x double] @meta(double %x)
  %e = extractvalue [2 x double] %0, 0
  ret double %e
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (...) @__enzyme_autodiff(double (double)* nonnull @tester, double %x)
  ret double %0
}

declare [2 x double] @sincospi(double)

; Function Attrs: nounwind
declare double @__enzyme_autodiff(...)

; CHECK: define internal { double } @diffemeta(double %x, [2 x double] %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call {{(fast )?}}[2 x double] @sincospi(double %x)
; CHECK-NEXT:   %1 = extractvalue [2 x double] %0, 1
; CHECK-NEXT:   %2 = extractvalue [2 x double] %differeturn, 0
; CHECK-NEXT:   %3 = fmul fast double %1, %2
; CHECK-NEXT:   %4 = fmul fast double 0x400921FB54442D18, %3
; CHECK-NEXT:   %5 = extractvalue [2 x double] %0, 0
; CHECK-NEXT:   %6 = extractvalue [2 x double] %differeturn, 1
; CHECK-NEXT:   %7 = fmul fast double %5, %6
; CHECK-NEXT:   %8 = {{(fsub fast double -0.000000e\+00,|fneg fast double)}} %7
; CHECK-NEXT:   %9 = fmul fast double 0x400921FB54442D18, %8
; CHECK-NEXT:   %10 = fadd fast double %4, %9
; CHECK-NEXT:   %11 = insertvalue { double } undef, double %10, 0
; CHECK-NEXT:   ret { double } %11
; CHECK-NEXT: }
