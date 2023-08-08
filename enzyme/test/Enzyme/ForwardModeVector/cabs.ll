; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -early-cse -instsimplify -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,%simplifycfg,early-cse,instsimplify)" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %call = call double @cabs(double %x, double %y)
  ret double %call
}

define [3 x double] @test_derivative(double %x, double %y) {
entry:
  %0 = tail call [3 x double] (double (double, double)*, ...) @__enzyme_fwddiff(double (double, double)* nonnull @tester, metadata !"enzyme_width", i64 3, double %x, double 1.0, double 1.3, double 2.0, double %y, double 1.0, double 0.0, double 2.0)
  ret [3 x double] %0
}

declare double @cabs(double, double)

; Function Attrs: nounwind
declare [3 x double] @__enzyme_fwddiff(double (double, double)*, ...)


; CHECK: define internal [3 x double] @fwddiffe3tester(double %x, [3 x double] %"x'", double %y, [3 x double] %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call fast double @cabs(double %x, double %y)
; CHECK-NEXT:   %1 = fdiv fast double %x, %0
; CHECK-NEXT:   %2 = extractvalue [3 x double] %"x'", 0
; CHECK-NEXT:   %3 = fmul fast double %2, %1
; CHECK-NEXT:   %4 = extractvalue [3 x double] %"x'", 1
; CHECK-NEXT:   %5 = fmul fast double %4, %1
; CHECK-NEXT:   %6 = extractvalue [3 x double] %"x'", 2
; CHECK-NEXT:   %7 = fmul fast double %6, %1
; CHECK-NEXT:   %8 = fdiv fast double %y, %0
; CHECK-NEXT:   %9 = extractvalue [3 x double] %"y'", 0
; CHECK-NEXT:   %10 = fmul fast double %9, %8
; CHECK-NEXT:   %11 = extractvalue [3 x double] %"y'", 1
; CHECK-NEXT:   %12 = fmul fast double %11, %8
; CHECK-NEXT:   %13 = extractvalue [3 x double] %"y'", 2
; CHECK-NEXT:   %14 = fmul fast double %13, %8
; CHECK-NEXT:   %15 = fadd fast double %3, %10
; CHECK-NEXT:   %16 = insertvalue [3 x double] undef, double %15, 0
; CHECK-NEXT:   %17 = fadd fast double %5, %12
; CHECK-NEXT:   %18 = insertvalue [3 x double] %16, double %17, 1
; CHECK-NEXT:   %19 = fadd fast double %7, %14
; CHECK-NEXT:   %20 = insertvalue [3 x double] %18, double %19, 2
; CHECK-NEXT:   ret [3 x double] %20
; CHECK-NEXT: }