; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -early-cse -adce -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(early-cse),function(adce)" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define [2 x double] @tester(double %x) {
entry:
  %0 = tail call [2 x double] @__fd_sincos_1(double %x)
  ret [2 x double] %0
}

define [2 x [2 x double]] @test_derivative(double %x) {
entry:
  %0 = tail call [2 x [2 x double]] (...) @__enzyme_fwddiff([2 x double] (double)* nonnull @tester, metadata !"enzyme_width", i64 2, double %x, double 1.0, double 2.0)
  ret [2 x [2 x double]] %0
}

declare [2 x double] @__fd_sincos_1(double)

; Function Attrs: nounwind
declare [2 x [2 x double]] @__enzyme_fwddiff(...)

; CHECK: define internal [2 x [2 x double]] @fwddiffe2tester(double %x, [2 x double] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call {{(fast )?}}[2 x double] @__fd_sincos_1(double %x)
; CHECK-NEXT:   %1 = extractvalue [2 x double] %0, 1
; CHECK-NEXT:   %2 = extractvalue [2 x double] %"x'", 0
; CHECK-NEXT:   %3 = fmul fast double %1, %2
; CHECK-NEXT:   %4 = extractvalue [2 x double] %"x'", 1
; CHECK-NEXT:   %5 = fmul fast double %1, %4
; CHECK-NEXT:   %6 = insertvalue [2 x [2 x double]] zeroinitializer, double %3, 0, 0
; CHECK-NEXT:   %7 = insertvalue [2 x [2 x double]] %6, double %5, 1, 0
; CHECK-NEXT:   %8 = extractvalue [2 x double] %0, 0
; CHECK-NEXT:   %9 = fmul fast double %8, %2
; CHECK-NEXT:   %10 = fmul fast double %8, %4
; CHECK-NEXT:   %11 = {{(fneg fast double)|(fsub fast double \-0.000000e\+00,)}} %9
; CHECK-NEXT:   %12 = {{(fneg fast double)|(fsub fast double \-0.000000e\+00,)}} %10
; CHECK-NEXT:   %13 = insertvalue [2 x [2 x double]] %7, double %11, 0, 1
; CHECK-NEXT:   %14 = insertvalue [2 x [2 x double]] %13, double %12, 1, 1
; CHECK-NEXT:   ret [2 x [2 x double]] %14
; CHECK-NEXT: }
