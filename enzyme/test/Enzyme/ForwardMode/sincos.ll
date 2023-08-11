; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -adce -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(adce)" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define [2 x double] @tester(double %x) {
entry:
  %0 = tail call [2 x double] @__fd_sincos_1(double %x)
  ret [2 x double] %0
}

define [2 x double] @test_derivative(double %x) {
entry:
  %0 = tail call [2 x double] (...) @__enzyme_fwddiff([2 x double] (double)* nonnull @tester, double %x, double 1.0)
  ret [2 x double] %0
}

declare [2 x double] @__fd_sincos_1(double)

; Function Attrs: nounwind
declare [2 x double] @__enzyme_fwddiff(...)

; CHECK: define internal [2 x double] @fwddiffetester(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[i0:.+]] = call {{(fast )?}}[2 x double] @__fd_sincos_1(double %x)
; CHECK-NEXT:   %[[i1:.+]] = extractvalue [2 x double] %[[i0]], 1
; CHECK-NEXT:   %[[i2:.+]] = fmul fast double %[[i1]], %"x'"
; CHECK-NEXT:   %[[i3:.+]] = insertvalue [2 x double] undef, double %[[i2]], 0
; CHECK-NEXT:   %[[i4:.+]] = extractvalue [2 x double] %[[i0]], 0
; CHECK-NEXT:   %[[i5:.+]] = fmul fast double %[[i4]], %"x'"
; CHECK-NEXT:   %[[i6:.+]] = {{(fneg fast double)|(fsub fast double \-0.000000e\+00,)}} %[[i5]]
; CHECK-NEXT:   %[[i8:.+]] = insertvalue [2 x double] %[[i3]], double %[[i6]], 1
; CHECK-NEXT:   ret [2 x double] %[[i8]]
; CHECK-NEXT: }
