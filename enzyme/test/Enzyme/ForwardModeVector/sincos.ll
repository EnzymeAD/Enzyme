; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -early-cse -adce -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(early-cse),function(adce)" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define [2 x double] @tester(double %x) {
entry:
  %0 = tail call [2 x double] @__fd_sincos_1(double %x)
  ret [2 x double] %0
}

define [3 x [2 x double]] @test_derivative(double %x) {
entry:
  %0 = tail call [3 x [2 x double]] (...) @__enzyme_fwddiff([2 x double] (double)* nonnull @tester, metadata !"enzyme_width", i64 3, double %x, double 1.0, double 2.0, double 3.0)
  ret [3 x [2 x double]] %0
}

declare [2 x double] @__fd_sincos_1(double)

; Function Attrs: nounwind
declare [3 x [2 x double]] @__enzyme_fwddiff(...)

; CHECK: define internal [3 x [2 x double]] @fwddiffe3tester(double %x, [3 x double] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[i0:.+]] = call {{(fast )?}}[2 x double] @__fd_sincos_1(double %x)
; CHECK-NEXT:   %[[i1:.+]] = extractvalue [2 x double] %0, 1
; CHECK-NEXT:   %[[i2:.+]] = extractvalue [3 x double] %"x'", 0
; CHECK-NEXT:   %[[i3:.+]] = fmul fast double %[[i1]], %[[i2]]
; CHECK-NEXT:   %[[i4:.+]] = extractvalue [3 x double] %"x'", 1
; CHECK-NEXT:   %[[i5:.+]] = fmul fast double %[[i1]], %[[i4]]
; CHECK-NEXT:   %[[v4:.+]] = extractvalue [3 x double] %"x'", 2
; CHECK-NEXT:   %[[v5:.+]] = fmul fast double %[[i1]], %[[v4]]

; CHECK-NEXT:   %[[r00:.+]] = insertvalue [3 x [2 x double]] undef, double %[[i3]], 0, 0
; CHECK-NEXT:   %[[r10:.+]] = insertvalue [3 x [2 x double]] %[[r00]], double %[[i5]], 1, 0
; CHECK-NEXT:   %[[r20:.+]] = insertvalue [3 x [2 x double]] %[[r10]], double %[[v5]], 2, 0

; CHECK-NEXT:   %[[i8:.+]] = extractvalue [2 x double] %[[i0]], 0
; CHECK-NEXT:   %[[i9:.+]] = fmul fast double %[[i8]], %[[i2]]
; CHECK-NEXT:   %[[i10:.+]] = fmul fast double %[[i8]], %[[i4]]
; CHECK-NEXT:   %[[v10:.+]] = fmul fast double %[[i8]], %[[v4]]
; CHECK-NEXT:   %[[i11:.+]] = {{(fneg fast double)|(fsub fast double \-0.000000e\+00,)}} %[[i9]]
; CHECK-NEXT:   %[[i12:.+]] = {{(fneg fast double)|(fsub fast double \-0.000000e\+00,)}} %[[i10]]
; CHECK-NEXT:   %[[v12:.+]] = {{(fneg fast double)|(fsub fast double \-0.000000e\+00,)}} %[[v10]]

; CHECK-NEXT:   %[[r01:.+]] = insertvalue [3 x [2 x double]] %[[r20]], double %[[i11]], 0, 1
; CHECK-NEXT:   %[[r11:.+]] = insertvalue [3 x [2 x double]] %[[r01]], double %[[i12]], 1, 1
; CHECK-NEXT:   %[[r21:.+]] = insertvalue [3 x [2 x double]] %[[r11]], double %[[v12]], 2, 1

; CHECK-NEXT:   ret [3 x [2 x double]] %[[r21]]
; CHECK-NEXT: }
