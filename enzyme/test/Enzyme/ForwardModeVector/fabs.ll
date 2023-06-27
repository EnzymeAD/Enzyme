; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -early-cse -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,early-cse,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s

%struct.Gradients = type { double, double }

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_fwddiff(double (double)*, ...)

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %0 = tail call fast double @llvm.fabs.f64(double %x)
  ret double %0
}

define %struct.Gradients @test_derivative(double %x) {
entry:
  %0 = tail call %struct.Gradients (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @tester, metadata !"enzyme_width", i64 2, double %x, double 1.0, double 2.0)
  ret %struct.Gradients %0
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.fabs.f64(double)


; CHECK: define internal [2 x double] @fwddiffe2tester(double %x, [2 x double] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[i1:.+]] = fcmp fast olt double %x, 0.000000e+00
; CHECK-NEXT:   %[[i2:.+]] = select {{(fast )?}}i1 %[[i1]], double -1.000000e+00, double 1.000000e+00
; CHECK-NEXT:   %[[i0:.+]] = extractvalue [2 x double] %"x'", 0
; CHECK-NEXT:   %[[i3:.+]] = fmul fast double %[[i0]], %[[i2]]
; CHECK-NEXT:   %[[i4:.+]] = insertvalue [2 x double] undef, double %[[i3]], 0
; CHECK-NEXT:   %[[i5:.+]] = extractvalue [2 x double] %"x'", 1
; CHECK-NEXT:   %[[i6:.+]] = fmul fast double %[[i5]], %[[i2]]
; CHECK-NEXT:   %[[i7:.+]] = insertvalue [2 x double] %[[i4]], double %[[i6]], 1
; CHECK-NEXT:   ret [2 x double] %[[i7]]
; CHECK-NEXT: }