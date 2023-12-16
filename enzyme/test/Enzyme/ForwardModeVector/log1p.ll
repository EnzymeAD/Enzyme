; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -gvn -simplifycfg -instcombine -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,gvn,%simplifycfg,instcombine)" -enzyme-preopt=false -S | FileCheck %s

%struct.Gradients = type { double, double, double }

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_fwddiff(double (double)*, ...)

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %0 = tail call double @log1p(double %x)
  ret double %0
}

define %struct.Gradients @test_derivative(double %x) {
entry:
  %0 = tail call %struct.Gradients (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @tester, metadata !"enzyme_width", i64 3, double %x, double 1.0, double 2.0, double 3.0)
  ret %struct.Gradients %0
}

; Function Attrs: nounwind readnone speculatable
declare double @log1p(double)

; CHECK: define internal [3 x double] @fwddiffe3tester(double %x, [3 x double] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[i0:.+]] = fadd fast double %x, 1.000000e+00
; CHECK-NEXT:   %[[i1:.+]] = extractvalue [3 x double] %"x'", 0
; CHECK-NEXT:   %[[i2:.+]] = fdiv fast double %[[i1]], %[[i0]]
; CHECK-NEXT:   %[[i4:.+]] = extractvalue [3 x double] %"x'", 1
; CHECK-NEXT:   %[[i5:.+]] = fdiv fast double %[[i4]], %[[i0]]
; CHECK-NEXT:   %[[i7:.+]] = extractvalue [3 x double] %"x'", 2
; CHECK-NEXT:   %[[i8:.+]] = fdiv fast double %[[i7]], %[[i0]]
; CHECK-NEXT:   %[[i3:.+]] = insertvalue [3 x double] undef, double %[[i2]], 0
; CHECK-NEXT:   %[[i6:.+]] = insertvalue [3 x double] %[[i3]], double %[[i5]], 1
; CHECK-NEXT:   %[[i9:.+]] = insertvalue [3 x double] %[[i6]], double %[[i8]], 2
; CHECK-NEXT:   ret [3 x double] %[[i9]]
; CHECK-NEXT: }
