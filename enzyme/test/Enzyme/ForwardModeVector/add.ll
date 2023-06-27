; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s

%struct.Gradients = type { double, double }

define double @tester(double %x, double %y) {
entry:
  %add = fadd double %x, %y
  ret double %add
}

define %struct.Gradients @test_derivative(double %x, double %y){
entry:
  %call = call %struct.Gradients (double (double, double)*, ...) @__enzyme_fwddiff(double (double, double)* nonnull @tester, metadata !"enzyme_width", i64 2, double %x, double 1.000000e+00, double 0.000000e+00, double %y, double 0.000000e+00, double 1.000000e+00)
  ret %struct.Gradients %call
}

declare %struct.Gradients @__enzyme_fwddiff(double (double, double)*, ...)

; CHECK: define internal [2 x double] @fwddiffe2tester(double %x, [2 x double] %"x'", double %y, [2 x double] %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[i0:.+]] = extractvalue [2 x double] %"x'", 0
; CHECK-NEXT:   %[[i4:.+]] = extractvalue [2 x double] %"x'", 1
; CHECK-NEXT:   %[[i1:.+]] = extractvalue [2 x double] %"y'", 0
; CHECK-NEXT:   %[[i2:.+]] = fadd fast double %[[i0]], %[[i1]]
; CHECK-NEXT:   %[[i3:.+]] = insertvalue [2 x double] undef, double %[[i2]], 0
; CHECK-NEXT:   %[[i5:.+]] = extractvalue [2 x double] %"y'", 1
; CHECK-NEXT:   %[[i6:.+]] = fadd fast double %[[i4]], %[[i5]]
; CHECK-NEXT:   %[[i7:.+]] = insertvalue [2 x double] %[[i3]], double %[[i6]], 1
; CHECK-NEXT:   ret [2 x double] %[[i7]]
; CHECK-NEXT: }