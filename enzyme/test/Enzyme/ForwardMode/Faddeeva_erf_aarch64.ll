; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -early-cse -instsimplify -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(early-cse,instsimplify)" -enzyme-preopt=false -S | FileCheck %s

declare [2 x double] @Faddeeva_erf([2 x double], double)

define [2 x double] @tester([2 x double] %in) {
entry:
  %call = call [2 x double] @Faddeeva_erf([2 x double] %in, double 0.000000e+00)
  ret [2 x double] %call
}

define [2 x double] @test_derivative([2 x double] %x) {
entry:
  %0 = tail call [2 x double] ([2 x double] ([2 x double])*, ...) @__enzyme_fwddiff([2 x double] ([2 x double])*  @tester, [2 x double] %x, [2 x double] [ double 1.0, double 1.0 ])
  ret [2 x double] %0
}

; Function Attrs: nounwind
declare [2 x double] @__enzyme_fwddiff([2 x double] ([2 x double])*, ...)


; CHECK: define internal [2 x double] @fwddiffetester([2 x double] %in, [2 x double] %"in'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[a0:.+]] = extractvalue [2 x double] %in, 0
; CHECK-NEXT:   %[[a1:.+]] = extractvalue [2 x double] %in, 1
; CHECK-DAG:    %[[a2:.+]] = fmul fast double %[[a0]], %[[a0]]
; CHECK-DAG:    %[[a3:.+]] = fmul fast double %[[a1]], %[[a1]]
; CHECK-NEXT:   %[[a4:.+]] = fsub fast double %[[a2]], %[[a3]]
; CHECK-NEXT:   %[[a5:.+]] = fmul fast double %[[a0]], %[[a1]]
; CHECK-NEXT:   %[[a6:.+]] = fadd fast double %[[a5]], %[[a5]]
; CHECK-NEXT:   %[[a7:.+]] = {{(fsub fast double \-0.000000e\+00,|fneg fast double)}} %[[a4]]
; CHECK-NEXT:   %[[a8:.+]] = {{(fsub fast double \-0.000000e\+00,|fneg fast double)}} %[[a6]]
; CHECK-NEXT:   %[[a9:.+]] = call fast double @llvm.exp.f64(double %[[a7]])
; CHECK-NEXT:   %[[a10:.+]] = call fast double @llvm.cos.f64(double %[[a8]])
; CHECK-NEXT:   %[[a11:.+]] = fmul fast double %[[a9]], %[[a10]]
; CHECK-NEXT:   %[[a12:.+]] = call fast double @llvm.sin.f64(double %[[a8]])
; CHECK-NEXT:   %[[a13:.+]] = fmul fast double %[[a9]], %[[a12]]
; CHECK-NEXT:   %[[a14:.+]] = fmul fast double 0x3FF20DD750429B6D, %[[a11]]
; CHECK-NEXT:   %[[a16:.+]] = fmul fast double 0x3FF20DD750429B6D, %[[a13]]
; CHECK-NEXT:   %[[a18:.+]] = extractvalue [2 x double] %"in'", 0
; CHECK-NEXT:   %[[a19:.+]] = extractvalue [2 x double] %"in'", 1
; CHECK-DAG:    %[[a20:.+]] = fmul fast double %[[a18]], %[[a14]]
; CHECK-DAG:    %[[a21:.+]] = fmul fast double %[[a19]], %[[a16]]
; CHECK-NEXT:   %[[a22:.+]] = fsub fast double %[[a20]], %[[a21]]
; CHECK-DAG:    %[[a24:.+]] = fmul fast double %[[a18]], %[[a16]]
; CHECK-DAG:    %[[a25:.+]] = fmul fast double %[[a14]], %[[a19]]
; CHECK-NEXT:   %[[a26:.+]] = fadd fast double %[[a24]], %[[a25]]
; CHECK-NEXT:   %[[a23:.+]] = insertvalue [2 x double] undef, double %[[a22]], 0
; CHECK-NEXT:   %[[a27:.+]] = insertvalue [2 x double] %[[a23]], double %[[a26]], 1
; CHECK-NEXT:   ret [2 x double] %[[a27]]
; CHECK-NEXT: }
