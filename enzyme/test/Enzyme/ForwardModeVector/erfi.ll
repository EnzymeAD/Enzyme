; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -simplifycfg -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,sroa,instsimplify,%simplifycfg,adce)" -enzyme-preopt=false -S | FileCheck %s

%struct.Gradients = type { double, double }

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_fwddiff(double (double)*, ...)

declare double @erfi(double)

define double @tester(double %x) {
entry:
  %call = call double @erfi(double %x)
  ret double %call
}

define %struct.Gradients @test_derivative(double %x) {
entry:
  %0 = tail call %struct.Gradients (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @tester, metadata !"enzyme_width", i64 2, double %x, double 1.0, double 2.0)
  ret %struct.Gradients %0
}


; CHECK: define internal [2 x double] @fwddiffe2tester(double %x, [2 x double] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[a0:.+]] = fmul fast double %x, %x
; CHECK-NEXT:   %[[a1:.+]] = call fast double @llvm.exp.f64(double %[[a0]])
; CHECK-NEXT:   %[[a2:.+]] = fmul fast double 0x3FF20DD750429B6D, %[[a1]]
; CHECK-NEXT:   %[[a3:.+]] = extractvalue [2 x double] %"x'", 0
; CHECK-NEXT:   %[[a4:.+]] = fmul fast double %[[a3]], %[[a2]]
; CHECK-NEXT:   %[[a6:.+]] = extractvalue [2 x double] %"x'", 1
; CHECK-NEXT:   %[[a7:.+]] = fmul fast double %[[a6]], %[[a2]]
; CHECK-NEXT:   %[[a5:.+]] = insertvalue [2 x double] undef, double %[[a4]], 0
; CHECK-NEXT:   %[[a8:.+]] = insertvalue [2 x double] %[[a5]], double %[[a7]], 1
; CHECK-NEXT:   ret [2 x double] %[[a8]]
; CHECK-NEXT: }