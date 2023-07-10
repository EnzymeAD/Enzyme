; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -simplifycfg -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,sroa,instsimplify,%simplifycfg,adce)" -enzyme-preopt=false -S | FileCheck %s

%struct.Gradients = type { double, double }

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_fwddiff(double (double)*, ...)

declare double @erfc(double)

define double @tester(double %x) {
entry:
  %call = call double @erfc(double %x)
  ret double %call
}

define %struct.Gradients @test_derivative(double %x) {
entry:
  %0 = tail call %struct.Gradients (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @tester, metadata !"enzyme_width", i64 2, double %x, double 1.0, double 1.5)
  ret %struct.Gradients %0
}


; CHECK: define internal [2 x double] @fwddiffe2tester(double %x, [2 x double] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[a0:.+]] = fmul fast double %x, %x
; CHECK-NEXT:   %[[a1:.+]] = {{(fsub fast double \-?0.000000e\+00,|fneg fast double)}} %[[a0]]
; CHECK-NEXT:   %[[a2:.+]] = call fast double @llvm.exp.f64(double %[[a1]])
; CHECK-NEXT:   %[[a3:.+]] = fmul fast double 0xBFF20DD750429B6D, %[[a2]]
; CHECK-NEXT:   %[[a4:.+]] = extractvalue [2 x double] %"x'", 0
; CHECK-NEXT:   %[[a5:.+]] = fmul fast double %[[a4]], %[[a3]]
; CHECK-NEXT:   %[[a7:.+]] = extractvalue [2 x double] %"x'", 1
; CHECK-NEXT:   %[[a8:.+]] = fmul fast double %[[a7]], %[[a3]]
; CHECK-NEXT:   %[[a6:.+]] = insertvalue [2 x double] undef, double %[[a5]], 0
; CHECK-NEXT:   %[[a9:.+]] = insertvalue [2 x double] %[[a6]], double %[[a8]], 1
; CHECK-NEXT:   ret [2 x double] %[[a9]]
; CHECK-NEXT: }