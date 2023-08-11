; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -simplifycfg -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,sroa,instsimplify,%simplifycfg,adce)" -enzyme-preopt=false -S | FileCheck %s

%struct.Gradients = type { double, double, double }

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_fwddiff(double (double)*, ...)

declare double @erf(double)

define double @tester(double %x) {
entry:
  %call = call double @erf(double %x)
  ret double %call
}

define %struct.Gradients @test_derivative(double %x) {
entry:
  %0 = tail call %struct.Gradients (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @tester, metadata !"enzyme_width", i64 3, double %x, double 1.0, double 2.0, double 3.0)
  ret %struct.Gradients %0
}


; CHECK: define internal [3 x double] @fwddiffe3tester(double %x, [3 x double] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[i0:.+]] = fmul fast double %x, %x
; CHECK-NEXT:   %[[i1:.+]] = {{(fsub fast double \-?0.000000e\+00,|fneg fast double)}} %[[i0]]
; CHECK-NEXT:   %[[i2:.+]] = call fast double @llvm.exp.f64(double %[[i1]])
; CHECK-NEXT:   %[[i3:.+]] = fmul fast double 0x3FF20DD750429B6D, %[[i2]]
; CHECK-NEXT:   %[[i4:.+]] = extractvalue [3 x double] %"x'", 0
; CHECK-NEXT:   %[[i5:.+]] = fmul fast double %[[i4]], %[[i3]]
; CHECK-NEXT:   %[[i7:.+]] = extractvalue [3 x double] %"x'", 1
; CHECK-NEXT:   %[[i8:.+]] = fmul fast double %[[i7]], %[[i3]]
; CHECK-NEXT:   %[[i10:.+]] = extractvalue [3 x double] %"x'", 2
; CHECK-NEXT:   %[[i11:.+]] = fmul fast double %[[i10]], %[[i3]]
; CHECK-NEXT:   %[[i6:.+]] = insertvalue [3 x double] undef, double %[[i5]], 0
; CHECK-NEXT:   %[[i9:.+]] = insertvalue [3 x double] %[[i6]], double %[[i8]], 1
; CHECK-NEXT:   %[[i12:.+]] = insertvalue [3 x double] %[[i9]], double %[[i11]], 2
; CHECK-NEXT:   ret [3 x double] %[[i12]]
; CHECK-NEXT: }