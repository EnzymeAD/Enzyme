; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s

%struct.Gradients = type { double, double, double }

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_fwddiff(double (double)*, ...)

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %agg1 = insertvalue [3 x double] undef, double %x, 0
  %mul = fmul double %x, %x
  %agg2 = insertvalue [3 x double] %agg1, double %mul, 1
  %add = fadd double %mul, 2.0
  %agg3 = insertvalue [3 x double] %agg2, double %add, 2
  %res = extractvalue [3 x double] %agg2, 1
  ret double %res
}

define %struct.Gradients @test_derivative(double %x) {
entry:
  %0 = tail call %struct.Gradients (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @tester, metadata !"enzyme_width", i64 3, double %x, double 1.0, double 2.0, double 3.0)
  ret %struct.Gradients %0
}


; CHECK: define internal [3 x double] @fwddiffe3tester(double %x, [3 x double] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[i0:.+]] = extractvalue [3 x double] %"x'", 0
; CHECK-NEXT:   %[[i2:.+]] = fmul fast double %[[i0]], %x
; CHECK-NEXT:   %[[i5:.+]] = extractvalue [3 x double] %"x'", 1
; CHECK-NEXT:   %[[i7:.+]] = fmul fast double %[[i5]], %x
; CHECK-NEXT:   %[[i10:.+]] = extractvalue [3 x double] %"x'", 2
; CHECK-NEXT:   %[[i12:.+]] = fmul fast double %[[i10]], %x
; CHECK-NEXT:   %[[i1:.+]] = extractvalue [3 x double] %"x'", 0
; CHECK-NEXT:   %[[i3:.+]] = fmul fast double %[[i1]], %x
; CHECK-NEXT:   %[[i6:.+]] = extractvalue [3 x double] %"x'", 1
; CHECK-NEXT:   %[[i8:.+]] = fmul fast double %[[i6]], %x
; CHECK-NEXT:   %[[i11:.+]] = extractvalue [3 x double] %"x'", 2
; CHECK-NEXT:   %[[i13:.+]] = fmul fast double %[[i11]], %x
; CHECK-NEXT:   %[[i4:.+]] = fadd fast double %[[i2]], %[[i3]]
; CHECK-NEXT:   %[[i9:.+]] = fadd fast double %[[i7]], %[[i8]]
; CHECK-NEXT:   %[[i14:.+]] = fadd fast double %[[i12]], %[[i13]]
; CHECK-NEXT:   %[[i15:.+]] = insertvalue [3 x double] undef, double %[[i4]], 0
; CHECK-NEXT:   %[[i16:.+]] = insertvalue [3 x double] %[[i15]], double %[[i9]], 1
; CHECK-NEXT:   %[[i17:.+]] = insertvalue [3 x double] %[[i16]], double %[[i14]], 2
; CHECK-NEXT:   ret [3 x double] %[[i17]]
; CHECK-NEXT: }