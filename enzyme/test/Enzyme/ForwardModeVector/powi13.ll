; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s

%struct.Gradients = type { double, double, double }

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_fwddiff(double (double, i32)*, ...)

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x, i32 %y) {
entry:
  %0 = tail call fast double @llvm.powi.f64.i32(double %x, i32 %y)
  ret double %0
}

define %struct.Gradients @test_derivative(double %x, i32 %y) {
entry:
  %0 = tail call %struct.Gradients (double (double, i32)*, ...) @__enzyme_fwddiff(double (double, i32)* nonnull @tester, metadata !"enzyme_width", i64 3, double %x, double 1.0, double 2.0, double 3.0, i32 %y)
  ret %struct.Gradients %0
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.powi.f64.i32(double, i32)


; CHECK: define internal [3 x double] @fwddiffe3tester(double %x, [3 x double] %"x'", i32 %y)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[i3:.+]] = icmp eq i32 %y, 0
; CHECK-NEXT:   %[[i2:.+]] = sitofp i32 %y to double
; CHECK-NEXT:   %[[i0:.+]] = sub i32 %y, 1
; CHECK-NEXT:   %[[i1:.+]] = call fast double @llvm.powi.f64{{(\.i32)?}}(double %x, i32 %[[i0]])
; CHECK-NEXT:   %[[i6:.+]] = fmul fast double %[[i2]], %[[i1]]
; CHECK-NEXT:   %[[i4:.+]] = extractvalue [3 x double] %"x'", 0
; CHECK-NEXT:   %[[i5:.+]] = fmul fast double %[[i4]], %[[i6]]
; CHECK-NEXT:   %[[i9:.+]] = extractvalue [3 x double] %"x'", 1
; CHECK-NEXT:   %[[i11:.+]] = fmul fast double %[[i9]], %[[i6]]
; CHECK-NEXT:   %[[i14:.+]] = extractvalue [3 x double] %"x'", 2
; CHECK-NEXT:   %[[i16:.+]] = fmul fast double %[[i14]], %[[i6]]
; CHECK-NEXT:   %[[i7:.+]] = select {{(fast )?}}i1 %[[i3]], double 0.000000e+00, double %[[i5]]
; CHECK-NEXT:   %[[i12:.+]] = select {{(fast )?}}i1 %[[i3]], double 0.000000e+00, double %[[i11]]
; CHECK-NEXT:   %[[i17:.+]] = select {{(fast )?}}i1 %[[i3]], double 0.000000e+00, double %[[i16]]
; CHECK-NEXT:   %[[i8:.+]] = insertvalue [3 x double] undef, double %[[i7]], 0
; CHECK-NEXT:   %[[i13:.+]] = insertvalue [3 x double] %[[i8]], double %[[i12]], 1
; CHECK-NEXT:   %[[i18:.+]] = insertvalue [3 x double] %[[i13]], double %[[i17]], 2
; CHECK-NEXT:   ret [3 x double] %[[i18:.+]]
; CHECK-NEXT }
