; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s

%struct.Gradients = type { double, double }

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_fwddiff(double (double, double)*, ...)

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %0 = tail call fast double @llvm.pow.f64(double %x, double %y)
  ret double %0
}

define %struct.Gradients @test_derivative(double %x, double %y) {
entry:
  %0 = tail call %struct.Gradients (double (double, double)*, ...) @__enzyme_fwddiff(double (double, double)* nonnull @tester, metadata !"enzyme_width", i64 2, double %x, double 1.0, double 0.0, double %y, double 0.0, double 1.0)
  ret %struct.Gradients %0
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.pow.f64(double, double)


; CHECK: define internal [2 x double] @fwddiffe2tester(double %x, [2 x double] %"x'", double %y, [2 x double] %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = fsub fast double %y, 1.000000e+00
; CHECK-NEXT:   %1 = call fast double @llvm.pow.f64(double %x, double %0)
; CHECK-NEXT:   %2 = fmul fast double %y, %1
; CHECK-NEXT:   %3 = extractvalue [2 x double] %"x'", 0
; CHECK-NEXT:   %4 = fmul fast double %3, %2
; CHECK-NEXT:   %5 = extractvalue [2 x double] %"x'", 1
; CHECK-NEXT:   %6 = fmul fast double %5, %2
; CHECK-NEXT:   %7 = call fast double @llvm.pow.f64(double %x, double %y)
; CHECK-NEXT:   %8 = call fast double @llvm.log.f64(double %x)
; CHECK-NEXT:   %9 = fmul fast double %7, %8
; CHECK-NEXT:   %10 = extractvalue [2 x double] %"y'", 0
; CHECK-NEXT:   %11 = fmul fast double %10, %9
; CHECK-NEXT:   %[[i14:.+]] = extractvalue [2 x double] %"y'", 1
; CHECK-NEXT:   %[[i15:.+]] = fmul fast double %[[i14]], %9
; CHECK-NEXT:   %[[i12:.+]] = fadd fast double %4, %11
; CHECK-NEXT:   %[[i13:.+]] = insertvalue [2 x double] undef, double %[[i12]], 0
; CHECK-NEXT:   %[[i16:.+]] = fadd fast double %6, %[[i15]]
; CHECK-NEXT:   %[[i17:.+]] = insertvalue [2 x double] %[[i13]], double %[[i16]], 1
; CHECK-NEXT:   ret [2 x double] %[[i17]]
; CHECK-NEXT: }
