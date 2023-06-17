; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -early-cse -instcombine -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,early-cse,instcombine,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s

%struct.Gradients = type { double, double }

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_fwddiff(double (double, double)*, ...)

define double @tester(double %x, double %y) {
entry:
  %0 = tail call double @llvm.minnum.f64(double %x, double %y)
  ret double %0
}

define %struct.Gradients @test_derivative(double %x, double %y) {
entry:
  %0 = tail call %struct.Gradients (double (double, double)*, ...) @__enzyme_fwddiff(double (double, double)* nonnull @tester, metadata !"enzyme_width", i64 2, double %x, double 1.0, double 0.0, double %y, double 0.0, double 1.0)
  ret %struct.Gradients %0
}

declare double @llvm.minnum.f64(double, double)


; CHECK: define internal [2 x double] @fwddiffe2tester(double %x, [2 x double] %"x'", double %y, [2 x double] %"y'")
; CHECK-NEXT: entry:
; CHECK-DAG:   %[[i0:.+]] = fcmp fast olt double %x, %y
; CHECK-DAG:   %[[i1:.+]] = extractvalue [2 x double] %"x'", 0
; CHECK-DAG:   %[[i5:.+]] = extractvalue [2 x double] %"x'", 1
; CHECK-DAG:   %[[i2:.+]] = extractvalue [2 x double] %"y'", 0
; CHECK-DAG:   %[[i6:.+]] = extractvalue [2 x double] %"y'", 1
; CHECK-DAG:   %[[i3:.+]] = select {{(fast )?}}i1 %[[i0]], double %[[i1]], double %[[i2]]
; CHECK-DAG:   %[[i4:.+]] = insertvalue [2 x double] undef, double %[[i3]], 0
; CHECK-DAG:   %[[i7:.+]] = select {{(fast )?}}i1 %[[i0]], double %[[i5]], double %[[i6]]
; CHECK-DAG:   %[[i8:.+]] = insertvalue [2 x double] %[[i4]], double %[[i7]], 1
; CHECK-DAG:   ret [2 x double] %[[i8]]
; CHECK-NEXT: }
