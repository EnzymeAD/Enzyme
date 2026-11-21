; RUN: if [ %llvmver -ge 20 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %0 = tail call double @llvm.maximumnum.f64(double %x, double %y)
  ret double %0
}

define double @test_derivative(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__enzyme_fwddiff(double (double, double)* nonnull @tester, double %x, double 1.0, double %y, double 1.0)
  ret double %0
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.maximumnum.f64(double, double)

; Function Attrs: nounwind
declare double @__enzyme_fwddiff(double (double, double)*, ...)

; CHECK: define internal {{(dso_local )?}}double @fwddiffetester(double %x, double %"x'", double %y, double %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[lt:.+]] = fcmp fast olt double %x, %y
; CHECK-NEXT:   %[[eq:.+]] = fcmp fast oeq double %x, %y
; CHECK-NEXT:   %[[sx:.+]] = call fast double @llvm.copysign.f64(double 1.000000e+00, double %x)
; CHECK-NEXT:   %[[sy:.+]] = call fast double @llvm.copysign.f64(double 1.000000e+00, double %y)
; CHECK-NEXT:   %[[slt:.+]] = fcmp fast olt double %[[sx]], %[[sy]]
; CHECK-NEXT:   %[[tie:.+]] = and i1 %[[eq]], %[[slt]]
; CHECK-NEXT:   %[[cmp:.+]] = or i1 %[[lt]], %[[tie]]
; CHECK-NEXT:   %[[res:.+]] = select {{(fast )?}}i1 %[[cmp]], double %"y'", double %"x'"
; CHECK-NEXT:   ret double %[[res]]
; CHECK-NEXT: }
