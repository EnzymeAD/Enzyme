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
; CHECK-NEXT:   %[[ORD:.+]] = fcmp ord double %y, %y
; CHECK-NEXT:   %[[ULT:.+]] = fcmp ult double %x, %y
; CHECK-NEXT:   %[[OEQ:.+]] = fcmp oeq double %x, %y
; CHECK-NEXT:   %[[XI:.+]] = bitcast double %x to i64
; CHECK-NEXT:   %[[YI:.+]] = bitcast double %y to i64
; CHECK-NEXT:   %[[NOTYI:.+]] = xor i64 %[[YI]], -1
; CHECK-NEXT:   %[[MASK:.+]] = and i64 %[[XI]], %[[NOTYI]]
; CHECK-NEXT:   %[[SIGN:.+]] = icmp slt i64 %[[MASK]], 0
; CHECK-NEXT:   %[[NPZ:.+]] = and i1 %[[OEQ]], %[[SIGN]]
; CHECK-NEXT:   %[[OR:.+]] = or i1 %[[ULT]], %[[NPZ]]
; CHECK-NEXT:   %[[CMP:.+]] = and i1 %[[ORD]], %[[OR]]
; CHECK-NEXT:   %[[RES:.+]] = select {{(fast )?}}i1 %[[CMP]], double %"y'", double %"x'"
; CHECK-NEXT:   ret double %[[RES]]
; CHECK-NEXT: }
