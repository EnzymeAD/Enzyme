; RUN: if [ %llvmver -ge 20 ]; then %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi

; llvm.sincos(x) = {sin(x), cos(x)}: the tangent is {cos(x) dx, -sin(x) dx}

declare { double, double } @llvm.sincos.f64(double)

define { double, double } @tester(double %x) {
entry:
  %sc = call { double, double } @llvm.sincos.f64(double %x)
  ret { double, double } %sc
}

define { double, double } @test_derivative(double %x) {
entry:
  %0 = tail call { double, double } (ptr, ...) @__enzyme_fwddiff(ptr nonnull @tester, double %x, double 1.0)
  ret { double, double } %0
}

declare { double, double } @__enzyme_fwddiff(ptr, ...)

; CHECK: define internal { double, double } @fwddiffetester(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call {{(fast )?}}{ double, double } @llvm.sincos.f64(double %x)
; CHECK-NEXT:   %1 = extractvalue { double, double } %0, 1
; CHECK-NEXT:   %2 = fmul fast double %1, %"x'"
; CHECK-NEXT:   %3 = insertvalue { double, double } undef, double %2, 0
; CHECK-NEXT:   %4 = extractvalue { double, double } %0, 0
; CHECK-NEXT:   %5 = fmul fast double %4, %"x'"
; CHECK-NEXT:   %6 = fneg fast double %5
; CHECK-NEXT:   %7 = insertvalue { double, double } %3, double %6, 1
; CHECK-NEXT:   ret { double, double } %7
; CHECK-NEXT: }
