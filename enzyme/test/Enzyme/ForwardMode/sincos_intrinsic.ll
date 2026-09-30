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
; CHECK: call {{(fast )?}}double @llvm.cos.f64(double %x)
; CHECK: call {{(fast )?}}double @llvm.sin.f64(double %x)
; CHECK: ret { double, double }
