; RUN: if [ %llvmver -ge 20 ]; then %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi

; llvm.sincos(x) = {sin(x), cos(x)}: tester(x) = 2 sin(x) + 3 cos(x),
; d/dx = 2 cos(x) - 3 sin(x)

declare { double, double } @llvm.sincos.f64(double)

define double @tester(double %x) {
entry:
  %sc = call { double, double } @llvm.sincos.f64(double %x)
  %s = extractvalue { double, double } %sc, 0
  %c = extractvalue { double, double } %sc, 1
  %s2 = fmul double %s, 2.000000e+00
  %c3 = fmul double %c, 3.000000e+00
  %res = fadd double %s2, %c3
  ret double %res
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (ptr, ...) @__enzyme_autodiff(ptr nonnull @tester, double %x)
  ret double %0
}

declare double @__enzyme_autodiff(ptr, ...)

; CHECK: define internal { double } @diffetester(double %x, double %differeturn)
; CHECK-NOT: unreachable
; CHECK: call {{(fast )?}}double @llvm.cos.f64(double %x)
; CHECK: call {{(fast )?}}double @llvm.sin.f64(double %x)
; CHECK: ret { double }
