; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -S | FileCheck %s; fi

declare double @__enzyme_autodiff(...)

define double @f(double %x, i1 %cond) {
entry:
  br i1 %cond, label %ok, label %dead

dead:
  store double %x, ptr null, align 8
  %v = load double, ptr null, align 8
  ret double %v

ok:
  %sq = fmul double %x, %x
  ret double %sq
}

define double @df(double %x) {
entry:
  %r = call double (...) @__enzyme_autodiff(ptr @f, double %x, i1 true)
  ret double %r
}

; CHECK-LABEL: define internal { double } @diffef(
; CHECK: ret { double }
