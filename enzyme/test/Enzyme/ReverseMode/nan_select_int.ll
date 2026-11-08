; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -enzyme-check-nan=1 -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -enzyme-check-nan=1 -S | FileCheck %s

declare double @__enzyme_autodiff(...)

define double @f(double %x, i1 %cond) {
entry:
  %xi = bitcast double %x to i64
  br i1 %cond, label %then, label %merge

then:
  br label %merge

merge:
  %phi = phi i64 [ %xi, %then ], [ 0, %entry ]
  %res = bitcast i64 %phi to double
  ret double %res
}

define double @df(double %x, i1 %cond) {
entry:
  %r = call double (...) @__enzyme_autodiff(ptr @f, double %x, i1 %cond)
  ret double %r
}

; CHECK-LABEL: define internal { double } @diffef(double %x, i1 %cond, double %differeturn)
; CHECK: call void @__enzyme_sanitize_nan_double(double
