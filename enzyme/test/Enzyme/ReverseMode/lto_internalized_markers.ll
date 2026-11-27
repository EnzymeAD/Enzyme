; RUN: %opt < %s %newLoadEnzyme -passes="preserve-nvvm,globalopt,instcombine,enzyme,preserve-nvvm-end,function(mem2reg,instsimplify),globalopt" -S | FileCheck %s

@enzyme_out = internal global i32 0, align 4
@enzyme_const = internal constant i32 0, align 4

define double @f(double %x, double %y) {
entry:
  %mul = fmul double %x, %y
  %div = fdiv double 1.000000e+00, %y
  %add = fadd double %mul, %div
  ret double %add
}

define { double, double } @test_both_out(double %x, double %y) {
entry:
  %m0 = load i32, i32* @enzyme_out, align 4
  %m1 = load i32, i32* @enzyme_out, align 4
  %res = call { double, double } (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (double, double)* @f to i8*), i32 %m0, double %x, i32 %m1, double %y)
  ret { double, double } %res
}

define double @test_const_and_out(double %x, double %y) {
entry:
  %m0 = load i32, i32* @enzyme_out, align 4
  %m1 = load i32, i32* @enzyme_const, align 4
  %res = call double (i8*, ...) @__enzyme_autodiff_single(i8* bitcast (double (double, double)* @f to i8*), i32 %m0, double %x, i32 %m1, double %y)
  ret double %res
}

declare { double, double } @__enzyme_autodiff(i8*, ...)
declare double @__enzyme_autodiff_single(i8*, ...)

; CHECK-NOT: @enzyme_out
; CHECK-NOT: @enzyme_const

; CHECK-LABEL: define { double, double } @test_both_out(double %x, double %y)
; CHECK: call {{(fastcc )?}}{ double, double } @diffef(double %x, double %y, double 1.000000e+00)

; CHECK-LABEL: define double @test_const_and_out(double %x, double %y)
; CHECK: call {{(fastcc )?}}{ double } @diffef.1(double %x, double %y, double 1.000000e+00)
