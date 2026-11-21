; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -enzyme-preopt=false -S | FileCheck %s
; RUN: if [ %llvmver -ge 17 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -enzyme-preopt=false -S | %lli - | FileCheck %s --check-prefix=EVAL; fi
; EVAL: 1.000000 1.000000 0.000000 1.000000

; llvm.maximum and llvm.minimum order -0.0 below +0.0, so the derivative must
; follow the operand the intrinsic actually returns: relu(x) - relu(-x) has
; derivative 1 at x = 0 rather than 2, maximum(-0.0, 0.0) returns the constant
; second operand, and minimum(-0.0, 0.0) returns the active first operand.

@.fmt = private unnamed_addr constant [25 x i8] c"%f %f %f %f\0A\00\00\00\00\00\00\00\00\00\00\00\00\00", align 1

declare double @llvm.maximum.f64(double, double)
declare double @llvm.minimum.f64(double, double)
declare double @__enzyme_autodiff(...)
declare double @__enzyme_fwddiff(...)
declare i32 @printf(ptr, ...)

define double @relu(double %x) {
entry:
  %r = call double @llvm.maximum.f64(double %x, double 0.000000e+00)
  ret double %r
}

define double @odd_relu(double %x) {
entry:
  %pos = call double @llvm.maximum.f64(double %x, double 0.000000e+00)
  %nx = fneg double %x
  %neg = call double @llvm.maximum.f64(double %nx, double 0.000000e+00)
  %res = fsub double %pos, %neg
  ret double %res
}

define double @min_with_zero(double %x) {
entry:
  %r = call double @llvm.minimum.f64(double %x, double 0.000000e+00)
  ret double %r
}

; CHECK-LABEL: define i32 @main(
define i32 @main() {
entry:
  ; CHECK: call { double } @diffeodd_relu(double 0.000000e+00, double 1.000000e+00)
  %rev = call double (...) @__enzyme_autodiff(ptr @odd_relu, double 0.000000e+00)
  ; CHECK: call fast double @fwddiffeodd_relu(double 0.000000e+00, double 1.000000e+00)
  %fwd = call double (...) @__enzyme_fwddiff(ptr @odd_relu, double 0.000000e+00, double 1.000000e+00)
  ; CHECK: call { double } @differelu(double -0.000000e+00, double 1.000000e+00)
  %relu_negzero = call double (...) @__enzyme_autodiff(ptr @relu, double -0.000000e+00)
  ; CHECK: call { double } @diffemin_with_zero(double -0.000000e+00, double 1.000000e+00)
  %min_negzero = call double (...) @__enzyme_autodiff(ptr @min_with_zero, double -0.000000e+00)
  call i32 (ptr, ...) @printf(ptr @.fmt, double %rev, double %fwd, double %relu_negzero, double %min_negzero)
  ret i32 0
}

; CHECK-LABEL: define internal { double } @differelu(
; CHECK: %[[lt:.+]] = fcmp fast olt double %x, 0.000000e+00
; CHECK-NEXT: %[[eq:.+]] = fcmp fast oeq double %x, 0.000000e+00
; CHECK-NEXT: %[[sx:.+]] = call fast double @llvm.copysign.f64(double 1.000000e+00, double %x)
; CHECK-NEXT: %[[sy:.+]] = call fast double @llvm.copysign.f64(double 1.000000e+00, double 0.000000e+00)
; CHECK-NEXT: %[[slt:.+]] = fcmp fast olt double %[[sx]], %[[sy]]
; CHECK-NEXT: %[[tie:.+]] = and i1 %[[eq]], %[[slt]]
; CHECK-NEXT: %[[cmp:.+]] = or i1 %[[lt]], %[[tie]]
; CHECK-NEXT: %[[dx:.+]] = select {{(fast )?}}i1 %[[cmp]], double 0.000000e+00, double %{{.+}}
