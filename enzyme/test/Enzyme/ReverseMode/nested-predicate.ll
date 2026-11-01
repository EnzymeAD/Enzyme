; RUN: %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg)" -S -o %t
; RUN: FileCheck %s < %t
; RUN: %lli %t
; RUN: %opt %t -passes="default<O2>" -o %t.opt
; RUN: %lli %t.opt

; The inner predicate is not evaluated when fan is false. Reusing it at the
; reverse merge must not speculate the load from p, which may be null.
; Check the conservative cached-edge fallback as well as the fast path for
; an available predicate, including poison on an untaken path.

@enzyme_const = external global i32

define double @nested(double %x, ptr %p, i1 %fan) {
entry:
  br i1 %fan, label %inner, label %exit
inner:
  %v = load double, ptr %p
  %positive = fcmp ogt double %v, 0.0
  %twice = fmul double %x, 2.0
  br i1 %positive, label %square, label %exit
square:
  %sq = fmul double %x, %x
  br label %exit
exit:
  %result = phi double [ %x, %entry ], [ %twice, %inner ], [ %sq, %square ]
  ret double %result
}
; Overwriting p forces the inner predicate to be saved rather than reloaded.
define double @cached(double %x, ptr %p, i1 %fan) {
entry:
  br i1 %fan, label %inner, label %exit
inner:
  %v = load double, ptr %p
  %positive = fcmp ogt double %v, 0.0
  store double 0.0, ptr %p
  %twice = fmul double %x, 2.0
  br i1 %positive, label %square, label %exit
square:
  %sq = fmul double %x, %x
  br label %exit
exit:
  %result = phi double [ %x, %entry ], [ %twice, %inner ], [ %sq, %square ]
  ret double %result
}
declare double @__enzyme_autodiff(...)
define double @derivative(double %x, ptr %p, i1 %fan) {
  %d = call double (...) @__enzyme_autodiff(ptr @nested, double %x, ptr @enzyme_const, ptr %p, i1 %fan)
  ret double %d
}

define double @cached_derivative(double %x, ptr %p, i1 %fan) {
  %d = call double (...) @__enzyme_autodiff(ptr @cached, double %x, ptr @enzyme_const, ptr %p, i1 %fan)
  ret double %d
}

; A condition available at entry can still use the three-target fast path.
; It may be poison when fan is false, so PHI masks must short-circuit.
define double @available(double %x, i1 %fan, i1 %positive) {
entry:
  br i1 %fan, label %inner, label %exit
inner:
  %twice = fmul double %x, 2.0
  br i1 %positive, label %square, label %exit
square:
  %sq = fmul double %x, %x
  br label %exit
exit:
  %result = phi double [ %x, %entry ], [ %twice, %inner ], [ %sq, %square ]
  ret double %result
}
define double @available_derivative(double %x, i1 %fan, i1 %positive) {
  %d = call double (...) @__enzyme_autodiff(ptr @available, double %x, i1 %fan, i1 %positive)
  ret double %d
}
define i32 @main() {
  %p = alloca double
  %d0 = call double @derivative(double 3.0, ptr null, i1 false)
  store double -1.0, ptr %p
  %d1 = call double @derivative(double 3.0, ptr %p, i1 true)
  store double 1.0, ptr %p
  %d2 = call double @derivative(double 3.0, ptr %p, i1 true)
  %d3 = call double @available_derivative(double 3.0, i1 false, i1 poison)
  %d4 = call double @available_derivative(double 3.0, i1 true, i1 false)
  %d5 = call double @available_derivative(double 3.0, i1 true, i1 true)
  %d6 = call double @cached_derivative(double 3.0, ptr null, i1 false)
  store double -1.0, ptr %p
  %d7 = call double @cached_derivative(double 3.0, ptr %p, i1 true)
  store double 1.0, ptr %p
  %d8 = call double @cached_derivative(double 3.0, ptr %p, i1 true)
  %ok6 = fcmp oeq double %d6, 1.0
  %ok7 = fcmp oeq double %d7, 2.0
  %ok8 = fcmp oeq double %d8, 6.0
  %ok67 = and i1 %ok6, %ok7
  %ok678 = and i1 %ok67, %ok8
  %ok0 = fcmp oeq double %d0, 1.0
  %ok1 = fcmp oeq double %d1, 2.0
  %ok2 = fcmp oeq double %d2, 6.0
  %ok3 = fcmp oeq double %d3, 1.0
  %ok4 = fcmp oeq double %d4, 2.0
  %ok5 = fcmp oeq double %d5, 6.0
  %ok01 = and i1 %ok0, %ok1
  %ok23 = and i1 %ok2, %ok3
  %ok45 = and i1 %ok4, %ok5
  %ok0123 = and i1 %ok01, %ok23
  %ok = and i1 %ok0123, %ok45
  %all = and i1 %ok, %ok678
  %status = select i1 %all, i32 0, i32 1
  ret i32 %status
}

; CHECK-LABEL: define internal { double } @diffenested(
; CHECK: inner:
; CHECK: %v = load double, ptr %p
; CHECK: exit:
; CHECK: phi i8
; CHECK: invertexit:
; CHECK-NOT: load double, ptr %p
; CHECK: switch i8
; CHECK-NOT: load double, ptr %p
; CHECK: }

; CHECK-LABEL: define internal { double } @diffecached(
; CHECK: exit:
; CHECK: phi i8
; CHECK: invertexit:
; CHECK: switch i8
; CHECK: }

; CHECK-LABEL: define internal { double } @diffeavailable(
; CHECK: invertexit:
; CHECK: select i1 %fan, i1 %positive, i1 false
; CHECK: %[[NOT:.+]] = xor i1 %positive, true
; CHECK: select i1 %fan, i1 %[[NOT]], i1 false
; CHECK: br i1 %fan, label %staging, label %invertentry
; CHECK: staging:
; CHECK-NEXT: br i1 %positive, label %invertsquare, label %invertinner
