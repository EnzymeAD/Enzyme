; RUN: if [ %llvmver -ge 17 ]; then %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi

; The unsigned max of the bits of two nonnegative doubles is their max, as in
; Julia's `hypot`. Its derivative goes to the larger operand.

declare i64 @llvm.umax.i64(i64, i64)

define double @f(double %x, double %y) {
entry:
  %xi = bitcast double %x to i64
  %yi = bitcast double %y to i64
  %m = call i64 @llvm.umax.i64(i64 %xi, i64 %yi)
  %r = bitcast i64 %m to double
  ret double %r
}

declare { double, double } @__enzyme_autodiff(...)

define { double, double } @df(double %x, double %y) {
entry:
  %r = call { double, double } (...) @__enzyme_autodiff(ptr @f, double %x, double %y)
  ret { double, double } %r
}

; CHECK: define internal { double, double } @diffef(double %x, double %y, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %xi = bitcast double %x to i64
; CHECK-NEXT:   %yi = bitcast double %y to i64
; CHECK-NEXT:   %0 = icmp ult i64 %xi, %yi
; CHECK-NEXT:   %1 = bitcast double %differeturn to i64
; CHECK-NEXT:   %2 = select i1 %0, i64 0, i64 %1
; CHECK-NEXT:   %3 = icmp ult i64 %xi, %yi
; CHECK-NEXT:   %4 = bitcast double %differeturn to i64
; CHECK-NEXT:   %5 = select i1 %3, i64 %4, i64 0
; CHECK-NEXT:   %6 = bitcast i64 %5 to double
; CHECK-NEXT:   %7 = bitcast i64 %2 to double
; CHECK-NEXT:   %8 = insertvalue { double, double } undef, double %7, 0
; CHECK-NEXT:   %9 = insertvalue { double, double } %8, double %6, 1
; CHECK-NEXT:   ret { double, double } %9
; CHECK-NEXT: }
